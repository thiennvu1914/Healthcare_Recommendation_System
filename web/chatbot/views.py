from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.paginator import Paginator
from django.db.models import Q, Count
from .models import Article, QA, ChatHistory
import requests
import json
import uuid


def index(request):
    """Homepage with chatbot interface"""
    # Generate session ID if not exists
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    context = {
        'session_id': request.session['session_id']
    }
    return render(request, 'chatbot/index.html', context)


@csrf_exempt
def chat_api(request):
    """Handle chat requests and call backend API"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({'error': 'Câu hỏi không được để trống'}, status=400)
        
        if len(query) < 5:
            return JsonResponse({'error': 'Câu hỏi quá ngắn (tối thiểu 5 ký tự)'}, status=400)
        
        if len(query) > 500:
            return JsonResponse({'error': 'Câu hỏi quá dài (tối đa 500 ký tự)'}, status=400)
        
        # Call backend API
        api_url = f"{settings.HEALTHCARE_API_URL}/api/chat"
        
        response = requests.post(
            api_url,
            json={'query': query, 'include_sources': True},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save to database (optional)
            # from .models import ChatHistory
            # ChatHistory.objects.create(
            #     session_id=request.session.get('session_id'),
            #     query=query,
            #     answer=result.get('answer', ''),
            #     specialty=result.get('specialty', ''),
            #     confidence=result.get('confidence', 0.0)
            # )
            
            return JsonResponse(result)
        else:
            error_msg = f"API Error: {response.status_code}"
            try:
                error_detail = response.json().get('detail', error_msg)
            except:
                error_detail = error_msg
            
            return JsonResponse({'error': error_detail}, status=response.status_code)
    
    except requests.exceptions.ConnectionError:
        return JsonResponse({
            'error': 'Không thể kết nối đến API server. Vui lòng đảm bảo API đang chạy tại ' + settings.HEALTHCARE_API_URL
        }, status=503)
    
    except requests.exceptions.Timeout:
        return JsonResponse({
            'error': 'Request timeout. Vui lòng thử lại.'
        }, status=504)
    
    except Exception as e:
        return JsonResponse({
            'error': f'Đã xảy ra lỗi: {str(e)}'
        }, status=500)


def health_check(request):
    """Check backend API health"""
    try:
        api_url = f"{settings.HEALTHCARE_API_URL}/api/health"
        response = requests.get(api_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return JsonResponse({
                'status': 'healthy',
                'backend': data,
                'api_url': settings.HEALTHCARE_API_URL
            })
        else:
            return JsonResponse({
                'status': 'unhealthy',
                'error': f'Backend returned {response.status_code}'
            }, status=503)
    
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=503)


def specialties(request):
    """Get available specialties from backend"""
    try:
        api_url = f"{settings.HEALTHCARE_API_URL}/api/specialties"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            return JsonResponse(response.json())
        else:
            return JsonResponse({'error': 'Failed to fetch specialties'}, status=response.status_code)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ============================================
# NEW VIEWS FOR ENHANCED PORTAL
# ============================================

def home(request):
    """Enhanced homepage with stats and features"""
    stats = {
        'total_articles': Article.objects.count(),
        'total_qas': QA.objects.count(),
        'total_topics': QA.objects.values('topic').distinct().count(),
    }
    
    recent_articles = Article.objects.all()[:6]
    popular_topics = QA.objects.values('topic').annotate(count=Count('id')).order_by('-count')[:8]
    
    context = {
        'stats': stats,
        'recent_articles': recent_articles,
        'popular_topics': popular_topics,
    }
    return render(request, 'chatbot/home.html', context)


def ai_advisor(request):
    """AI Chatbot page"""
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    context = {
        'session_id': request.session['session_id']
    }
    return render(request, 'chatbot/ai_advisor.html', context)


def article_list(request):
    """List all articles with pagination and search"""
    articles = Article.objects.all()
    
    # Search
    query = request.GET.get('q', '').strip()
    if query:
        articles = articles.filter(
            Q(title__icontains=query) | Q(text__icontains=query)
        )
    
    # Pagination
    paginator = Paginator(articles, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'query': query,
        'total': articles.count(),
    }
    return render(request, 'chatbot/article_list.html', context)


def article_detail(request, article_id):
    """Article detail page"""
    article = get_object_or_404(Article, article_id=article_id)
    
    # Increment views
    article.views += 1
    article.save(update_fields=['views'])
    
    # Extract URL from article text if present
    import re
    article_url = None
    if article.text:
        url_match = re.search(r'https?://[^\s]+', article.text)
        if url_match:
            article_url = url_match.group(0)
    
    # Related articles
    related = Article.objects.exclude(id=article.id)[:5]
    
    context = {
        'article': article,
        'article_url': article_url,
        'related': related,
    }
    return render(request, 'chatbot/article_detail.html', context)


def topic_list(request):
    """List Q&A by topics"""
    topic = request.GET.get('topic', '').strip()
    
    if topic:
        qas = QA.objects.filter(topic=topic)
    else:
        qas = QA.objects.all()
    
    # Search within topic
    query = request.GET.get('q', '').strip()
    if query:
        qas = qas.filter(
            Q(question__icontains=query) | Q(answer__icontains=query)
        )
    
    # Get all topics for sidebar
    topics = QA.objects.values('topic').annotate(count=Count('id')).order_by('-count')
    
    # Pagination
    paginator = Paginator(qas, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'topics': topics,
        'current_topic': topic,
        'query': query,
        'total': qas.count(),
    }
    return render(request, 'chatbot/topic_list.html', context)


def qa_detail(request, qa_id):
    """Q&A detail page"""
    qa = get_object_or_404(QA, qa_id=qa_id)
    
    # Increment views
    qa.views += 1
    qa.save(update_fields=['views'])
    
    # Related Q&A in same topic
    related = QA.objects.filter(topic=qa.topic).exclude(id=qa.id)[:5]
    
    context = {
        'qa': qa,
        'related': related,
    }
    return render(request, 'chatbot/qa_detail.html', context)


def search(request):
    """Unified search across articles and Q&A"""
    query = request.GET.get('q', '').strip()
    search_type = request.GET.get('type', 'all')  # all, articles, qa
    
    results = {
        'articles': [],
        'qas': [],
        'query': query,
    }
    
    if query:
        if search_type in ['all', 'articles']:
            results['articles'] = Article.objects.filter(
                Q(title__icontains=query) | Q(text__icontains=query)
            )[:20]
        
        if search_type in ['all', 'qa']:
            results['qas'] = QA.objects.filter(
                Q(question__icontains=query) | Q(answer__icontains=query)
            )[:20]
    
    context = {
        'query': query,
        'search_type': search_type,
        'articles': results['articles'],
        'qas': results['qas'],
        'total_articles': len(results['articles']),
        'total_qas': len(results['qas']),
    }
    return render(request, 'chatbot/search.html', context)
