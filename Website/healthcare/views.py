from django.shortcuts import render
from django.views.generic import ListView, DetailView, TemplateView
from django.db.models import Q
from django.http import JsonResponse
from .models import Article, QuestionAnswer, SearchQuery
from difflib import SequenceMatcher
import json


def calculate_relevance(query, text):
    """Tính điểm độ liên quan giữa query và text"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Kiểm tra nếu query là substring của text
    if query_lower in text_lower:
        return 1.0
    
    # Tính toán similarity
    similarity = SequenceMatcher(None, query_lower, text_lower).ratio()
    return similarity


def extract_keywords(text):
    """Trích xuất các từ khóa từ text"""
    # Loại bỏ các từ dừng phổ biến
    stop_words = {'và', 'hoặc', 'là', 'có', 'được', 'trong', 'của', 'để', 'từ', 'đến', 'cái', 'chiếc'}
    words = text.lower().split()
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    return keywords

class HomeView(TemplateView):
    """Trang chủ - hiển thị hệ thống tìm kiếm"""
    template_name = 'healthcare/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['total_qa'] = QuestionAnswer.objects.count()
        context['total_articles'] = Article.objects.count()
        # Thêm bài viết gợi ý trên trang chủ
        context['featured_articles'] = Article.objects.all()[:6]
        return context


class SearchView(TemplateView):
    """Trang tìm kiếm câu hỏi-trả lời và bài viết với xếp hạng độ liên quan"""
    template_name = 'healthcare/search.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        query = self.request.GET.get('q', '').strip()
        
        if query:
            # Lưu lịch sử tìm kiếm
            SearchQuery.objects.create(
                query=query,
                user_ip=self.get_client_ip()
            )
            
            # Tìm kiếm câu hỏi-trả lời với xếp hạng
            all_qas = QuestionAnswer.objects.filter(
                Q(question__icontains=query) | Q(answer__icontains=query)
            )
            
            # Xếp hạng theo độ liên quan
            qa_scores = []
            for qa in all_qas:
                score = calculate_relevance(query, qa.question)
                score += calculate_relevance(query, qa.answer) * 0.5
                qa_scores.append((qa, score))
            
            # Sắp xếp theo điểm và lấy top 15
            qa_scores.sort(key=lambda x: x[1], reverse=True)
            qas = [qa for qa, _ in qa_scores[:15]]
            
            # Tìm kiếm bài viết với xếp hạng
            all_articles = Article.objects.filter(
                Q(title__icontains=query) | Q(content__icontains=query)
            )
            
            article_scores = []
            for article in all_articles:
                score = calculate_relevance(query, article.title) * 2
                score += calculate_relevance(query, article.content) * 0.3
                article_scores.append((article, score))
            
            article_scores.sort(key=lambda x: x[1], reverse=True)
            articles = [article for article, _ in article_scores[:10]]
            
            # Phân tích query để gợi ý chuyên khoa
            keywords = extract_keywords(query)
            suggested_topic = self._suggest_topic(keywords)
            
            context['query'] = query
            context['qas'] = qas
            context['articles'] = articles
            context['suggested_topic'] = suggested_topic
            context['qa_count'] = all_qas.count()
            context['article_count'] = all_articles.count()
        
        context['topics'] = [choice[0] for choice in QuestionAnswer.TOPIC_CHOICES]
        return context
    
    def _suggest_topic(self, keywords):
        """Gợi ý chuyên khoa dựa trên keywords"""
        topic_keywords = {
            'chỉnh hình': ['xương', 'khớp', 'gãy', 'đau', 'chân', 'tay', 'cột sống'],
            'nhi khoa': ['bé', 'trẻ', 'em', 'bé sơ sinh', 'trẻ sơ sinh'],
            'tim mạch': ['tim', 'nhịp', 'huyết áp', 'mạch', 'tim mạch'],
            'tiêu hóa': ['dạ dày', 'ăn', 'tiêu', 'ruột', 'gan', 'tụy'],
            'hô hấp': ['phổi', 'thở', 'ho', 'cảm lạnh', 'asthma'],
        }
        
        for topic, topic_kws in topic_keywords.items():
            for kw in keywords:
                for tkw in topic_kws:
                    if kw in tkw or tkw in kw:
                        return topic
        
        return None
    
    def get_client_ip(self):
        """Lấy IP của client"""
        x_forwarded_for = self.request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = self.request.META.get('REMOTE_ADDR')
        return ip


class QADetailView(DetailView):
    """Hiển thị chi tiết câu hỏi-trả lời"""
    model = QuestionAnswer
    template_name = 'healthcare/qa_detail.html'
    context_object_name = 'qa'
    
    def get_object(self):
        return QuestionAnswer.objects.get(qa_id=self.kwargs['qa_id'])


class ArticleDetailView(DetailView):
    """Hiển thị chi tiết bài viết"""
    model = Article
    template_name = 'healthcare/article_detail.html'
    context_object_name = 'article'


class TopicListView(ListView):
    """Liệt kê câu hỏi-trả lời theo chuyên khoa"""
    model = QuestionAnswer
    template_name = 'healthcare/topic_list.html'
    context_object_name = 'qas'
    paginate_by = 20
    
    def get_queryset(self):
        topic = self.kwargs['topic']
        return QuestionAnswer.objects.filter(topic=topic)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['topic'] = self.kwargs['topic']
        return context


class AIAdvisorView(TemplateView):
    """View cho trang AI Advisor - Trợ lý sức khỏe thông minh"""
    template_name = 'healthcare/ai_advisor.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Trợ Lý Sức Khỏe AI'
        
        # Sample queries để gợi ý cho user
        context['sample_queries'] = [
            'Bé bị sốt 38 độ phải làm sao?',
            'Đau đầu và buồn nôn là bệnh gì?',
            'Bị đau khớp gối khi đi bộ',
            'Có thai nên ăn gì tốt cho bé?'
        ]
        
        return context


# API Views
def api_search(request):
    """API endpoint tìm kiếm với xếp hạng độ liên quan"""
    query = request.GET.get('q', '').strip()
    
    if not query or len(query) < 2:
        return JsonResponse({'error': 'Query quá ngắn'}, status=400)
    
    # Tìm kiếm Q&A
    all_qas = QuestionAnswer.objects.filter(
        Q(question__icontains=query) | Q(answer__icontains=query)
    )
    
    qa_scores = []
    for qa in all_qas:
        score = calculate_relevance(query, qa.question)
        score += calculate_relevance(query, qa.answer) * 0.5
        qa_scores.append({
            'qa_id': qa.qa_id,
            'question': qa.question,
            'topic': qa.topic,
            'score': score
        })
    
    qa_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Tìm kiếm Articles
    all_articles = Article.objects.filter(
        Q(title__icontains=query) | Q(content__icontains=query)
    )
    
    article_scores = []
    for article in all_articles:
        score = calculate_relevance(query, article.title) * 2
        score += calculate_relevance(query, article.content) * 0.3
        article_scores.append({
            'id': article.id,
            'title': article.title,
            'link': article.link,
            'score': score
        })
    
    article_scores.sort(key=lambda x: x['score'], reverse=True)
    
    return JsonResponse({
        'query': query,
        'qas': qa_scores[:20],
        'articles': article_scores[:10],
        'total_qas': len(qa_scores),
        'total_articles': len(article_scores)
    })


def api_qa_detail(request, qa_id):
    """API endpoint chi tiết Q&A"""
    try:
        qa = QuestionAnswer.objects.get(qa_id=qa_id)
        return JsonResponse({
            'qa_id': qa.qa_id,
            'question': qa.question,
            'answer': qa.answer,
            'topic': qa.topic
        })
    except QuestionAnswer.DoesNotExist:
        return JsonResponse({'error': 'Không tìm thấy'}, status=404)


def api_recommend(request):
    """
    API endpoint cho AI Recommendation với RAG (Retrieval-Augmented Generation)
    Uses: PhoBERT/TF-IDF Retrieval + Gemini LLM Generation
    
    Parameters:
    - query: câu hỏi của user
    - mode: rag | simple (default: rag)
    - top_k: số lượng context Q&As (default: 5)
    """
    from .rag_service import get_rag_service
    
    query = request.GET.get('query', '').strip()
    if not query:
        return JsonResponse({'error': 'Vui lòng nhập câu hỏi'}, status=400)
    
    mode = request.GET.get('mode', 'rag')
    top_k = int(request.GET.get('top_k', 5))
    
    # Get RAG service and generate response
    rag_service = get_rag_service()
    
    if mode == 'rag':
        # Full RAG pipeline: Retrieve + LLM Generate
        result = rag_service.generate_rag_response(query, top_k=top_k)
    else:
        # Simple retrieval only (fallback)
        result = {
            'query': query,
            'context_qas': rag_service.retrieve_context(query, top_k=top_k),
            'suggested_specialty': rag_service.suggest_specialty(query),
            'ai_answer': '',
            'used_llm': False
        }
    
    return JsonResponse(result)


def api_topic_list(request, topic):
    """API endpoint liệt kê theo chuyên khoa"""
    qas = QuestionAnswer.objects.filter(topic=topic).values(
        'qa_id', 'question', 'topic'
    )[:50]
    
    return JsonResponse({
        'topic': topic,
        'count': len(list(qas)),
        'qas': list(qas)
    })


class ArticleListView(ListView):
    """Liệt kê tất cả bài viết"""
    model = Article
    template_name = 'healthcare/article_list.html'
    context_object_name = 'articles'
    paginate_by = 20
    ordering = '-created_at'


def api_articles(request):
    """API endpoint danh sách bài viết"""
    articles = Article.objects.all().values('id', 'title', 'link', 'created_at')[:50]
    return JsonResponse({
        'count': Article.objects.count(),
        'articles': list(articles)
    })

