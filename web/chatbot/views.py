from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
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
