from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    # Homepage
    path('', views.home, name='home'),
    
    # AI Chatbot
    path('ai-advisor/', views.ai_advisor, name='ai_advisor'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/health/', views.health_check, name='health_check'),
    path('api/specialties/', views.specialties, name='specialties'),
    
    # Articles
    path('articles/', views.article_list, name='article_list'),
    path('articles/<str:article_id>/', views.article_detail, name='article_detail'),
    
    # Q&A / Topics
    path('topics/', views.topic_list, name='topic_list'),
    path('qa/<str:qa_id>/', views.qa_detail, name='qa_detail'),
    
    # Search
    path('search/', views.search, name='search'),
]
