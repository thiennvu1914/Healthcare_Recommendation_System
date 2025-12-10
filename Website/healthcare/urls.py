from django.urls import path
from . import views

app_name = 'healthcare'

urlpatterns = [
    # Web views
    path('', views.HomeView.as_view(), name='home'),
    path('search/', views.SearchView.as_view(), name='search'),
    path('qa/<int:qa_id>/', views.QADetailView.as_view(), name='qa_detail'),
    path('article/<int:pk>/', views.ArticleDetailView.as_view(), name='article_detail'),
    path('articles/', views.ArticleListView.as_view(), name='article_list'),
    path('topic/<str:topic>/', views.TopicListView.as_view(), name='topic_list'),
    path('ai-advisor/', views.AIAdvisorView.as_view(), name='ai_advisor'),
    
    # API endpoints
    path('api/search/', views.api_search, name='api_search'),
    path('api/qa/<int:qa_id>/', views.api_qa_detail, name='api_qa_detail'),
    path('api/topic/<str:topic>/', views.api_topic_list, name='api_topic_list'),
    path('api/articles/', views.api_articles, name='api_articles'),
    path('api/recommend/', views.api_recommend, name='api_recommend'),
]
