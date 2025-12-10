from django.contrib import admin
from .models import Article, QuestionAnswer, SearchQuery

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at', 'updated_at')
    search_fields = ('title', 'content')
    list_filter = ('created_at', 'updated_at')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(QuestionAnswer)
class QuestionAnswerAdmin(admin.ModelAdmin):
    list_display = ('qa_id', 'topic', 'created_at')
    search_fields = ('question', 'answer')
    list_filter = ('topic', 'created_at')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ('query', 'timestamp', 'user_ip')
    search_fields = ('query',)
    list_filter = ('timestamp',)
    readonly_fields = ('query', 'timestamp', 'user_ip')

