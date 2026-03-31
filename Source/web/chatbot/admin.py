from django.contrib import admin
from .models import ChatHistory


@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('query_short', 'specialty', 'confidence', 'created_at')
    list_filter = ('specialty', 'created_at')
    search_fields = ('query', 'answer')
    readonly_fields = ('created_at',)
    
    def query_short(self, obj):
        return obj.query[:100] + ('...' if len(obj.query) > 100 else '')
    query_short.short_description = 'Câu hỏi'
