from django.db import models
from django.utils import timezone


class ChatHistory(models.Model):
    """Store chat conversation history"""
    session_id = models.CharField(max_length=100, db_index=True)
    query = models.TextField(verbose_name="Câu hỏi")
    answer = models.TextField(verbose_name="Câu trả lời")
    specialty = models.CharField(max_length=200, verbose_name="Chuyên khoa", blank=True)
    confidence = models.FloatField(verbose_name="Độ tin cậy", default=0.0)
    created_at = models.DateTimeField(default=timezone.now, verbose_name="Thời gian")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Lịch sử chat"
        verbose_name_plural = "Lịch sử chat"
    
    def __str__(self):
        return f"{self.query[:50]}... ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
