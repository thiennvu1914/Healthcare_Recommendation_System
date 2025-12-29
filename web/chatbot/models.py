from django.db import models
from django.utils import timezone
from django.utils.text import slugify


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


class Article(models.Model):
    """Medical articles"""
    article_id = models.CharField(max_length=50, unique=True, db_index=True)
    title = models.CharField(max_length=500, db_index=True)
    text = models.TextField()
    slug = models.SlugField(max_length=500, blank=True)
    views = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Bài viết"
        verbose_name_plural = "Bài viết"
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['-created_at']),
        ]
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)[:500]
        super().save(*args, **kwargs)
    
    def __str__(self):
        return self.title[:100]


class QA(models.Model):
    """Q&A pairs"""
    qa_id = models.CharField(max_length=50, unique=True, db_index=True)
    question = models.TextField(db_index=True)
    answer = models.TextField()
    topic = models.CharField(max_length=200, db_index=True)
    topic_original = models.CharField(max_length=200, blank=True)
    views = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Câu hỏi - Trả lời"
        verbose_name_plural = "Câu hỏi - Trả lời"
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"{self.question[:100]}... ({self.topic})"
