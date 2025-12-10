from django.db import models

class Article(models.Model):
    """Mô hình lưu trữ bài viết y tế từ Bloomax"""
    link = models.URLField(unique=True)
    title = models.CharField(max_length=500)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return self.title


class QuestionAnswer(models.Model):
    """Mô hình lưu trữ cặp câu hỏi-trả lời y tế"""
    TOPIC_CHOICES = [
        ('chỉnh hình', 'Chỉnh hình'),
        ('nhi khoa', 'Nhi khoa'),
        ('tim mạch', 'Tim mạch'),
        ('tiêu hóa', 'Tiêu hóa'),
        ('hô hấp', 'Hô hấp'),
        ('thần kinh', 'Thần kinh'),
        ('da liễu', 'Da liễu'),
        ('sản phụ khoa', 'Sản phụ khoa'),
        ('y tế chung', 'Y tế chung'),
    ]
    
    qa_id = models.IntegerField(unique=True)
    question = models.TextField()
    answer = models.TextField()
    topic = models.CharField(max_length=50, choices=TOPIC_CHOICES, default='y tế chung')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['topic']),
            models.Index(fields=['qa_id']),
        ]
    
    def __str__(self):
        return f"Q{self.qa_id}: {self.question[:100]}..."


class SearchQuery(models.Model):
    """Lưu trữ lịch sử tìm kiếm để phân tích"""
    query = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.query} - {self.timestamp}"

