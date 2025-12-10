# Healthcare Recommendation System

Hệ thống tư vấn sức khỏe thông minh với AI/RAG - Django 4.2.7

## 🎯 Tính năng

- **AI Advisor**: Trợ lý AI với RAG (Retrieval-Augmented Generation)
- **Tìm kiếm**: 73,598 Q&As + 378 bài viết y tế
- **Phân loại**: 9 chuyên khoa (Nhi, Tim mạch, Tiêu hóa...)
- **API**: RESTful endpoints

## 🚀 Quick Start

### 1. Cài đặt

```bash
cd Website
pip install -r requirements.txt
```

### 2. Setup Database

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py import_data
```

### 3. Chạy Server

```bash
python manage.py runserver
```

**URLs:**
- Trang chủ: http://localhost:8000/
- AI Advisor: http://localhost:8000/ai-advisor/
- Admin: http://localhost:8000/admin/

## 🤖 AI/RAG Service

**Pipeline:**
1. **Retrieve**: TF-IDF + Cosine Similarity → Top K Q&As
2. **Generate**: Template-based answer synthesis
3. **Specialty Detection**: Tự động gợi ý chuyên khoa

**API:**
```
GET /api/recommend/?query=<câu_hỏi>&mode=rag&top_k=5
```

## 📁 Cấu Trúc

```
Website/
├── healthcare/
│   ├── models.py           # Article, QuestionAnswer, SearchQuery
│   ├── views.py            # Web + API views
│   ├── rag_service.py      # RAG Service
│   └── templates/
├── db.sqlite3              # 73,598 Q&As + 378 Articles
└── requirements.txt
```

## 🛠️ Tech Stack

Django 4.2.7 · scikit-learn · Bootstrap 5.3 · SQLite
