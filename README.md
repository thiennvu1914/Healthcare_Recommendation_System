# 🏥 Healthcare Recommendation System

Hệ thống tư vấn sức khỏe thông minh sử dụng công nghệ RAG (Retrieval-Augmented Generation) với AI.

## 📊 Tổng quan

- **88,590** bài viết y tế
- **60,234** câu hỏi đã được giải đáp
- **15** chuyên khoa y tế
- Công nghệ: **PhoBERT** + **Vistral-7B-Chat** + **FAISS**
- GPU: Tối ưu cho **NVIDIA H200** (hỗ trợ CPU fallback)

## 🏗️ Kiến trúc hệ thống

```
Healthcare_Recommendation_System/
├── api/                    # FastAPI Backend
│   ├── main.py            # API endpoints
│   ├── rag_engine.py      # RAG core logic
│   ├── models.py          # Pydantic schemas
│   └── config.py          # Settings
├── web/                   # Django Web Frontend
│   ├── chatbot/          # Main app
│   ├── templates/        # HTML templates
│   └── manage.py
├── data/                 # Datasets
│   ├── articles.csv      # 88,590 bài viết
│   └── QAs.csv          # 60,234 Q&As
├── cache/               # FAISS indices cache
├── scripts/             # Utility scripts
│   ├── rebuild_fast.py  # Rebuild FAISS với GPU
│   └── test_api.py      # API testing
└── requirements.txt
```

## ✨ Tính năng chính

### 🤖 API Backend (FastAPI)
- ✅ Tìm kiếm ngữ nghĩa với **PhoBERT embeddings**
- ✅ FAISS index với **HNSW** (fast retrieval)
- ✅ **LLM generation** với Vistral-7B-Chat
- ✅ Index caching - khởi động **5 giây** thay vì 8 phút
- ✅ GPU/CPU auto-detection
- ✅ CORS enabled cho web frontend
- ✅ Auto documentation tại `/docs`

### 🌐 Web Interface (Django)
- ✅ Chat UI hiện đại với **Bootstrap 5**
- ✅ Hiển thị **4 section**: Chuyên khoa, Câu trả lời, Sources, Disclaimer
- ✅ Source citations với **badges** (Q&A/Article)
- ✅ Modal chi tiết nguồn tham khảo
- ✅ Link trực tiếp đến bài viết gốc
- ✅ Permanent disclaimer ngoài chat
- ✅ Responsive design

## 🚀 Cài đặt & Khởi động

### 1. Clone repository

```bash
git clone <repository-url>
cd Healthcare_Recommendation_System
```

### 2. Cài đặt dependencies

```bash
# Backend API
pip install -r requirements.txt

# Web frontend
cd web
pip install -r requirements.txt
cd ..
```

### 3. Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env theo nhu cầu
```

**Recommended `.env` settings:**
```env
# Cache để khởi động nhanh (5s thay vì 8 phút)
ENABLE_CACHE=1

# LLM generation với Vistral-7B
ENABLE_LLM_GENERATION=1

# Sample size (0 = dùng toàn bộ data)
SAMPLE_SIZE=0

# HuggingFace token (nếu dùng gated models)
HUGGINGFACE_HUB_TOKEN=your_token_here
```

### 4. Khởi động API Backend

```bash
# Terminal 1: FastAPI (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Lần đầu: ~8 phút (build + cache FAISS indices)
# Lần sau: ~5 giây (load từ cache) 🚀
```

API sẽ chạy tại: **http://localhost:8000**
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

### 5. Khởi động Web Frontend

```bash
# Terminal 2: Django (port 8080)
cd web
python manage.py migrate
python manage.py runserver 0.0.0.0:8080
```

Web UI sẽ chạy tại: **http://localhost:8080**

## 📖 Sử dụng

### Chat với AI

1. Truy cập http://localhost:8080/ai-advisor/
2. Nhập câu hỏi về sức khỏe
3. Nhận câu trả lời từ AI với:
   - Chuyên khoa gợi ý
   - Câu trả lời tự nhiên (LLM-generated)
   - Top nguồn tham khảo (Q&A + Articles)
   - Click [Trích dẫn] để xem chi tiết

### API Endpoints

```bash
# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Bé 3 tuổi sốt 39 độ nên làm gì?","include_sources":true}'

# Health check
curl http://localhost:8000/api/health
```

**Response format:**
```json
{
  "answer": "Khi con bạn 3 tuổi sốt 39 độ, bước đầu tiên là...",
  "specialty": "Nhi Khoa",
  "confidence": 0.869,
  "sources": [
    {
      "type": "qa",
      "id": "qa_12345",
      "question": "...",
      "full_answer": "...",
      "score": 0.92
    },
    {
      "type": "article",
      "id": "https://...",
      "title": "...",
      "link": "https://...",
      "score": 0.85
    }
  ],
  "disclaimer": "Thông tin chỉ mang tính tham khảo..."
}
```

## 🛠️ Scripts hữu ích

### Rebuild FAISS Index (GPU-optimized)

```bash
python scripts/rebuild_fast.py
```

- Sử dụng GPU để tăng tốc
- Batch processing với fp16
- Rebuild cả QA + Article indices
- Lưu cache vào `cache/`

### Test API

```bash
python scripts/test_api.py
```

## 🔧 Cấu hình nâng cao

### API Settings (`api/config.py`)

```python
# Retrieval
TOP_K_QA = 5              # Số Q&A trả về
TOP_K_ARTICLES = 3        # Số Articles trả về
QUESTION_SIM_THRESHOLD = 0.3   # Ngưỡng similarity

# LLM Generation
MAX_NEW_TOKENS = 384      # Độ dài câu trả lời
TEMPERATURE = 0.8         # Tính sáng tạo
TOP_P = 0.92             # Nucleus sampling

# Performance
ENABLE_CACHE = 1          # Cache FAISS indices
SAMPLE_SIZE = 0           # 0 = dùng toàn bộ data
```

### Django Settings (`web/healthcare_web/settings.py`)

```python
# API endpoint
API_ENDPOINT = "http://localhost:8000"

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

## 📚 Chi tiết kỹ thuật

### RAG Pipeline

1. **Query Processing**: Tiền xử lý câu hỏi (lowercase, remove noise)
2. **Embedding**: PhoBERT embedding với mean-pooling
3. **Retrieval**: FAISS IndexFlatIP (cosine similarity)
   - Top-5 Q&As
   - Top-3 Articles
4. **Re-ranking**: Kết hợp semantic + lexical overlap
5. **Generation**: Vistral-7B-Chat tổng hợp câu trả lời tự nhiên
6. **Response**: JSON với specialty, answer, sources, disclaimer

### Models

- **PhoBERT** (`vinai/phobert-base`): Vietnamese BERT for embeddings
- **Vistral-7B-Chat** (`Viet-Mistral/Vistral-7B-Chat`): Vietnamese LLM for generation
- **FAISS**: IndexFlatIP cho cosine similarity search

### Performance

- **Startup**: 5s với cache (vs 8 phút không cache)
- **Query latency**: ~2-3s (bao gồm LLM generation)
- **VRAM usage**: ~4GB (PhoBERT + Vistral fp16)
- **Index size**: 
  - QA: 185MB (60,234 vectors)
  - Articles: 272MB (88,590 vectors)

## 🐛 Troubleshooting

### API không khởi động được

```bash
# Kiểm tra log
tail -f api.log

# Thử disable cache nếu indices bị lỗi
ENABLE_CACHE=0 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Rebuild indices
python scripts/rebuild_fast.py
```

### Web không kết nối được API

```bash
# Kiểm tra API health
curl http://localhost:8000/api/health

# Kiểm tra CORS settings trong api/config.py
CORS_ORIGINS = ["http://localhost:8080", "http://127.0.0.1:8080"]
```

### Out of memory (GPU)

```bash
# Chuyển sang CPU mode
FORCE_CPU=1 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Hoặc giảm sample size
SAMPLE_SIZE=5000 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Câu trả lời không tự nhiên

```bash
# Bật LLM generation
ENABLE_LLM_GENERATION=1 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Tăng temperature để câu trả lời đa dạng hơn
# Chỉnh trong api/config.py: TEMPERATURE = 0.8
```

## 📝 Changelog

### v2.0 (Latest)
- ✅ Tích hợp Vistral-7B-Chat cho câu trả lời tự nhiên
- ✅ UI 4-section: Specialty, Answer, Sources, Disclaimer
- ✅ Source citations với badges + modal chi tiết
- ✅ Link trực tiếp đến bài viết gốc
- ✅ Permanent disclaimer ngoài chat
- ✅ Full answer hiển thị cho Q&A
- ✅ GPU-optimized rebuild script
- ✅ Code cleanup + folder restructure

### v1.0
- ✅ FastAPI backend với PhoBERT
- ✅ FAISS caching
- ✅ Django web interface
- ✅ Basic chat functionality

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

**Lưu ý quan trọng:**
- Thông tin từ AI chỉ mang tính chất tham khảo, dựa trên cơ sở dữ liệu y tế.
- Bạn nên đi khám trực tiếp để được tư vấn chính xác hơn.
- Hệ thống này KHÔNG thay thế cho ý kiến của bác sĩ chuyên khoa.
- Trong trường hợp khẩn cấp, hãy gọi **115** hoặc đến bệnh viện ngay lập tức.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Created by DS300 - Group 12** | Technology: PhoBERT + Vistral-7B-Chat
