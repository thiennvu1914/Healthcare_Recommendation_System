# Healthcare RAG API

API backend cho hệ thống tư vấn sức khỏe thông minh sử dụng RAG (Retrieval-Augmented Generation).

## 🚀 Tính năng

- ✅ **REST API** với FastAPI
- ✅ **Tìm kiếm ngữ nghĩa** với PhoBERT + HNSW
- ✅ **Tích hợp Q&A Database** (60K+ câu hỏi)
- ✅ **Tích hợp Articles** (200K+ bài viết y tế)
- ✅ **Index Caching** - khởi động nhanh 96x (5 giây vs 8 phút) 🚀
- ✅ **Medical Safety Guardrails** - phát hiện tình huống khẩn cấp
- ✅ **LLM Generation** với Vistral-7B-Chat
- ✅ **CORS** cho web frontend
- ✅ **Auto Documentation** tại `/docs`
- ✅ **GPU/CPU Support** tự động

## ⚡ Performance Improvements

**vs Notebook gốc:**
- **Startup time:** 8 phút → 5 giây (với cache enabled) 🚀
- **No CLI blocking** - production-ready REST API
- **Environment-based config** - không hard-code paths
- **Emergency detection** - 6 categories với response ưu tiên

Chi tiết: Xem [FIXES_REPORT.md](FIXES_REPORT.md)

## 📋 Yêu cầu hệ thống

- Python 3.9+
- RAM: 8GB+ (16GB khuyến nghị)
- GPU: Optional (CUDA 11.8+ nếu có)
- Disk: 5GB+ cho models + 2GB cho cache

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone <your-repo>
cd Healthcare_Recommendation_System
```

### 2. Tạo virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

```bash
# Copy file .env.example
cp .env.example .env

# Chỉnh sửa .env theo nhu cầu
notepad .env  # Windows
nano .env     # Linux/Mac
```

**Recommended settings:**
```env
# Enable cache để khởi động nhanh (5s vs 8 phút)
ENABLE_CACHE=1

# Force CPU nếu không có GPU
FORCE_CPU=0

# Sample size (tăng = chính xác hơn, chậm hơn)
SAMPLE_SIZE=5000
```

### 5. Đảm bảo dữ liệu có sẵn

Kiểm tra các file trong thư mục `data/`:
- `QAs.csv`
- `articles.csv`
- `rag_gold_eval_semantic.json` (optional)

## 🚀 Chạy API

### Development mode

```bash
# Lần đầu: ~8 phút (build + cache indices)
# Lần sau: ~5 giây (load từ cache) 🚀
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production mode

```bash
# Với gunicorn (Linux/Mac)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Với uvicorn (Windows)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Performance tips:**
- ✅ Để `ENABLE_CACHE=1` (recommended)
- ✅ Clear cache nếu data thay đổi: `rm -rf cache/`
- ✅ Dùng GPU nếu có (tăng tốc 3-5x)

API sẽ chạy tại: `http://localhost:8000`

## 📚 API Endpoints

### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "gpu_available": true
}
```

### 2. Chat (Hỏi đáp)
```http
POST /api/chat
Content-Type: application/json

{
  "query": "Bé 2 tuổi sốt 38.5 độ, tôi phải làm gì?",
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "Chào bạn, dựa trên thông tin y tế...",
  "specialty": "Nhi Khoa",
  "confidence": 0.85,
  "sources": [
    {
      "type": "qa",
      "question": "Bé tôi sốt cao phải làm sao?",
      "score": 0.89,
      "snippet": "Sốt 38.5°C ở trẻ nhỏ..."
    }
  ]
}
```

### 3. Danh sách chuyên khoa
```http
GET /api/specialties
```

**Response:**
```json
{
  "specialties": [
    {"name": "Nhi Khoa", "count": 15234},
    {"name": "Tim Mạch", "count": 8765}
  ],
  "total": 25
}
```

### 4. Thống kê hệ thống
```http
GET /api/stats
```

## 🧪 Testing

### Sử dụng curl

```bash
# Health check
curl http://localhost:8000/api/health

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Bé sốt 38 độ phải làm sao?"}'
```

### Sử dụng Python

```python
import requests

# Chat request
response = requests.post(
    "http://localhost:8000/api/chat",
    json={
        "query": "Bé 2 tuổi sốt 38.5 độ, tôi phải làm gì?",
        "include_sources": True
    }
)

print(response.json())
```

### Sử dụng Swagger UI

Truy cập: `http://localhost:8000/docs`

## 🏗️ Cấu trúc dự án

```
Healthcare_Recommendation_System/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   ├── rag_engine.py     # Core RAG logic
│   └── config.py         # Configuration
├── data/
│   ├── QAs.csv
│   ├── articles.csv
│   └── rag_gold_eval_semantic.json
├── requirements.txt
├── .env.example
└── README_API.md
```

## ⚙️ Configuration

Các biến môi trường trong `.env`:

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `FORCE_CPU` | Bắt buộc dùng CPU | `0` |
| `SAMPLE_SIZE` | Số lượng mẫu load | `5000` |
| `TOP_K_QA` | Số Q&A trả về | `5` |
| `TOP_K_ARTICLES` | Số bài viết trả về | `1` |
| `CORS_ORIGINS` | Allowed origins | `http://localhost:3000` |

## 🐳 Docker Deployment (Optional)

### Tạo Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build và Run

```bash
docker build -t healthcare-rag-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data healthcare-rag-api
```

## 📊 Performance

### Thời gian khởi động
- CPU: ~30-60 giây
- GPU: ~15-30 giây

### Thời gian response
- Retrieval: 50-200ms
- Generation: 500-2000ms (CPU) / 100-500ms (GPU)

### Resource usage
- RAM: 4-8GB (tùy SAMPLE_SIZE)
- GPU VRAM: 4-6GB (nếu dùng GPU)

## 🔧 Troubleshooting

### Lỗi: "CUDA out of memory"
```bash
# Trong .env
FORCE_CPU=1
```

### Lỗi: "Models not loaded"
```bash
# Kiểm tra kết nối internet (download models)
# Hoặc pre-download models:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/phobert-base')"
```

### Lỗi: "Data files not found"
```bash
# Kiểm tra đường dẫn trong config.py
# Đảm bảo QAs.csv và articles.csv trong thư mục data/
```

## 🌐 Web Demo Integration

### React Example

```javascript
const askQuestion = async (query) => {
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, include_sources: true })
  });
  
  const data = await response.json();
  return data;
};
```

### Vue.js Example

```javascript
export default {
  methods: {
    async askQuestion(query) {
      const res = await this.$http.post('/api/chat', {
        query: query,
        include_sources: true
      });
      return res.data;
    }
  }
}
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

## 📧 Contact

For questions or support, please contact: [your-email]

---

**Lưu ý:** API này chỉ mang tính chất tham khảo, không thay thế cho tư vấn y tế chuyên nghiệp.
