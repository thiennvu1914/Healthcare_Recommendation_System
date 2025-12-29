# 🌐 Healthcare Chatbot Web Interface

Web application Django để tương tác với Healthcare RAG API.

## 🚀 Tính năng

- ✅ **Chat Interface** đẹp mắt, responsive
- ✅ **Real-time messaging** với typing indicator
- ✅ **Hiển thị sources** (Q&A + Articles)
- ✅ **Confidence score** với visual indicator
- ✅ **Specialty badges** cho từng câu trả lời
- ✅ **Emergency detection** warnings
- ✅ **Session management** cho lịch sử chat
- ✅ **Bootstrap 5** UI hiện đại

## 📋 Yêu cầu

- Python 3.9+
- Django 4.2+
- Healthcare RAG API đang chạy (mặc định: `http://localhost:8000`)

## 🛠️ Cài đặt

### 1. Di chuyển vào thư mục web

```bash
cd web
```

### 2. Tạo virtual environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Migrate database

```bash
python manage.py migrate
```

### 5. (Optional) Tạo superuser để truy cập admin

```bash
python manage.py createsuperuser
```

## 🚀 Chạy Web Application

### 1. Đảm bảo API backend đang chạy

```bash
# Trong terminal khác, ở thư mục gốc
cd ..
uvicorn api.main:app --reload
```

API sẽ chạy tại: `http://localhost:8000`

### 2. Chạy Django development server

```bash
# Trong thư mục web/
python manage.py runserver
```

Web app sẽ chạy tại: **http://localhost:8000** (Django default port)

⚠️ **Lưu ý:** Port mặc định của Django là 8000, trùng với API. Có 2 cách:

**Cách 1: Chạy Django trên port khác**
```bash
python manage.py runserver 8080
```
Web: `http://localhost:8080`

**Cách 2: Chạy API trên port khác**
```bash
uvicorn api.main:app --port 8001
```

Sau đó set environment variable:
```bash
# Windows CMD
set HEALTHCARE_API_URL=http://localhost:8001

# Windows PowerShell
$env:HEALTHCARE_API_URL="http://localhost:8001"

# Linux/Mac
export HEALTHCARE_API_URL=http://localhost:8001
```

## 📁 Cấu trúc project

```
web/
├── manage.py                 # Django management script
├── healthcare_web/           # Main project folder
│   ├── __init__.py
│   ├── settings.py          # ✅ Cấu hình Django
│   ├── urls.py              # URL routing chính
│   ├── wsgi.py              # WSGI entry point
│   └── asgi.py              # ASGI entry point
├── chatbot/                  # Chatbot app
│   ├── __init__.py
│   ├── models.py            # Models (ChatHistory)
│   ├── views.py             # ✅ Business logic
│   ├── urls.py              # App URLs
│   └── admin.py             # Django admin
├── templates/
│   └── chatbot/
│       └── index.html       # ✅ Main chat interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎨 Features Details

### 1. Chat Interface

- **Modern UI** với Bootstrap 5 + custom CSS
- **Gradient background** đẹp mắt
- **Message bubbles** khác biệt cho user/bot
- **Typing indicator** khi đang chờ response
- **Auto-scroll** đến tin nhắn mới nhất

### 2. Response Display

```python
# Mỗi response hiển thị:
- Answer text (với line breaks)
- Specialty badge (e.g., "Nhi khoa")
- Confidence score (progress bar + %)
- Sources:
  * Q&A references
  * Article links
```

### 3. API Integration

```python
# views.py - Gọi backend API
def chat_api(request):
    api_url = f"{settings.HEALTHCARE_API_URL}/api/chat"
    response = requests.post(api_url, json={'query': query})
    return JsonResponse(response.json())
```

### 4. Error Handling

- ✅ Connection errors (API không chạy)
- ✅ Timeout handling
- ✅ Validation (min 5, max 500 chars)
- ✅ Display friendly error messages

## 🔧 Configuration

### Environment Variables

Tạo file `.env` trong `web/`:

```bash
# API Backend URL
HEALTHCARE_API_URL=http://localhost:8000

# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here

# Database (optional, default SQLite)
DATABASE_URL=sqlite:///db.sqlite3
```

### Settings.py

```python
# healthcare_web/settings.py

# API Configuration
HEALTHCARE_API_URL = os.getenv('HEALTHCARE_API_URL', 'http://localhost:8000')

# Language
LANGUAGE_CODE = 'vi-VN'
TIME_ZONE = 'Asia/Ho_Chi_Minh'
```

## 📊 Admin Panel

Truy cập admin tại: `http://localhost:8080/admin/`

Features:
- View chat history
- Filter by specialty, date
- Search queries
- Export data

## 🚢 Production Deployment

### 1. Collect Static Files

```bash
python manage.py collectstatic --nolint
```

### 2. Update Settings

```python
# healthcare_web/settings.py
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']

# Use WhiteNoise for static files
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
```

### 3. Run with Gunicorn

```bash
gunicorn healthcare_web.wsgi:application --bind 0.0.0.0:8080 --workers 4
```

### 4. Nginx Configuration (example)

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/web/staticfiles/;
    }
}
```

## 🧪 Testing

```bash
# Test chat endpoint
curl -X POST http://localhost:8080/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Bị đau đầu phải làm sao?"}'

# Test health check
curl http://localhost:8080/api/health/
```

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối đến API server"

**Nguyên nhân:** API backend chưa chạy

**Giải pháp:**
```bash
# Terminal 1: Chạy API
cd ..
uvicorn api.main:app --reload

# Terminal 2: Chạy Web
cd web
python manage.py runserver 8080
```

### Lỗi: "CSRF verification failed"

**Nguyên nhân:** Missing CSRF token

**Giải pháp:** Đã handle trong template với `getCookie('csrftoken')`

### Lỗi: Port already in use

**Giải pháp:**
```bash
# Dùng port khác
python manage.py runserver 8080
```

## 📝 Development Tips

### Hot Reload

Django development server tự động reload khi code thay đổi:
- Edit `views.py` → auto reload
- Edit `index.html` → refresh browser
- Edit `models.py` → cần migrate

### Debug Mode

Enable debug toolbar:

```python
# settings.py
INSTALLED_APPS += ['debug_toolbar']
MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
INTERNAL_IPS = ['127.0.0.1']
```

### Logging

```python
# views.py
import logging
logger = logging.getLogger(__name__)

logger.info(f"User query: {query}")
logger.error(f"API error: {error}")
```

## 🔮 Future Enhancements

- [ ] User authentication
- [ ] Save chat history to database
- [ ] Export chat to PDF
- [ ] Voice input
- [ ] Multi-language support
- [ ] Dark mode
- [ ] Chat analytics dashboard

## 📞 Support

Nếu gặp vấn đề:
1. Check API logs: `uvicorn api.main:app --reload`
2. Check Django logs: terminal output
3. Check browser console: F12 → Console tab
4. Check `/api/health/` endpoint

---

**Tạo bởi:** Healthcare AI Team  
**Last updated:** December 29, 2025  
**Version:** 1.0.0
