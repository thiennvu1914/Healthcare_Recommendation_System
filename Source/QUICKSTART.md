# 🚀 Quick Start Guide

## Khởi động hệ thống

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
cd /root/Healthcare_Recommendation_System
./START.sh
```

Script sẽ tự động:
- ✅ Kiểm tra data files
- ✅ Khởi động API backend (port 8000)
- ✅ Đợi API ready
- ✅ Khởi động Django web (port 8080)
- ✅ Hiển thị URLs và PIDs

### Cách 2: Chạy thủ công

#### Terminal 1: API Backend

```bash
cd /root/Healthcare_Recommendation_System

# Với cache (nhanh - 5s)
SAMPLE_SIZE=0 ENABLE_CACHE=1 ENABLE_LLM_GENERATION=1 \
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Hoặc chạy background
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

**Đợi API ready** (~5-30 giây):
```bash
# Kiểm tra health
curl http://localhost:8000/api/health
```

#### Terminal 2: Django Web

```bash
cd /root/Healthcare_Recommendation_System/web

# Development mode
python manage.py runserver 0.0.0.0:8080

# Background mode
nohup python manage.py runserver 0.0.0.0:8080 > web.log 2>&1 &
```

## Truy cập hệ thống

- **Trang chủ**: http://localhost:8080
- **AI Chatbot**: http://localhost:8080/ai-advisor/
- **Bài viết**: http://localhost:8080/articles/
- **Q&A**: http://localhost:8080/topics/
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/api/health

## Dừng hệ thống

### Cách 1: Sử dụng script

```bash
./STOP.sh
```

### Cách 2: Thủ công

```bash
# Dừng API
pkill -9 -f 'uvicorn api.main'

# Dừng Web
pkill -9 -f 'manage.py runserver'
```

## Kiểm tra logs

```bash
# API logs
tail -f api.log

# Web logs
tail -f web/web.log

# Xem logs realtime
tail -f api.log web/web.log
```

## Kiểm tra processes

```bash
# Xem API process
ps aux | grep uvicorn

# Xem Web process
ps aux | grep manage.py

# Xem tất cả
ps aux | grep -E "uvicorn|manage.py" | grep -v grep
```

## Troubleshooting

### API không khởi động

```bash
# Xem logs
tail -50 api.log

# Thử disable cache
ENABLE_CACHE=0 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Rebuild FAISS index
python scripts/rebuild_fast.py
```

### Web không kết nối API

```bash
# Kiểm tra API health
curl http://localhost:8000/api/health

# Kiểm tra port
netstat -tlnp | grep 8000
```

### Port bị chiếm

```bash
# Kill process trên port 8000
lsof -ti:8000 | xargs kill -9

# Kill process trên port 8080
lsof -ti:8080 | xargs kill -9
```

### Out of memory

```bash
# Dùng CPU thay vì GPU
FORCE_CPU=1 uvicorn api.main:app --host 0.0.0.0 --port 8000

# Hoặc giảm data
SAMPLE_SIZE=5000 uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

Tạo file `.env` từ template:
```bash
cp .env.example .env
nano .env
```

**Recommended settings:**
```env
# Cache để startup nhanh
ENABLE_CACHE=1

# LLM generation
ENABLE_LLM_GENERATION=1

# Use all data
SAMPLE_SIZE=0

# HuggingFace token (nếu cần)
HUGGINGFACE_HUB_TOKEN=your_token_here
```

## First Time Setup

Nếu lần đầu chạy hoặc có lỗi:

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd web && pip install -r requirements.txt && cd ..

# 2. Migrate database
cd web
python manage.py migrate
python manage.py import_data  # Import CSV data
cd ..

# 3. Build FAISS cache (optional - tăng tốc startup)
python scripts/rebuild_fast.py

# 4. Start system
./START.sh
```

## Daily Usage

**Mỗi ngày chỉ cần:**
```bash
./START.sh
```

**Khi kết thúc:**
```bash
./STOP.sh
```
