# 🚀 Hướng Dẫn Nhanh - Kết Nối Remote GPU

## Bước 1: Upload gpu_service.py lên container

### Cách 1: Dùng SSH (nhanh nhất)
```bash
# Copy SSH command từ FPT AI Factory
ssh root@tcp-endpoint.serverless.fptcloud.jp:36038 -i ~/.ssh/private_key

# Upload file
scp -i ~/.ssh/private_key gpu_service.py root@tcp-endpoint.serverless.fptcloud.jp:36038:/workspace/
```

### Cách 2: Dùng Jupyter Notebook
1. Truy cập: https://my-container-etnt7h6b-8888.serverless.fptcloud.jp
2. Click **Upload** → chọn `gpu_service.py`
3. Upload vào `/workspace/`

---

## Bước 2: Chạy GPU Service trên container

```bash
# SSH vào container
ssh root@tcp-endpoint.serverless.fptcloud.jp:36038 -i ~/.ssh/private_key

# Cài dependencies
pip install fastapi uvicorn transformers torch

# Chạy service
cd /workspace
python gpu_service.py
```

Service sẽ chạy trên **port 8888** (đã expose HTTP).

---

## Bước 3: Cấu hình Local

Tạo file `.env`:
```env
REMOTE_GPU_ENABLED=1
REMOTE_GPU_URL=https://my-container-etnt7h6b-8888.serverless.fptcloud.jp
```

---

## Bước 4: Test

```bash
# Test từ local
curl https://my-container-etnt7h6b-8888.serverless.fptcloud.jp/health
```

Kết quả:
```json
{
  "status": "healthy",
  "device": "cuda",
  "models_loaded": {"phobert": true, "vistral": true},
  "vram_gb": 15.2
}
```

---

## Chạy Local với Remote GPU

```powershell
# Terminal 1: API
.venv\Scripts\activate
python -m uvicorn api.main:app --port 8000

# Terminal 2: Web
cd web
.venv\Scripts\activate
python manage.py runserver 8080
```

Mở browser: http://localhost:8080

✅ Xong! System chạy local nhưng inference qua GPU H200.
