# Scripts

Thư mục chứa các script tiện ích cho hệ thống.

## Rebuild Scripts

### `rebuild_fast.py`
Script rebuild FAISS index với toàn bộ data trên GPU.

**Sử dụng:**
```bash
cd /root/Healthcare_Recommendation_System
python scripts/rebuild_fast.py
```

**Tính năng:**
- Sử dụng GPU (CUDA) để tăng tốc
- Batch processing với fp16
- Rebuild cả QA index và Article index
- Lưu cache vào thư mục `cache/`

### `rebuild_index.py`
Script rebuild index đơn giản (legacy).

## Test Scripts

### `test_api.py`
Script test API endpoint.

**Sử dụng:**
```bash
cd /root/Healthcare_Recommendation_System
python scripts/test_api.py
```
