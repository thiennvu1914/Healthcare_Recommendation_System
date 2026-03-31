"""Pydantic models for API request/response"""
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(..., description="Câu hỏi của người dùng", min_length=5, max_length=500)
    include_sources: bool = Field(default=True, description="Có trả về nguồn tham khảo không")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Bé 2 tuổi sốt 38.5 độ, tôi phải làm gì?",
                "include_sources": True
            }
        }

class SourceInfo(BaseModel):
    """Information about a source"""
    type: str = Field(..., description="Loại nguồn: 'qa' hoặc 'article'")
    id: Optional[str] = Field(None, description="ID của nguồn (index hoặc link)")
    title: Optional[str] = Field(None, description="Tiêu đề bài viết (nếu là article)")
    question: Optional[str] = Field(None, description="Câu hỏi gốc (nếu là qa)")
    full_answer: Optional[str] = Field(None, description="Câu trả lời đầy đủ (nếu là qa)")
    link: Optional[str] = Field(None, description="Link bài viết gốc (nếu là article)")
    score: float = Field(..., description="Điểm tương đồng")
    snippet: str = Field(..., description="Đoạn trích dẫn")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="Câu trả lời từ hệ thống")
    specialty: str = Field(..., description="Chuyên khoa liên quan")
    confidence: float = Field(..., description="Độ tin cậy của câu trả lời (0-1)")
    sources: Optional[List[SourceInfo]] = Field(None, description="Danh sách nguồn tham khảo")
    disclaimer: Optional[str] = Field(
        None,
        description="Cảnh báo/yêu cầu thăm khám (tách riêng để UI hiển thị rõ)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Với bé 2 tuổi sốt 38.5°C, bạn nên theo dõi nhiệt độ và cho bé nghỉ ngơi...",
                "specialty": "Nhi Khoa",
                "confidence": 0.85,
                "disclaimer": "Thông tin chỉ mang tính tham khảo và không thay thế chẩn đoán của bác sĩ...",
                "sources": [
                    {
                        "type": "qa",
                        "question": "Bé tôi sốt cao phải làm sao?",
                        "score": 0.89,
                        "snippet": "Sốt 38.5°C ở trẻ nhỏ cần theo dõi nhiệt độ..."
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Trạng thái hệ thống")
    version: str = Field(..., description="Phiên bản API")
    models_loaded: bool = Field(..., description="Các model đã load chưa")
    gpu_available: bool = Field(..., description="GPU có khả dụng không")

class SpecialtyInfo(BaseModel):
    """Information about a medical specialty"""
    name: str = Field(..., description="Tên chuyên khoa")
    count: int = Field(..., description="Số lượng câu hỏi liên quan")

class SpecialtiesResponse(BaseModel):
    """List of available specialties"""
    specialties: List[SpecialtyInfo] = Field(..., description="Danh sách chuyên khoa")
    total: int = Field(..., description="Tổng số chuyên khoa")

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Mô tả lỗi")
    detail: Optional[str] = Field(None, description="Chi tiết lỗi")
