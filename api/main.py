"""FastAPI main application for Healthcare RAG API"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from typing import List

from api.config import settings
from api.models import (
    ChatRequest, ChatResponse, HealthResponse, 
    SpecialtiesResponse, SpecialtyInfo, SourceInfo, ErrorResponse
)
from api.rag_engine import get_rag_engine, HealthcareRAGEngine

# Global RAG engine
rag_engine: HealthcareRAGEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global rag_engine
    print("[API] Initializing Healthcare RAG Engine...")
    try:
        rag_engine = get_rag_engine()
        print("[API] RAG Engine initialized successfully")
    except Exception as e:
        print(f"[API] Failed to initialize RAG Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    print("[API] Shutting down...")

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Healthcare RAG API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = rag_engine is not None and rag_engine.model_phobert is not None
    gpu_available = torch.cuda.is_available()
    
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        version=settings.API_VERSION,
        models_loaded=models_loaded,
        gpu_available=gpu_available
    )

@app.post("/api/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat(request: ChatRequest):
    """
    Main chat endpoint - receive user question and return AI-generated answer
    
    - **query**: Câu hỏi của người dùng (5-500 ký tự)
    - **include_sources**: Có trả về nguồn tham khảo không (mặc định: True)
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine chưa sẵn sàng")
    
    try:
        # Retrieve relevant Q&A pairs
        qa_results = rag_engine.retrieve_qa(request.query, k=settings.TOP_K_QA)
        
        # Retrieve relevant articles
        article_results = rag_engine.retrieve_articles(request.query, k=settings.TOP_K_ARTICLES)
        
        # Generate answer
        answer, specialty, confidence = rag_engine.generate_answer(
            request.query, 
            qa_results, 
            article_results
        )
        
        # Prepare sources if requested
        sources = None
        if request.include_sources:
            sources = []
            
            # Add Q&A sources
            for qa in qa_results[:3]:
                sources.append(SourceInfo(
                    type="qa",
                    id=f"qa_{qa.get('index', 0)}",
                    question=qa.get("question", ""),
                    full_answer=qa.get("answer", ""),
                    score=qa.get("score", 0.0),
                    snippet=qa.get("answer", "")[:200] + "..."
                ))
            
            # Add article sources
            for article in article_results[:2]:
                article_link = article.get("link", "")
                sources.append(SourceInfo(
                    type="article",
                    id=article_link if article_link else f"article_{article.get('index', 0)}",
                    title=article.get("title", ""),
                    link=article_link,
                    score=article.get("score", 0.0),
                    snippet=article.get("snippet", "")[:200] + "..."
                ))
        
        return ChatResponse(
            answer=answer,
            specialty=specialty,
            confidence=confidence,
            sources=sources,
            disclaimer=rag_engine.build_disclaimer(specialty=specialty, confidence=confidence)
        )
    
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/api/specialties", response_model=SpecialtiesResponse)
async def get_specialties():
    """
    Get list of available medical specialties
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine chưa sẵn sàng")
    
    try:
        specialties_data = rag_engine.get_specialties()
        
        specialties = [
            SpecialtyInfo(name=s["name"], count=s["count"])
            for s in specialties_data
        ]
        
        return SpecialtiesResponse(
            specialties=specialties,
            total=len(specialties)
        )
    
    except Exception as e:
        print(f"[API Error] {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine chưa sẵn sàng")
    
    return {
        "total_qa_pairs": len(rag_engine.df_qa) if rag_engine.df_qa is not None else 0,
        "total_articles": len(rag_engine.df_articles) if rag_engine.df_articles is not None else 0,
        "qa_index_size": rag_engine.qa_index.get_current_count() if rag_engine.qa_index else 0,
        "article_index_size": rag_engine.article_index.get_current_count() if rag_engine.article_index else 0,
        "device": rag_engine.device if rag_engine else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
