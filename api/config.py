"""Configuration settings for Healthcare RAG API"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Healthcare RAG API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Hệ thống tư vấn sức khỏe thông minh sử dụng RAG"
    
    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8080"]
    
    # Model Settings
    RETRIEVAL_MODEL: str = "vinai/phobert-base"
    GENERATION_MODEL: str = "Viet-Mistral/Vistral-7B-Chat"
    
    # Device Settings
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "0") == "1"
    USE_GPU: bool = not FORCE_CPU
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    QA_CSV_PATH: Path = DATA_DIR / "QAs.csv"
    ARTICLES_CSV_PATH: Path = DATA_DIR / "articles.csv"
    
    # Cache Paths (fix issue #6: save/load indices to avoid rebuild)
    CACHE_DIR: Path = BASE_DIR / "cache"
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "1") == "1"
    QA_INDEX_CACHE: Path = CACHE_DIR / "qa_index.bin"
    ARTICLE_INDEX_CACHE: Path = CACHE_DIR / "article_index.bin"
    QA_EMBEDDINGS_CACHE: Path = CACHE_DIR / "qa_embeddings.npy"
    ARTICLE_EMBEDDINGS_CACHE: Path = CACHE_DIR / "article_embeddings.npy"
    METADATA_CACHE: Path = CACHE_DIR / "metadata.pkl"
    
    # Index Settings
    SAMPLE_SIZE: int = int(os.getenv("SAMPLE_SIZE", "5000"))  # Tăng từ 1000 lên 5000
    INDEX_SPACE: str = "cosine"
    
    # Retrieval Settings
    TOP_K_QA: int = 5
    TOP_K_ARTICLES: int = 1
    QUESTION_SIM_THRESHOLD: float = 0.55
    COMBINED_SCORE_THRESHOLD: float = 0.25
    
    # Generation Settings
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # Hugging Face Token (optional)
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
