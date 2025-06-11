from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://scout:secure_password_change_me@localhost:5432/paperscout"
    
    # File storage
    pdf_storage_path: str = "./pdfs"
    content_storage_path: str = "./content"
    
    # LLM settings
    llm_endpoint: str = "http://localhost:1234/v1/chat/completions"
    llm_model: str = "llama2-7b-chat"
    
    # Task-specific LLM endpoints
    metadata_llm_endpoint: Optional[str] = None  # Endpoint for fast metadata extraction
    summary_llm_endpoint: Optional[str] = None   # Endpoint for section summaries
    comprehensive_llm_endpoint: Optional[str] = None  # Endpoint for comprehensive analysis
    
    # Task-specific LLM models
    metadata_llm_model: Optional[str] = None  # Small model for metadata (e.g. tinyllama-1.1b-chat)
    summary_llm_model: Optional[str] = None   # Medium model for summaries (e.g. llama2-7b-chat)
    comprehensive_llm_model: Optional[str] = None  # Large model for analysis (e.g. llama2-13b-chat)
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Logging
    log_level: str = "INFO"
    
    # Worker mode
    worker_mode: bool = False
    
    # Processing settings
    max_concurrent_downloads: int = 5
    max_concurrent_processing: int = 2
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

# Simple CORS origins list
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
