from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://scout:secure_password_change_me@localhost:5432/paperscout"
    
    # File storage
    pdf_storage_path: str = "/app/pdfs"
    
    # LLM settings
    llm_endpoint: str = "http://localhost:1234/v1/chat/completions"
    llm_model: str = "local-model"
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Logging
    log_level: str = "INFO"
    
    # Worker mode
    worker_mode: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Simple CORS origins list
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
