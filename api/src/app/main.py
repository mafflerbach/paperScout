from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db
from routers import papers, search, processing
from config import settings, CORS_ORIGINS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown - cleanup if needed

app = FastAPI(
    title="PaperScout API",
    description="AI-powered research paper discovery and analysis",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Use the separate variable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])

@app.get("/")
async def root():
    return {"message": "Welcome to PaperScout API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "paperscout-api"}
