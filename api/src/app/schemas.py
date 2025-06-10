from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ImplementationFeasibility(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Paper schemas
class PaperBase(BaseModel):
    title: str
    url: str
    arxiv_id: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    published_date: Optional[date] = None
    categories: Optional[List[str]] = None


class PaperCreate(PaperBase):
    pass


class PaperUpdate(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    main_topics: Optional[List[str]] = None
    techniques: Optional[List[str]] = None
    applications: Optional[List[str]] = None
    difficulty_level: Optional[DifficultyLevel] = None
    implementation_feasibility: Optional[ImplementationFeasibility] = None
    business_relevance_score: Optional[int] = None
    content_created: Optional[bool] = None
    content_note_path: Optional[str] = None


class PaperResponse(PaperBase):
    id: int
    pdf_path: Optional[str]
    download_status: DownloadStatus
    processing_status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    
    # LLM generated metadata
    main_topics: Optional[List[str]]
    techniques: Optional[List[str]]
    applications: Optional[List[str]]
    difficulty_level: Optional[DifficultyLevel]
    implementation_feasibility: Optional[ImplementationFeasibility]
    business_relevance_score: Optional[int]
    
    # Content creation
    content_created: bool
    content_note_path: Optional[str]
    
    class Config:
        from_attributes = True


# Section schemas
class PaperSectionBase(BaseModel):
    section_type: Optional[str]
    section_title: Optional[str]
    content: str
    order_index: Optional[int]


class PaperSectionCreate(PaperSectionBase):
    paper_id: int


class PaperSectionResponse(PaperSectionBase):
    id: int
    paper_id: int
    summary: Optional[str]
    
    # Multi-style refined summaries
    summary_academic: Optional[str] = None    # Technical precision for researchers
    summary_business: Optional[str] = None    # ROI and practical applications focus
    summary_social: Optional[str] = None      # Optimized for LinkedIn/Twitter posts
    summary_educational: Optional[str] = None # Accessible for students/learners
    
    # Refinement tracking
    refinement_status: Optional[str] = "pending"
    refined_at: Optional[datetime] = None
    
    created_at: datetime
    
    class Config:
        from_attributes = True


# Batch input for MVP
class PaperBatchInput(BaseModel):
    papers: List[Dict[str, str]]  # [{"title": "...", "url": "..."}]


# Search schemas
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    papers: List[PaperResponse]
    total_count: int
    query: str


# Tag schemas
class TagBase(BaseModel):
    name: str
    category: Optional[str]


class TagCreate(TagBase):
    pass


class TagResponse(TagBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Processing schemas
class ProcessingJobStatus(BaseModel):
    paper_id: int
    status: ProcessingStatus
    message: Optional[str]
    progress: Optional[float]  # 0.0 to 1.0


class ProcessingStats(BaseModel):
    total_papers: int
    pending: int
    processing: int
    completed: int
    failed: int
