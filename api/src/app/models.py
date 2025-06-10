from sqlalchemy import Column, Integer, String, Text, Date, Boolean, ARRAY, Float, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from database import Base


class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False, unique=True)
    pdf_path = Column(Text)
    arxiv_id = Column(Text)
    authors = Column(ARRAY(Text))
    abstract = Column(Text)
    published_date = Column(Date)
    categories = Column(ARRAY(Text))
    
    # Processing status
    download_status = Column(String(20), default="pending")
    processing_status = Column(String(20), default="pending")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Content
    full_text = Column(Text)
    
    # LLM generated metadata
    main_topics = Column(ARRAY(Text))
    techniques = Column(ARRAY(Text))
    applications = Column(ARRAY(Text))
    difficulty_level = Column(String(20))
    implementation_feasibility = Column(String(20))
    business_relevance_score = Column(Integer)
    
    # Content creation tracking
    content_created = Column(Boolean, default=False)
    content_note_path = Column(Text)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('business_relevance_score >= 1 AND business_relevance_score <= 10'),
    )
    
    # Relationships
    sections = relationship("PaperSection", back_populates="paper", cascade="all, delete-orphan")
    paper_tags = relationship("PaperTag", back_populates="paper", cascade="all, delete-orphan")


class PaperSection(Base):
    __tablename__ = "paper_sections"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    section_type = Column(String(50))
    section_title = Column(Text)
    content = Column(Text, nullable=False)
    summary = Column(Text)
    order_index = Column(Integer)
    
    # Vector embedding
    embedding = Column(Vector(384))  # Adjust dimension as needed
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    paper = relationship("Paper", back_populates="sections")


class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    paper_tags = relationship("PaperTag", back_populates="tag", cascade="all, delete-orphan")


class PaperTag(Base):
    __tablename__ = "paper_tags"
    
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True)
    confidence_score = Column(Float)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1'),
    )
    
    # Relationships
    paper = relationship("Paper", back_populates="paper_tags")
    tag = relationship("Tag", back_populates="paper_tags")


class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    query_vector = Column(Vector(384))
    result_count = Column(Integer)
    user_feedback = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
