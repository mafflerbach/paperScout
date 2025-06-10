from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func, and_, or_
from typing import List, Optional, Dict, Any
import logging

from database import get_db
from models import Paper, PaperSection, Tag, PaperTag
from schemas import SearchRequest, SearchResult, PaperResponse
from services.embedding import get_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/semantic", response_model=SearchResult)
async def semantic_search(
    search_request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Semantic search using vector embeddings"""
    try:
        # Get embedding for the query
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.get_embedding(search_request.query)
        
        # Build the vector search query
        # Search in paper sections for semantic similarity
        vector_query = text("""
            SELECT DISTINCT p.*, 
                   ps.embedding <-> :query_vector as distance,
                   ps.section_type,
                   ps.summary
            FROM papers p
            JOIN paper_sections ps ON p.id = ps.paper_id
            WHERE ps.embedding IS NOT NULL
            ORDER BY ps.embedding <-> :query_vector
            LIMIT :limit
        """)
        
        result = await db.execute(
            vector_query,
            {
                "query_vector": str(query_embedding),
                "limit": search_request.limit
            }
        )
        
        rows = result.fetchall()
        
        # Convert to paper objects (deduplicate by paper_id)
        seen_paper_ids = set()
        papers = []
        
        for row in rows:
            if row.id not in seen_paper_ids:
                # Create paper object from row
                paper = Paper(
                    id=row.id,
                    title=row.title,
                    url=row.url,
                    pdf_path=row.pdf_path,
                    arxiv_id=row.arxiv_id,
                    authors=row.authors,
                    abstract=row.abstract,
                    published_date=row.published_date,
                    categories=row.categories,
                    download_status=row.download_status,
                    processing_status=row.processing_status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    processed_at=row.processed_at,
                    full_text=row.full_text,
                    main_topics=row.main_topics,
                    techniques=row.techniques,
                    applications=row.applications,
                    difficulty_level=row.difficulty_level,
                    implementation_feasibility=row.implementation_feasibility,
                    business_relevance_score=row.business_relevance_score,
                    content_created=row.content_created,
                    content_note_path=row.content_note_path
                )
                papers.append(paper)
                seen_paper_ids.add(row.id)
        
        return SearchResult(
            papers=papers,
            total_count=len(papers),
            query=search_request.query
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/", response_model=SearchResult)
async def search_papers(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    topics: Optional[List[str]] = Query(None, description="Filter by topics"),
    techniques: Optional[List[str]] = Query(None, description="Filter by techniques"),
    applications: Optional[List[str]] = Query(None, description="Filter by applications"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty level"),
    min_relevance: Optional[int] = Query(None, ge=1, le=10, description="Minimum business relevance score"),
    content_created: Optional[bool] = Query(None, description="Filter by content creation status"),
    db: AsyncSession = Depends(get_db)
):
    """Full-text and metadata search with filters"""
    
    # Build base query
    query = select(Paper).where(Paper.processing_status == "completed")
    
    # Text search in title, abstract, and topics
    if q:
        search_conditions = []
        search_term = f"%{q.lower()}%"
        
        # Search in title and abstract
        search_conditions.extend([
            Paper.title.ilike(search_term),
            Paper.abstract.ilike(search_term)
        ])
        
        # Search in array fields (topics, techniques, applications)
        for field in [Paper.main_topics, Paper.techniques, Paper.applications]:
            search_conditions.append(
                func.array_to_string(field, ' ').ilike(search_term)
            )
        
        query = query.where(or_(*search_conditions))
    
    # Apply filters
    if topics:
        query = query.where(Paper.main_topics.overlap(topics))
    
    if techniques:
        query = query.where(Paper.techniques.overlap(techniques))
    
    if applications:
        query = query.where(Paper.applications.overlap(applications))
    
    if difficulty:
        query = query.where(Paper.difficulty_level == difficulty)
    
    if min_relevance:
        query = query.where(Paper.business_relevance_score >= min_relevance)
    
    if content_created is not None:
        query = query.where(Paper.content_created == content_created)
    
    # Order by relevance and date
    query = query.order_by(
        Paper.business_relevance_score.desc().nulls_last(),
        Paper.created_at.desc()
    ).limit(limit)
    
    result = await db.execute(query)
    papers = result.scalars().all()
    
    return SearchResult(
        papers=papers,
        total_count=len(papers),
        query=q
    )


@router.get("/tags", response_model=List[str])
async def get_popular_tags(
    category: Optional[str] = Query(None, description="Filter by tag category"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get popular tags for autocomplete/filtering"""
    
    # Query to get tags with paper counts
    query = text("""
        SELECT t.name, COUNT(pt.paper_id) as paper_count
        FROM tags t
        LEFT JOIN paper_tags pt ON t.id = pt.tag_id
        WHERE (:category IS NULL OR t.category = :category)
        GROUP BY t.id, t.name
        ORDER BY paper_count DESC, t.name
        LIMIT :limit
    """)
    
    result = await db.execute(
        query,
        {"category": category, "limit": limit}
    )
    
    tags = [row.name for row in result.fetchall()]
    return tags


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=2, description="Partial search term"),
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db)
):
    """Get search suggestions based on paper titles and topics"""
    
    search_term = f"%{q.lower()}%"
    
    # Get title suggestions
    title_query = select(Paper.title).where(
        and_(
            Paper.title.ilike(search_term),
            Paper.processing_status == "completed"
        )
    ).limit(limit)
    
    title_result = await db.execute(title_query)
    title_suggestions = [row.title for row in title_result.fetchall()]
    
    # Get topic suggestions from tags
    tag_query = select(Tag.name).where(
        Tag.name.ilike(search_term)
    ).limit(limit)
    
    tag_result = await db.execute(tag_query)
    tag_suggestions = [row.name for row in tag_result.fetchall()]
    
    return {
        "titles": title_suggestions[:3],  # Limit to 3 title suggestions
        "topics": tag_suggestions[:5]     # Up to 5 topic suggestions
    }


@router.get("/stats")
async def get_search_stats(db: AsyncSession = Depends(get_db)):
    """Get search/discovery statistics"""
    
    stats_query = text("""
        SELECT 
            COUNT(*) as total_papers,
            COUNT(*) FILTER (WHERE processing_status = 'completed') as processed_papers,
            COUNT(*) FILTER (WHERE content_created = true) as content_created_papers,
            AVG(business_relevance_score) FILTER (WHERE business_relevance_score IS NOT NULL) as avg_relevance,
            COUNT(DISTINCT unnest(main_topics)) as unique_topics,
            COUNT(DISTINCT unnest(techniques)) as unique_techniques,
            COUNT(DISTINCT unnest(applications)) as unique_applications
        FROM papers
    """)
    
    result = await db.execute(stats_query)
    row = result.fetchone()
    
    return {
        "total_papers": row.total_papers,
        "processed_papers": row.processed_papers,
        "content_created_papers": row.content_created_papers,
        "avg_relevance_score": round(float(row.avg_relevance or 0), 2),
        "unique_topics": row.unique_topics,
        "unique_techniques": row.unique_techniques,
        "unique_applications": row.unique_applications,
        "content_pipeline_remaining": row.processed_papers - row.content_created_papers
    }


@router.get("/similar/{paper_id}", response_model=List[PaperResponse])
async def find_similar_papers(
    paper_id: int,
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db)
):
    """Find papers similar to a given paper using embeddings"""
    
    # First, get the target paper's sections
    sections_query = select(PaperSection.embedding).where(
        and_(
            PaperSection.paper_id == paper_id,
            PaperSection.embedding.is_not(None)
        )
    ).limit(1)  # Use the first section's embedding as representative
    
    section_result = await db.execute(sections_query)
    section = section_result.scalar_one_or_none()
    
    if not section or not section.embedding:
        raise HTTPException(
            status_code=404,
            detail="Paper not found or no embeddings available"
        )
    
    # Find similar papers using vector similarity
    similar_query = text("""
        SELECT DISTINCT p.*,
               MIN(ps.embedding <-> :target_embedding) as min_distance
        FROM papers p
        JOIN paper_sections ps ON p.id = ps.paper_id
        WHERE p.id != :paper_id 
          AND ps.embedding IS NOT NULL
          AND p.processing_status = 'completed'
        GROUP BY p.id
        ORDER BY min_distance
        LIMIT :limit
    """)
    
    result = await db.execute(
        similar_query,
        {
            "target_embedding": str(section.embedding),
            "paper_id": paper_id,
            "limit": limit
        }
    )
    
    rows = result.fetchall()
    
    # Convert to Paper objects
    papers = []
    for row in rows:
        paper = Paper(
            id=row.id,
            title=row.title,
            url=row.url,
            pdf_path=row.pdf_path,
            arxiv_id=row.arxiv_id,
            authors=row.authors,
            abstract=row.abstract,
            published_date=row.published_date,
            categories=row.categories,
            download_status=row.download_status,
            processing_status=row.processing_status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            processed_at=row.processed_at,
            full_text=row.full_text,
            main_topics=row.main_topics,
            techniques=row.techniques,
            applications=row.applications,
            difficulty_level=row.difficulty_level,
            implementation_feasibility=row.implementation_feasibility,
            business_relevance_score=row.business_relevance_score,
            content_created=row.content_created,
            content_note_path=row.content_note_path
        )
        papers.append(paper)
    
    return papers
