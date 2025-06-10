from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from typing import List, Optional
from services.llm_processor import LLMProcessor

from database import get_db
from models import Paper, PaperSection
from schemas import PaperCreate, PaperResponse, PaperUpdate, PaperBatchInput, PaperSectionResponse

router = APIRouter()


@router.post("/batch", response_model=dict)
async def create_papers_batch(
    batch_input: PaperBatchInput,
    db: AsyncSession = Depends(get_db)
):
    """Create multiple papers from JSON input - MVP endpoint"""
    created_papers = []
    failed_papers = []
    
    for paper_data in batch_input.papers:
        try:
            # Basic validation
            if "title" not in paper_data or "url" not in paper_data:
                failed_papers.append({
                    "data": paper_data,
                    "error": "Missing title or url"
                })
                continue
            
            # Check if paper already exists
            result = await db.execute(
                select(Paper).where(Paper.url == paper_data["url"])
            )
            if result.scalar_one_or_none():
                failed_papers.append({
                    "data": paper_data,
                    "error": "Paper with this URL already exists"
                })
                continue
            
            # Create paper
            paper = Paper(
                title=paper_data["title"],
                url=paper_data["url"],
                arxiv_id=paper_data.get("arxiv_id"),
                authors=paper_data.get("authors", []) if paper_data.get("authors") else None,
                abstract=paper_data.get("abstract"),
                categories=paper_data.get("categories", []) if paper_data.get("categories") else None
            )
            
            db.add(paper)
            await db.flush()  # Get the ID
            created_papers.append(paper.id)
            
        except Exception as e:
            failed_papers.append({
                "data": paper_data,
                "error": str(e)
            })
    
    await db.commit()
    
    return {
        "created_count": len(created_papers),
        "failed_count": len(failed_papers),
        "created_paper_ids": created_papers,
        "failed_papers": failed_papers
    }


@router.get("/", response_model=List[PaperResponse])
async def get_papers(
    skip: int = 0,
    limit: int = 20,
    status: Optional[str] = None,
    content_created: Optional[bool] = None,
    min_relevance: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get papers with optional filtering"""
    query = select(Paper)
    
    # Apply filters
    if status:
        query = query.where(Paper.processing_status == status)
    if content_created is not None:
        query = query.where(Paper.content_created == content_created)
    if min_relevance:
        query = query.where(Paper.business_relevance_score >= min_relevance)
    
    # Order by relevance score and creation date
    query = query.order_by(
        Paper.business_relevance_score.desc().nulls_last(),
        Paper.created_at.desc()
    ).offset(skip).limit(limit)
    
    result = await db.execute(query)
    papers = result.scalars().all()
    
    return papers


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(paper_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific paper by ID"""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    return paper


@router.get("/{paper_id}/sections", response_model=List[PaperSectionResponse])
async def get_paper_sections(paper_id: int, db: AsyncSession = Depends(get_db)):
    """Get all sections and summaries for a specific paper"""
    # Check if paper exists
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    # Get paper sections
    sections_result = await db.execute(
        select(PaperSection)
        .where(PaperSection.paper_id == paper_id)
        .order_by(PaperSection.order_index.asc().nulls_last(), PaperSection.id.asc())
    )
    sections = sections_result.scalars().all()
    
    return sections


@router.put("/{paper_id}", response_model=PaperResponse)
async def update_paper(
    paper_id: int,
    paper_update: PaperUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update paper metadata"""
    # Check if paper exists
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    # Update fields
    update_data = paper_update.model_dump(exclude_unset=True)
    if update_data:
        await db.execute(
            update(Paper)
            .where(Paper.id == paper_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(paper)
    
    return paper


@router.delete("/{paper_id}")
async def delete_paper(paper_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a paper"""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    await db.execute(delete(Paper).where(Paper.id == paper_id))
    await db.commit()
    
    return {"message": "Paper deleted successfully"}


@router.get("/unused/content", response_model=List[PaperResponse])
async def get_unused_papers(
    min_relevance: int = 7,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Get papers that haven't been used for content creation yet"""
    query = select(Paper).where(
        Paper.content_created == False,
        Paper.processing_status == "completed",
        Paper.business_relevance_score >= min_relevance
    ).order_by(
        Paper.business_relevance_score.desc(),
        Paper.created_at.desc()
    ).limit(limit)
    
    result = await db.execute(query)
    papers = result.scalars().all()
    
    return papers


@router.get("/{paper_id}/comprehensive-summary")
async def get_comprehensive_summary(
    paper_id: int,
    style: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get a comprehensive summary combining all section summaries"""
    # Check if paper exists
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found"
        )
    
    # Check if we already have a cached summary for this style
    if style:
        summary_column = f"comprehensive_summary_{style}"
        if hasattr(paper, summary_column) and getattr(paper, summary_column):
            return {
                "paper_id": paper_id,
                "title": paper.title,
                "comprehensive_summary": getattr(paper, summary_column),
                "style": style,
                "cached": True
            }
    elif paper.comprehensive_summary:
        return {
            "paper_id": paper_id,
            "title": paper.title,
            "comprehensive_summary": paper.comprehensive_summary,
            "style": "default",
            "cached": True
        }
    
    # Get paper sections
    sections_result = await db.execute(
        select(PaperSection)
        .where(PaperSection.paper_id == paper_id)
        .order_by(PaperSection.order_index.asc().nulls_last(), PaperSection.id.asc())
    )
    sections = sections_result.scalars().all()
    
    if not sections:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No sections found for this paper"
        )
    
    # Collect summaries based on style
    section_summaries = []
    for section in sections:
        if style == 'academic' and section.summary_academic:
            section_summaries.append({
                'type': section.section_type,
                'summary': section.summary_academic
            })
        elif style == 'business' and section.summary_business:
            section_summaries.append({
                'type': section.section_type,
                'summary': section.summary_business
            })
        elif style == 'social' and section.summary_social:
            section_summaries.append({
                'type': section.section_type,
                'summary': section.summary_social
            })
        elif style == 'educational' and section.summary_educational:
            section_summaries.append({
                'type': section.section_type,
                'summary': section.summary_educational
            })
        elif section.summary:  # Default to original summary if style not specified or not available
            section_summaries.append({
                'type': section.section_type,
                'summary': section.summary
            })
    
    if not section_summaries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No summaries found for style: {style if style else 'default'}"
        )
    
    # Use LLM to generate comprehensive summary
    llm_processor = LLMProcessor()
    comprehensive_summary = await llm_processor.generate_comprehensive_summary(
        paper.title,
        paper.abstract,
        section_summaries
    )
    
    # Save the summary in the database
    update_values = {
        "comprehensive_summary_updated_at": func.now()
    }
    
    if style:
        update_values[f"comprehensive_summary_{style}"] = comprehensive_summary
    else:
        update_values["comprehensive_summary"] = comprehensive_summary
    
    await db.execute(
        update(Paper)
        .where(Paper.id == paper_id)
        .values(**update_values)
    )
    await db.commit()
    
    return {
        "paper_id": paper_id,
        "title": paper.title,
        "comprehensive_summary": comprehensive_summary,
        "style": style or "default",
        "sections_included": [s['type'] for s in section_summaries],
        "cached": False
    }
