from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, delete
from typing import List, Optional
import logging

from database import get_db
from models import Paper, PaperSection
from schemas import ProcessingJobStatus, ProcessingStats
from services.pdf_processor import get_pdf_processor
from services.llm_processor import get_llm_processor
from services.embedding import get_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/papers/{paper_id}/download")
async def download_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Download PDF for a specific paper"""
    # Check if paper exists
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if paper.download_status == "downloaded":
        return {"message": "Paper already downloaded", "paper_id": paper_id}
    
    # Add to background task
    background_tasks.add_task(download_paper_task, paper_id)
    
    # Update status to processing
    await db.execute(
        update(Paper)
        .where(Paper.id == paper_id)
        .values(download_status="processing")
    )
    await db.commit()
    
    return {"message": "Download started", "paper_id": paper_id}


@router.post("/papers/{paper_id}/process")
async def process_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Process paper (extract text, generate summaries, embeddings)"""
    # Check if paper exists and is downloaded
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if paper.download_status != "downloaded":
        raise HTTPException(status_code=400, detail="Paper must be downloaded first")
    
    if paper.processing_status == "completed":
        return {"message": "Paper already processed", "paper_id": paper_id}
    
    # Add to background task
    background_tasks.add_task(process_paper_task, paper_id)
    
    # Update status
    await db.execute(
        update(Paper)
        .where(Paper.id == paper_id)
        .values(processing_status="processing")
    )
    await db.commit()
    
    return {"message": "Processing started", "paper_id": paper_id}


@router.post("/papers/{paper_id}/full")
async def download_and_process_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Download and process paper in one go"""
    # Check if paper exists
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Add to background task
    background_tasks.add_task(full_pipeline_task, paper_id)
    
    # Update statuses
    await db.execute(
        update(Paper)
        .where(Paper.id == paper_id)
        .values(
            download_status="processing",
            processing_status="processing"
        )
    )
    await db.commit()
    
    return {"message": "Full pipeline started", "paper_id": paper_id}


@router.post("/batch/process")
async def process_batch(
    background_tasks: BackgroundTasks,
    limit: int = 10,
    download_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Process multiple papers in batch"""
    if download_only:
        # Get papers that need downloading
        query = select(Paper).where(
            Paper.download_status == "pending"
        ).limit(limit)
    else:
        # Get papers that need full processing
        query = select(Paper).where(
            Paper.processing_status == "pending"
        ).limit(limit)
    
    result = await db.execute(query)
    papers = result.scalars().all()
    
    if not papers:
        return {"message": "No papers to process", "count": 0}
    
    # Add batch processing task
    paper_ids = [p.id for p in papers]
    background_tasks.add_task(batch_processing_task, paper_ids, download_only)
    
    return {
        "message": f"Batch processing started for {len(papers)} papers",
        "paper_ids": paper_ids,
        "download_only": download_only
    }


@router.get("/status/{paper_id}", response_model=ProcessingJobStatus)
async def get_processing_status(paper_id: int, db: AsyncSession = Depends(get_db)):
    """Get processing status for a specific paper"""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # Determine overall status
    if paper.processing_status == "completed":
        status = "completed"
        message = "Paper processing completed"
        progress = 1.0
    elif paper.processing_status == "processing":
        status = "processing"
        message = "Paper is being processed"
        progress = 0.5
    elif paper.download_status == "failed" or paper.processing_status == "failed":
        status = "failed"
        message = "Processing failed"
        progress = 0.0
    else:
        status = "pending"
        message = "Waiting to be processed"
        progress = 0.0
    
    return ProcessingJobStatus(
        paper_id=paper_id,
        status=status,
        message=message,
        progress=progress
    )


@router.get("/stats", response_model=ProcessingStats)
async def get_processing_stats(db: AsyncSession = Depends(get_db)):
    """Get overall processing statistics"""
    result = await db.execute(
        select(
            Paper.processing_status,
            func.count().label("count")
        ).group_by(Paper.processing_status)
    )
    
    status_counts = {row.processing_status: row.count for row in result.fetchall()}
    
    total = sum(status_counts.values())
    
    return ProcessingStats(
        total_papers=total,
        pending=status_counts.get("pending", 0),
        processing=status_counts.get("processing", 0),
        completed=status_counts.get("completed", 0),
        failed=status_counts.get("failed", 0)
    )


@router.post("/reprocess/{paper_id}")
async def reprocess_paper(
    paper_id: int,
    background_tasks: BackgroundTasks,
    force: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """Reprocess a paper (useful for failed or updated processing logic)"""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if paper.processing_status == "processing" and not force:
        raise HTTPException(
            status_code=400, 
            detail="Paper is currently being processed. Use force=true to override."
        )
    
    # Clear existing sections
    await db.execute(
        delete(PaperSection).where(PaperSection.paper_id == paper_id)
    )
    
    # Reset processing status
    await db.execute(
        update(Paper)
        .where(Paper.id == paper_id)
        .values(
            processing_status="processing",
            processed_at=None
        )
    )
    await db.commit()
    
    # Add to background task
    background_tasks.add_task(process_paper_task, paper_id)
    
    return {"message": "Reprocessing started", "paper_id": paper_id}





# Background task functions
async def download_paper_task(paper_id: int):
    """Background task to download a paper"""
    try:
        from database import async_session_maker
        
        async with async_session_maker() as db:
            pdf_processor = get_pdf_processor()
            
            # Get paper details
            result = await db.execute(select(Paper).where(Paper.id == paper_id))
            paper = result.scalar_one_or_none()
            
            if not paper:
                logger.error(f"Paper {paper_id} not found for download")
                return
            
            # Download PDF
            pdf_path = await pdf_processor.download_pdf(paper.url, paper_id)
            
            # Update paper with PDF path and status
            await db.execute(
                update(Paper)
                .where(Paper.id == paper_id)
                .values(
                    pdf_path=pdf_path,
                    download_status="downloaded"
                )
            )
            await db.commit()
            
            logger.info(f"Successfully downloaded paper {paper_id}")
            
    except Exception as e:
        logger.error(f"download_paper_task - Failed to download paper {paper_id}: {e}")
        
        # Update status to failed
        from database import async_session_maker
        async with async_session_maker() as db:
            await db.execute(
                update(Paper)
                .where(Paper.id == paper_id)
                .values(download_status="failed")
            )
            await db.commit()


async def process_paper_task(paper_id: int):
    """Background task to process a paper"""
    try:
        from database import async_session_maker
        
        async with async_session_maker() as db:
            pdf_processor = get_pdf_processor()
            llm_processor = get_llm_processor()
            embedding_service = get_embedding_service()
            
            # Get paper
            result = await db.execute(select(Paper).where(Paper.id == paper_id))
            paper = result.scalar_one_or_none()
            
            if not paper:
                logger.error(f"Paper {paper_id} not found for processing")
                return
            
            # Extract text from PDF
            text_content = await pdf_processor.extract_text(paper.pdf_path)
            
            # Split into sections
            sections = await pdf_processor.split_into_sections(text_content)
            
            # Generate summaries for each section
            section_summaries = []
            for i, section in enumerate(sections):
                summary = await llm_processor.summarize_section(
                    section["content"], 
                    section["type"]
                )
                section_summaries.append({
                    **section,
                    "summary": summary,
                    "order_index": i
                })
            
            # Generate paper metadata using LLM
            metadata = await llm_processor.extract_metadata(
                paper.title, 
                paper.abstract or "", 
                text_content[:5000]  # First 5k chars for context
            )
            
            # Generate embeddings for sections
            section_contents = [s["summary"] or s["content"][:1000] for s in section_summaries]
            embeddings = await embedding_service.get_embeddings_batch(section_contents)
            
            # Update paper with metadata
            await db.execute(
                update(Paper)
                .where(Paper.id == paper_id)
                .values(
                    full_text=text_content,
                    main_topics=metadata.get("main_topics", []),
                    techniques=metadata.get("techniques", []),
                    applications=metadata.get("applications", []),
                    difficulty_level=metadata.get("difficulty_level"),
                    implementation_feasibility=metadata.get("implementation_feasibility"),
                    business_relevance_score=metadata.get("business_relevance_score"),
                    processing_status="completed",
                    processed_at=func.now()
                )
            )
            
            # Save sections with embeddings
            for section_data, embedding in zip(section_summaries, embeddings):
                section = PaperSection(
                    paper_id=paper_id,
                    section_type=section_data["type"],
                    section_title=section_data.get("title"),
                    content=section_data["content"],
                    summary=section_data["summary"],
                    order_index=section_data["order_index"],
                    embedding=embedding
                )
                db.add(section)
            
            await db.commit()
            logger.info(f"Successfully processed paper {paper_id}")
            
    except Exception as e:
        logger.error(f"process_paper_task - Failed to process paper {paper_id}: {e}")
        
        from database import async_session_maker
        async with async_session_maker() as db:
            await db.execute(
                update(Paper)
                .where(Paper.id == paper_id)
                .values(processing_status="failed")
            )
            await db.commit()


async def full_pipeline_task(paper_id: int):
    """Background task for full download + process pipeline"""
    await download_paper_task(paper_id)
    
    # Check if download was successful
    from database import async_session_maker
    async with async_session_maker() as db:
        result = await db.execute(select(Paper).where(Paper.id == paper_id))
        paper = result.scalar_one_or_none()
        
        if paper and paper.download_status == "downloaded":
            await process_paper_task(paper_id)


async def batch_processing_task(paper_ids: List[int], download_only: bool):
    """Background task for batch processing"""
    for paper_id in paper_ids:
        try:
            if download_only:
                await download_paper_task(paper_id)
            else:
                await full_pipeline_task(paper_id)
        except Exception as e:
            logger.error(f"Failed to process paper {paper_id} in batch: {e}")
            continue
