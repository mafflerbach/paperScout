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


# ===== REFINEMENT BACKGROUND TASKS =====

async def refine_paper_summaries_task(paper_id: int, styles: List[str]):
    """Background task to refine all summaries for a paper"""
    try:
        from database import async_session_maker
        import asyncio
        
        async with async_session_maker() as db:
            # Get all sections for the paper
            result = await db.execute(
                select(PaperSection).where(PaperSection.paper_id == paper_id)
            )
            sections = result.scalars().all()
            
            if not sections:
                logger.warning(f"No sections found for paper {paper_id}")
                return
            
            logger.info(f"Starting refinement of {len(sections)} sections for paper {paper_id}")
            
            # Process each section sequentially to be nice to M1 Mac
            for section in sections:
                if not section.summary:
                    logger.warning(f"Section {section.id} has no summary to refine")
                    continue
                
                await db.execute(
                    update(PaperSection)
                    .where(PaperSection.id == section.id)
                    .values(refinement_status="processing")
                )
                await db.commit()
                
                # Refine for each style
                for style in styles:
                    try:
                        refined_summary = await refine_single_style_task(section.summary, style)
                        
                        # Update the appropriate column
                        update_values = {}
                        if style == 'academic':
                            update_values['summary_academic'] = refined_summary
                        elif style == 'business':
                            update_values['summary_business'] = refined_summary
                        elif style == 'social':
                            update_values['summary_social'] = refined_summary
                        elif style == 'educational':
                            update_values['summary_educational'] = refined_summary
                        
                        await db.execute(
                            update(PaperSection)
                            .where(PaperSection.id == section.id)
                            .values(**update_values)
                        )
                        await db.commit()
                        
                        logger.info(f"Refined {style} summary for section {section.id}")
                        
                        # Small delay to be nice to M1 Mac
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Failed to refine {style} summary for section {section.id}: {e}")
                        continue
                
                # Mark section as completed
                await db.execute(
                    update(PaperSection)
                    .where(PaperSection.id == section.id)
                    .values(
                        refinement_status="completed",
                        refined_at=func.now()
                    )
                )
                await db.commit()
            
            logger.info(f"Completed refinement for paper {paper_id}")
            
    except Exception as e:
        logger.error(f"refine_paper_summaries_task - Failed to refine paper {paper_id}: {e}")


async def refine_section_summary_task(section_id: int, style: str):
    """Background task to refine a single section summary in a specific style"""
    try:
        from database import async_session_maker
        
        async with async_session_maker() as db:
            # Get section
            result = await db.execute(select(PaperSection).where(PaperSection.id == section_id))
            section = result.scalar_one_or_none()
            
            if not section:
                logger.error(f"Section {section_id} not found for refinement")
                return
            
            if not section.summary:
                logger.error(f"Section {section_id} has no summary to refine")
                return
            
            # Refine the summary
            refined_summary = await refine_single_style_task(section.summary, style)
            
            # Update the appropriate column
            update_values = {'refinement_status': 'processing'}
            if style == 'academic':
                update_values['summary_academic'] = refined_summary
            elif style == 'business':
                update_values['summary_business'] = refined_summary
            elif style == 'social':
                update_values['summary_social'] = refined_summary
            elif style == 'educational':
                update_values['summary_educational'] = refined_summary
            
            await db.execute(
                update(PaperSection)
                .where(PaperSection.id == section_id)
                .values(**update_values)
            )
            await db.commit()
            
            logger.info(f"Successfully refined {style} summary for section {section_id}")
            
    except Exception as e:
        logger.error(f"refine_section_summary_task - Failed to refine section {section_id}: {e}")


async def batch_refine_summaries_task(paper_ids: List[int], styles: List[str]):
    """Background task for batch refinement"""
    import asyncio
    
    logger.info(f"Starting batch refinement for {len(paper_ids)} papers")
    
    # Process papers in small batches to respect M1 memory
    batch_size = 2
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        
        # Process current batch
        tasks = [refine_paper_summaries_task(paper_id, styles) for paper_id in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Small break between batches
        await asyncio.sleep(1.0)
        logger.info(f"Completed batch {i//batch_size + 1}/{(len(paper_ids) + batch_size - 1)//batch_size}")
    
    logger.info(f"Completed batch refinement for {len(paper_ids)} papers")


async def refine_single_style_task(original_summary: str, style: str) -> str:
    """Helper function to refine a summary in a specific style"""
    
    REFINEMENT_PROMPTS = {
        'academic': """
Transform this whitepaper summary for peer researchers and academics:
Original: {summary}

Requirements:
- Preserve exact methodological approaches and experimental design details
- Include statistical significance, sample sizes, and confidence intervals where applicable
- Highlight novel contributions to the field and theoretical implications
- Use discipline-specific terminology and maintain academic rigor
- Structure: [Research objective] → [Key methodology] → [Primary findings with quantitative results]
Maximum 3 sentences.

Refined summary:""",
        
        'business': """
Convert this academic whitepaper summary for C-suite executives and business strategists:
Original: {summary}

Requirements:
- Translate research findings into market opportunities and competitive advantages
- Quantify potential ROI, cost savings, or revenue impact where possible
- Identify implementation timelines and resource requirements
- Address scalability and risk factors
- Focus on: What can we do with this? How much will it cost/save? When can we implement?
Maximum 3 sentences.

Refined summary:""",
        
        'social': """
Adapt this whitepaper summary for LinkedIn thought leadership:
Original: {summary}

Requirements:
- Lead with a compelling hook about industry transformation or breakthrough
- Use conversational yet authoritative tone
- Frame findings as actionable insights for professional networks
- Structure: [Attention-grabbing insight] → [Key implication for industry] → [Call to engagement]
- End with a comprehensive taglist of 8-12 relevant hashtags covering: research topic, industry, methodology, and trending keywords

Refined summary:""",
        
        'educational': """
Simplify this whitepaper summary for graduate students and early-career researchers:
Original: {summary}

Requirements:
- Define technical terms and explain complex methodologies in accessible language
- Connect findings to broader field context and real-world applications
- Highlight learning opportunities and career relevance
- Use encouraging language that builds confidence in the subject matter
- Structure: [What was studied and why] → [How it was done (simplified)] → [What it means for the field]
Maximum 3 sentences.

Refined summary:"""
    }
    
    try:
        from services.llm_processor import get_llm_processor
        
        llm_processor = get_llm_processor()
        prompt = REFINEMENT_PROMPTS[style].format(summary=original_summary)
        
        refined = await llm_processor._call_llm(prompt, max_tokens=300)
        return refined.strip()
        
    except Exception as e:
        logger.error(f"Failed to refine summary in {style} style: {e}")
        # Return original as fallback
        return original_summary


# ===== SUMMARY REFINEMENT ENDPOINTS =====

@router.post("/papers/{paper_id}/refine-summaries")
async def refine_paper_summaries(
    paper_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    styles: List[str] = None  # ['academic', 'business', 'social', 'educational']
):
    """Refine summaries for a specific paper in multiple styles"""
    # Check if paper exists and has been processed
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if paper.processing_status != "completed":
        raise HTTPException(status_code=400, detail="Paper must be processed first")
    
    # Default to all styles if none specified
    if not styles:
        styles = ['academic', 'business', 'social', 'educational']
    
    # Validate styles
    valid_styles = {'academic', 'business', 'social', 'educational'}
    invalid_styles = set(styles) - valid_styles
    if invalid_styles:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid styles: {invalid_styles}. Valid: {valid_styles}"
        )
    
    # Add to background task
    background_tasks.add_task(refine_paper_summaries_task, paper_id, styles)
    
    return {
        "message": "Summary refinement started", 
        "paper_id": paper_id,
        "styles": styles
    }


@router.post("/papers/{paper_id}/sections/{section_id}/refine")
async def refine_section_summary(
    paper_id: int,
    section_id: int,
    style: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Refine a specific section summary in a specific style"""
    # Validate style
    valid_styles = {'academic', 'business', 'social', 'educational'}
    if style not in valid_styles:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid style: {style}. Valid: {valid_styles}"
        )
    
    # Check if section exists
    result = await db.execute(
        select(PaperSection).where(
            PaperSection.id == section_id,
            PaperSection.paper_id == paper_id
        )
    )
    section = result.scalar_one_or_none()
    
    if not section:
        raise HTTPException(status_code=404, detail="Section not found")
    
    if not section.summary:
        raise HTTPException(status_code=400, detail="Section has no summary to refine")
    
    # Add to background task
    background_tasks.add_task(refine_section_summary_task, section_id, style)
    
    return {
        "message": "Section refinement started",
        "section_id": section_id,
        "style": style
    }


@router.post("/batch/refine-summaries")
async def batch_refine_summaries(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    styles: List[str] = None,
    paper_ids: List[int] = None,
    limit: int = 10
):
    """Batch refine summaries for multiple papers"""
    # Default styles
    if not styles:
        styles = ['academic', 'business', 'social', 'educational']
    
    # Get papers to process
    if paper_ids:
        # Specific papers
        query = select(Paper).where(
            Paper.id.in_(paper_ids),
            Paper.processing_status == "completed"
        )
    else:
        # All completed papers without refined summaries
        query = select(Paper).where(
            Paper.processing_status == "completed"
        ).limit(limit)
    
    result = await db.execute(query)
    papers = result.scalars().all()
    
    if not papers:
        return {"message": "No papers found for refinement", "papers_processed": 0}
    
    # Add to background task
    background_tasks.add_task(
        batch_refine_summaries_task, 
        [p.id for p in papers], 
        styles
    )
    
    return {
        "message": "Batch refinement started",
        "papers_count": len(papers),
        "styles": styles,
        "paper_ids": [p.id for p in papers]
    }


@router.get("/papers/{paper_id}/refinement-status")
async def get_refinement_status(paper_id: int, db: AsyncSession = Depends(get_db)):
    """Get refinement status for a paper"""
    result = await db.execute(
        select(PaperSection).where(PaperSection.paper_id == paper_id)
    )
    sections = result.scalars().all()
    
    if not sections:
        raise HTTPException(status_code=404, detail="Paper sections not found")
    
    # Count refinement status
    total_sections = len(sections)
    refined_sections = sum(1 for s in sections if s.refinement_status == "completed")
    
    # Check which styles are complete
    style_completion = {
        'academic': sum(1 for s in sections if s.summary_academic) / total_sections,
        'business': sum(1 for s in sections if s.summary_business) / total_sections,
        'social': sum(1 for s in sections if s.summary_social) / total_sections,
        'educational': sum(1 for s in sections if s.summary_educational) / total_sections,
    }
    
    return {
        "paper_id": paper_id,
        "total_sections": total_sections,
        "sections_with_refinements": refined_sections,
        "overall_progress": refined_sections / total_sections if total_sections > 0 else 0,
        "style_completion": style_completion
    }
