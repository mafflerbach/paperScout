import os
import re
import httpx
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import logging
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Service for downloading and processing PDF files"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or settings.pdf_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def download_pdf(self, url: str, paper_id: int) -> str:
        """Download PDF from URL and save to filesystem"""
        try:
            # Generate filename
            filename = f"paper_{paper_id}.pdf"
            file_path = self.storage_path / filename
            
            # Download with httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                logger.info(f"Downloading PDF from {url}")
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Save to file
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"PDF saved to {file_path}")
                return str(file_path)
                
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            raise
    
    async def extract_text(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)
            
            doc.close()
            
            # Join all pages
            full_text = "\n\n".join(text_content)
            
            # Basic cleaning
            full_text = self._clean_text(full_text)
            
            logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    async def split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """Split text into logical sections"""
        try:
            sections = []
            
            # Common academic paper section patterns
            section_patterns = [
                (r'(?i)^abstract\s*', 'abstract'),
                (r'(?i)^(1\.?\s*)?introduction\s*', 'introduction'),
                (r'(?i)^(2\.?\s*)?related\s+work\s*', 'related_work'),
                (r'(?i)^(3\.?\s*)?methodology?\s*', 'methodology'),
                (r'(?i)^(3\.?\s*)?methods?\s*', 'methodology'),
                (r'(?i)^(4\.?\s*)?results?\s*', 'results'),
                (r'(?i)^(4\.?\s*)?experiments?\s*', 'results'),
                (r'(?i)^(5\.?\s*)?discussion\s*', 'discussion'),
                (r'(?i)^(6\.?\s*)?conclusion\s*', 'conclusion'),
                (r'(?i)^references\s*', 'references'),
                (r'(?i)^acknowledgments?\s*', 'acknowledgments'),
            ]
            
            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            current_section = {
                'type': 'unknown',
                'title': None,
                'content': []
            }
            
            for paragraph in paragraphs:
                # Check if this paragraph is a section header
                is_section_header = False
                
                for pattern, section_type in section_patterns:
                    if re.match(pattern, paragraph.strip()):
                        # Save previous section if it has content
                        if current_section['content']:
                            sections.append({
                                'type': current_section['type'],
                                'title': current_section['title'],
                                'content': '\n\n'.join(current_section['content'])
                            })
                        
                        # Start new section
                        current_section = {
                            'type': section_type,
                            'title': paragraph.strip(),
                            'content': []
                        }
                        is_section_header = True
                        break
                
                # If not a section header, add to current section content
                if not is_section_header:
                    current_section['content'].append(paragraph)
            
            # Add the last section
            if current_section['content']:
                sections.append({
                    'type': current_section['type'],
                    'title': current_section['title'],
                    'content': '\n\n'.join(current_section['content'])
                })
            
            # If we didn't find proper sections, split by length
            if len(sections) == 1 and sections[0]['type'] == 'unknown':
                logger.warning("No clear sections found, splitting by length")
                sections = self._split_by_length(text)
            
            logger.info(f"Split text into {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Failed to split text into sections: {e}")
            # Fallback: return entire text as one section
            return [{
                'type': 'full_text',
                'title': 'Full Document',
                'content': text
            }]
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common artifacts
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Fix common PDF extraction issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('−', '-')
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _split_by_length(self, text: str, max_chars: int = 3000) -> List[Dict[str, str]]:
        """Fallback method to split text by length"""
        sections = []
        words = text.split()
        current_section = []
        current_length = 0
        section_num = 1
        
        for word in words:
            if current_length + len(word) > max_chars and current_section:
                # Save current section
                sections.append({
                    'type': f'section_{section_num}',
                    'title': f'Section {section_num}',
                    'content': ' '.join(current_section)
                })
                
                # Start new section
                current_section = [word]
                current_length = len(word)
                section_num += 1
            else:
                current_section.append(word)
                current_length += len(word) + 1  # +1 for space
        
        # Add remaining content
        if current_section:
            sections.append({
                'type': f'section_{section_num}',
                'title': f'Section {section_num}',
                'content': ' '.join(current_section)
            })
        
        return sections
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        """Get PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            info = {
                'page_count': len(doc),
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'file_size': os.path.getsize(pdf_path)
            }
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get PDF info for {pdf_path}: {e}")
            return {}


# Global service instance
_pdf_processor = None


@lru_cache(maxsize=1)
def get_pdf_processor() -> PDFProcessor:
    """Get or create the global PDF processor instance"""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor
