import httpx
import json
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


class LLMProcessor:
    """Service for LLM-based text processing"""
    
    def __init__(self, endpoint: str = None, model: str = None):
        self.endpoint = endpoint or settings.llm_endpoint
        self.model = model or settings.llm_model
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def summarize_section(self, content: str, section_type: str) -> str:
        """Generate summary for a paper section"""
        try:
            prompt = self._get_summary_prompt(content, section_type)
            
            response = await self._call_llm(prompt, max_tokens=1000)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize {section_type} section: {e}")
            # Return truncated content as fallback
            return content[:500] + "..." if len(content) > 500 else content
    
    async def extract_metadata(self, title: str, abstract: str, content_sample: str) -> Dict[str, Any]:
        try:
            prompt = self._get_metadata_prompt(title, abstract, content_sample)
            
            llm_response = await self._call_llm(prompt, max_tokens=500)  # Use descriptive name
            
            # Parse JSON response
            try:
                cleaned_response = llm_response.strip()
                
                # Fix incomplete JSON
                if not cleaned_response.endswith('}'):
                    cleaned_response += '}'
                
                metadata = json.loads(cleaned_response)
                return self._validate_metadata(metadata)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.error(f"Response was: {llm_response}")  # Use the correct variable name
                
                # Try to fix and parse again
                try:
                    fixed_response = llm_response.strip() + '}'
                    metadata = json.loads(fixed_response)
                    logger.info("Successfully fixed incomplete JSON")
                    return self._validate_metadata(metadata)
                except:
                    logger.error("Could not fix JSON, using fallback")
                    return self._get_fallback_metadata(title, abstract)
                    
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return self._get_fallback_metadata(title, abstract)
    
    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call to local LLM"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False
            }
            
            response = await self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise Exception("Invalid response format from LLM")
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _get_summary_prompt(self, content: str, section_type: str) -> str:
        """Generate prompt for section summarization"""
        section_guidance = {
            'abstract': "Focus on the main problem, approach, and key findings.",
            'introduction': "Highlight the motivation, problem statement, and research questions.",
            'methodology': "Summarize the technical approach, algorithms, and experimental setup.",
            'results': "Focus on key findings, performance metrics, and significant outcomes.",
            'discussion': "Highlight implications, limitations, and future directions.",
            'conclusion': "Capture the main contributions and takeaways.",
            'related_work': "Summarize how this work relates to existing research.",
        }
        
        guidance = section_guidance.get(section_type, "Provide a clear, concise summary of the main points.")
        
        return f"""Please provide a concise summary of this {section_type} section from an academic paper.

{guidance}

Keep the summary to 2-3 sentences and focus on the most important information for someone evaluating the paper's relevance and implementation potential.

Section content:
{content[:2000]}

Summary:"""
    
    def _get_metadata_prompt(self, title: str, abstract: str, content_sample: str) -> str:
        """Generate prompt for metadata extraction"""
        return f"""Analyze this research paper and extract structured metadata. Respond ONLY with valid JSON.

Paper Title: {title}
Abstract: {abstract}
Content Sample: {content_sample[:1300]}
Extract the following information and return ONLY valid JSON (no markdown, no code blocks):

{{
    "main_topics": ["list of 2-4 main research topics/areas"],
    "techniques": ["list of 2-4 specific techniques/methods used"],
    "applications": ["list of 1-3 practical applications/domains"],
    "difficulty_level": "beginner|intermediate|advanced",
    "implementation_feasibility": "low|medium|high",
    "business_relevance_score": 1-10
}}

Guidelines:
- main_topics: broad areas like "computer vision", "natural language processing", "reinforcement learning"
- techniques: specific methods like "transformer architecture", "convolutional neural networks", "attention mechanism"
- applications: practical domains like "autonomous driving", "medical imaging", "financial modeling"
- difficulty_level: how complex is this for a software developer to understand and implement
- implementation_feasibility: how realistic is it to implement this with current tools/resources
- business_relevance_score: potential impact and applicability (1=academic only, 10=high business value)

JSON:"""
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted metadata"""
        validated = {}
        
        # Validate main_topics
        if "main_topics" in metadata and isinstance(metadata["main_topics"], list):
            validated["main_topics"] = [str(topic)[:100] for topic in metadata["main_topics"][:5]]
        else:
            validated["main_topics"] = []
        
        # Validate techniques
        if "techniques" in metadata and isinstance(metadata["techniques"], list):
            validated["techniques"] = [str(tech)[:100] for tech in metadata["techniques"][:5]]
        else:
            validated["techniques"] = []
        
        # Validate applications
        if "applications" in metadata and isinstance(metadata["applications"], list):
            validated["applications"] = [str(app)[:100] for app in metadata["applications"][:5]]
        else:
            validated["applications"] = []
        
        # Validate difficulty_level
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        if metadata.get("difficulty_level") in valid_difficulties:
            validated["difficulty_level"] = metadata["difficulty_level"]
        else:
            validated["difficulty_level"] = "intermediate"  # Default
        
        # Validate implementation_feasibility
        valid_feasibilities = ["low", "medium", "high"]
        if metadata.get("implementation_feasibility") in valid_feasibilities:
            validated["implementation_feasibility"] = metadata["implementation_feasibility"]
        else:
            validated["implementation_feasibility"] = "medium"  # Default
        
        # Validate business_relevance_score
        try:
            score = int(metadata.get("business_relevance_score", 5))
            validated["business_relevance_score"] = max(1, min(10, score))
        except (ValueError, TypeError):
            validated["business_relevance_score"] = 5  # Default
        
        return validated
    
    def _get_fallback_metadata(self, title: str, abstract: str) -> Dict[str, Any]:
        """Generate basic fallback metadata when LLM fails"""
        # Simple keyword-based classification
        text = f"{title} {abstract}".lower()
        
        # Basic topic detection
        topics = []
        topic_keywords = {
            "computer vision": ["vision", "image", "visual", "detection", "recognition", "segmentation"],
            "natural language processing": ["nlp", "language", "text", "linguistic", "translation", "sentiment"],
            "machine learning": ["learning", "training", "model", "algorithm", "prediction"],
            "deep learning": ["deep", "neural", "network", "cnn", "rnn", "transformer"],
            "reinforcement learning": ["reinforcement", "reward", "policy", "agent", "environment"],
            "robotics": ["robot", "robotic", "manipulation", "navigation", "control"],
            "optimization": ["optimization", "minimize", "maximize", "optimal", "genetic"],
            "finance": ["financial", "trading", "market", "portfolio", "risk", "economic"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return {
            "main_topics": topics[:3] if topics else ["artificial intelligence"],
            "techniques": [],
            "applications": [],
            "difficulty_level": "intermediate",
            "implementation_feasibility": "medium",
            "business_relevance_score": 5
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Global service instance
_llm_processor = None


@lru_cache(maxsize=1)
def get_llm_processor() -> LLMProcessor:
    """Get or create the global LLM processor instance"""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMProcessor()
    return _llm_processor
