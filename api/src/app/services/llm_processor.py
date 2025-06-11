import httpx
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from functools import lru_cache
from enum import Enum
import re

from config import settings

logger = logging.getLogger(__name__)


class LLMTask(Enum):
    METADATA = "metadata"  # Small, fast model for metadata extraction
    SUMMARY = "summary"    # Medium model for section summaries
    COMPREHENSIVE = "comprehensive"  # Large model for comprehensive analysis


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM fails"""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid"""
    pass


class LLMProcessor:
    """Service for LLM-based text processing with task-specific model selection"""
    
    def __init__(self):
        # Configure endpoints for different tasks
        self.endpoints = {
            LLMTask.METADATA: settings.metadata_llm_endpoint or settings.llm_endpoint,
            LLMTask.SUMMARY: settings.summary_llm_endpoint or settings.llm_endpoint,
            LLMTask.COMPREHENSIVE: settings.comprehensive_llm_endpoint or settings.llm_endpoint
        }
        
        # Configure models for different tasks
        self.models = {
            LLMTask.METADATA: settings.metadata_llm_model or "tinyllama-1.1b-chat",  # Fast, small model
            LLMTask.SUMMARY: settings.summary_llm_model or "llama2-7b-chat",         # Medium model
            LLMTask.COMPREHENSIVE: settings.comprehensive_llm_model or "llama2-13b-chat"  # Large model
        }
        
        # Configure timeouts and retries for different tasks
        self.timeouts = {
            LLMTask.METADATA: 10.0,  # Quick metadata tasks
            LLMTask.SUMMARY: 30.0,   # Medium complexity
            LLMTask.COMPREHENSIVE: 60.0  # Complex analysis
        }
        
        self.max_retries = {
            LLMTask.METADATA: 3,
            LLMTask.SUMMARY: 2,
            LLMTask.COMPREHENSIVE: 1
        }
        
        # Create separate clients for different timeouts
        self.clients = {
            task: httpx.AsyncClient(timeout=timeout)
            for task, timeout in self.timeouts.items()
        }

    async def _call_llm_with_retry(self, prompt: str, task: LLMTask, max_tokens: int = 500) -> str:
        """Make API call to LLM with retry logic and debug logging"""
        max_retries = self.max_retries[task]
        retry_delay = 1.0  # Start with 1 second delay
        
        # Debug: log prompt (truncated) once before retry loop
        prompt_snippet = prompt[:300].replace(chr(10), ' ')
        logger.debug(f"[LLM-{task.value}] Prompt (first 300 chars): {prompt_snippet}...")

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.debug(f"[LLM-{task.value}] Retry attempt {attempt + 1}/{max_retries}")
                response = await self._call_llm(prompt, task, max_tokens)
                # Debug: log response (truncated)
                response_snippet = response[:300].replace(chr(10), ' ')
                logger.debug(
                    f"[LLM-{task.value}] Response (first 300 chars): {response_snippet}..."
                )
                return response
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"[LLM-{task.value}] Call failed after {max_retries} attempts: {e}"
                    )
                    raise LLMConnectionError(f"Failed to connect to LLM service: {e}")

                logger.warning(
                    f"[LLM-{task.value}] Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            except Exception as e:
                logger.error(
                    f"[LLM-{task.value}] Unexpected error: {e}"
                )
                raise LLMError(f"Unexpected error: {e}")

    async def _call_llm(self, prompt: str, task: LLMTask, max_tokens: int = 500) -> str:
        """Make API call to appropriate LLM based on task"""
        try:
            payload = {
                "model": self.models[task],
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
            
            client = self.clients[task]
            endpoint = self.endpoints[task]
            
            # Extra debug information about endpoint and model
            logger.debug(
                f"[LLM-{task.value}] POST {endpoint} model={self.models[task]} max_tokens={max_tokens}"
            )

            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise LLMResponseError("Invalid response format from LLM")
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in LLM call for task {task.value}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM call for task {task.value}: {e}")
            raise

    async def extract_metadata(self, title: str, abstract: str, content_sample: str) -> Dict[str, Any]:
        """Extract metadata using fast, small model"""
        try:
            # Extract each component separately using the metadata-optimized model
            main_topics = await self._extract_topics(title, abstract)
            techniques = await self._extract_techniques(title, abstract, content_sample)
            applications = await self._extract_applications(title, abstract, content_sample)
            difficulty_level = await self._assess_difficulty(title, abstract, content_sample)
            implementation_feasibility = await self._assess_feasibility(title, abstract, content_sample)
            business_relevance = await self._assess_business_relevance(title, abstract)

            metadata = {
                "main_topics": main_topics,
                "techniques": techniques,
                "applications": applications,
                "difficulty_level": difficulty_level,
                "implementation_feasibility": implementation_feasibility,
                "business_relevance_score": business_relevance
            }

            return self._validate_metadata(metadata)

        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return self._get_fallback_metadata(title, abstract)

    async def _extract_topics(self, title: str, abstract: str) -> List[str]:
        """Extract main research topics using small model"""
        prompt = f"""List 2-4 main research topics/areas, one per line:
Title: {title}
Abstract: {abstract}
Topics:"""

        try:
            response = await self._call_llm(prompt, task=LLMTask.METADATA, max_tokens=50)
            topics = [t.strip() for t in response.strip().split('\n') if t.strip()]
            return topics[:4]
        except:
            return ["artificial intelligence"]

    async def _extract_techniques(self, title: str, abstract: str, content_sample: str) -> List[str]:
        """Extract specific techniques using small model"""
        prompt = f"""List 2-4 technical methods used, one per line:
Title: {title}
Abstract: {abstract}
Content: {content_sample[:300]}
Techniques:"""

        try:
            response = await self._call_llm(prompt, task=LLMTask.METADATA, max_tokens=50)
            techniques = [t.strip() for t in response.strip().split('\n') if t.strip()]
            return techniques[:4]
        except:
            return []

    async def _extract_applications(self, title: str, abstract: str, content_sample: str) -> List[str]:
        """Extract applications using small model"""
        prompt = f"""List 1-3 practical applications, one per line:
Title: {title}
Abstract: {abstract}
Content: {content_sample[:300]}
Applications:"""

        try:
            response = await self._call_llm(prompt, task=LLMTask.METADATA, max_tokens=50)
            applications = [a.strip() for a in response.strip().split('\n') if a.strip()]
            return applications[:3]
        except:
            return []

    async def _assess_difficulty(self, title: str, abstract: str, content_sample: str) -> str:
        """Assess difficulty using small model"""
        prompt = f"""Rate complexity, use just one of the three options: beginner, intermediate or advanced. No further reasons or explainations are needed:
Title: {title}
Abstract: {abstract}
Content: {content_sample[:300]}
Level:"""

        try:
            response = await self._call_llm(prompt, task=LLMTask.METADATA, max_tokens=20)
            level = response.strip().lower()
            return level if level in ["beginner", "intermediate", "advanced"] else "intermediate"
        except:
            return "intermediate"

    async def _assess_feasibility(self, title: str, abstract: str, content_sample: str) -> str:
        """Assess feasibility using small model"""
        prompt = f"""Rate implementation feasibility (low/medium/high), no further explaination or reason attached, answer with just one of the three Options:

Title: {title}
Abstract: {abstract}
Content: {content_sample[:300]}
Feasibility:"""

        try:
            response = await self._call_llm(prompt, task=LLMTask.METADATA, max_tokens=20)
            feasibility = response.strip().lower()
            return feasibility if feasibility in ["low", "medium", "high"] else "medium"
        except:
            return "medium"

    async def _assess_business_relevance(self, title: str, abstract: str) -> int:
        """Assess business relevance using small model"""
        prompt = f"""Rate business value just answer with a number between  1 and 10, no further explainations or reason attached:
Title: {title}
Abstract: {abstract}
Score:"""

        try:
            response = await self._call_llm_with_retry(prompt, task=LLMTask.METADATA, max_tokens=20)
            # Extract first integer from response
            match = re.search(r"\d+", response)
            if match:
                score = int(match.group())
                score_clamped = max(1, min(10, score))
                logger.debug(f"[Metadata] Parsed business relevance score: raw='{response.strip()}' parsed={score_clamped}")
                return score_clamped
            else:
                logger.debug(f"[Metadata] Could not parse integer from LLM response '{response.strip()}', defaulting to 5")
                return 5
        except Exception as e:
            logger.error(f"Failed to assess business relevance: {e}")
            return 5

    async def summarize_section(self, content: str, section_type: str) -> str:
        """Generate summary using medium-sized model with improved error handling"""
        try:
            prompt = self._get_summary_prompt(content, section_type)
            return await self._call_llm_with_retry(prompt, task=LLMTask.SUMMARY, max_tokens=1000)
        except LLMConnectionError:
            logger.warning(f"LLM connection failed for {section_type} summary, using fallback")
            return self._generate_fallback_summary(content)
        except Exception as e:
            logger.error(f"Failed to summarize {section_type} section: {e}")
            return self._generate_fallback_summary(content)

    def _generate_fallback_summary(self, content: str, max_length: int = 500) -> str:
        """Generate a simple fallback summary when LLM is unavailable"""
        if len(content) <= max_length:
            return content
        
        # Split into sentences (simple approach)
        sentences = content.split('. ')
        summary = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary.append(sentence)
            current_length += len(sentence) + 2  # +2 for '. '
        
        return '. '.join(summary) + ('...' if sentences else '')

    async def generate_comprehensive_summary(
        self,
        title: str,
        abstract: Optional[str],
        section_summaries: List[Dict[str, str]],
        style: Optional[str] = None
    ) -> str:
        """Generate comprehensive summary using large model"""
        try:
            sections_text = "\n\n".join([
                f"{s['type'].upper()}:\n{s['summary']}"
                for s in section_summaries
            ])
            
            prompt = self._get_comprehensive_prompt(title, abstract, sections_text, style)
            
            response = await self._call_llm(
                prompt,
                task=LLMTask.COMPREHENSIVE,
                max_tokens=2000
            )
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive summary: {e}")
            return "\n\n".join([
                f"{s['type'].upper()}:\n{s['summary']}"
                for s in section_summaries
            ])

    async def close(self):
        """Close all HTTP clients"""
        for client in self.clients.values():
            await client.aclose()

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


# Global service instance
_llm_processor = None


@lru_cache(maxsize=1)
def get_llm_processor() -> LLMProcessor:
    """Get or create the global LLM processor instance"""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMProcessor()
    return _llm_processor
