"""
Synthesis agent for generating final responses
"""

import anthropic
from typing import Dict, Any
import logging
import time

from config.settings import settings

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Agent responsible for synthesizing final responses"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    async def synthesize(
        self,
        query: str,
        analysis: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """Synthesize final response from analysis"""
        start_time = time.time()
        
        try:
            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(
                query,
                analysis["analysis"],
                context
            )
            
            # Call Claude for synthesis
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            synthesized_response = response.content[0].text
            
            synthesis_time = time.time() - start_time
            
            logger.info(f"Synthesis completed in {synthesis_time:.3f}s")
            
            return {
                "response": synthesized_response,
                "processing_time": synthesis_time,
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _create_synthesis_prompt(
        self,
        query: str,
        analysis: str,
        context: str
    ) -> str:
        """Create prompt for synthesis"""
        return f"""Generate a clear, accurate response to the user's query based on the analysis and source documents.

Query: {query}

Analysis:
{analysis}

Source Context:
{context}

Requirements:
- Answer the query directly and comprehensively
- Ground your response in the provided documents
- Be clear, concise, and well-structured
- Cite sources when making specific claims
- Acknowledge if information is insufficient to fully answer

Response:"""
    
    async def health_check(self) -> bool:
        """Check if synthesis agent is healthy"""
        try:
            # Test API connection
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"Synthesis agent health check failed: {e}")
            return False