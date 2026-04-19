"""
Clinical Guidelines Agent

Retrieves and interprets clinical guidelines using RAG
"""

from typing import Dict, Any, List
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse
from rag.medical_knowledge_retriever import MedicalKnowledgeRetriever

logger = logging.getLogger(__name__)


class ClinicalGuidelinesAgent(BaseAgent):
    """
    Agent for retrieving and interpreting clinical guidelines
    """
    
    def __init__(self):
        super().__init__(
            name="Clinical Guidelines Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.2  # Very low temperature for medical accuracy
        )
        
        # Initialize RAG system
        self.retriever = MedicalKnowledgeRetriever()
    
    def _get_system_prompt(self) -> str:
        """System prompt for clinical guidelines"""
        return """You are a medical literature expert specializing in clinical guidelines.

Your role:
1. Retrieve relevant clinical guidelines and research
2. Synthesize evidence from multiple sources
3. Cite sources accurately
4. Distinguish between established guidelines and emerging research
5. Note levels of evidence (strong, moderate, weak)

Guidelines:
- Always cite sources with publication year
- Distinguish between correlation and causation
- Note when guidelines conflict or are unclear
- Flag outdated information
- Indicate strength of evidence
- Never extrapolate beyond what evidence supports

Output format:
- Summary of relevant guidelines
- Level of evidence
- Source citations
- Conflicting information (if any)
- Recommendations based on current evidence"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Retrieve and interpret clinical guidelines
        
        Args:
            input_data: {
                "query": str or
                "variants": List[Dict],
                "conditions": List[str],
                "top_k": int (optional)
            }
        
        Returns:
            AgentResponse with clinical guidelines
        """
        query = input_data.get("query")
        variants = input_data.get("variants", [])
        conditions = input_data.get("conditions", [])
        top_k = input_data.get("top_k", 5)
        
        # Construct query if not provided
        if not query:
            query = self._construct_query(variants, conditions)
        
        logger.info(f"Searching clinical guidelines: {query}")
        
        # Retrieve guidelines using RAG
        await self.retriever.initialize()
        search_results = await self.retriever.search(
            query=query,
            top_k=top_k,
            filters={"document_type": "clinical_guideline"}
        )
        
        # Format retrieved guidelines
        guidelines_context = self._format_guidelines(search_results)
        
        # Get LLM interpretation
        interpretation = await self._interpret_guidelines(query, guidelines_context)
        
        # Extract citations
        citations = self._extract_citations(search_results)
        
        return AgentResponse(
            content=interpretation,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "query": query,
                "guidelines_retrieved": len(search_results),
                "citations": citations
            },
            confidence=self._calculate_confidence(search_results),
            sources=[result.document.id for result in search_results]
        )
    
    def _construct_query(
        self,
        variants: List[Dict],
        conditions: List[str]
    ) -> str:
        """Construct search query from variants and conditions"""
        
        parts = []
        
        # Add variant information
        for variant in variants:
            gene = variant.get("gene", "")
            var = variant.get("variant", "")
            if gene and var:
                parts.append(f"{gene} {var}")
        
        # Add conditions
        parts.extend(conditions)
        
        # Add general terms
        parts.extend(["clinical guidelines", "management", "treatment"])
        
        return " ".join(parts)
    
    def _format_guidelines(self, search_results: List) -> str:
        """Format retrieved guidelines for LLM"""
        
        formatted = []
        
        for i, result in enumerate(search_results, 1):
            doc = result.document
            formatted.append(f"""
Guideline {i} (Relevance: {result.score:.2f}):
Source: {doc.metadata.get('source', 'Unknown')}
Publication Year: {doc.metadata.get('year', 'Unknown')}
Evidence Level: {doc.metadata.get('evidence_level', 'Unknown')}

Content:
{doc.content}

---
""")
        
        return "\n".join(formatted)
    
    async def _interpret_guidelines(
        self,
        query: str,
        guidelines_context: str
    ) -> str:
        """Interpret clinical guidelines using LLM"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Query: {query}

Retrieved Clinical Guidelines:
{guidelines_context}

Synthesize these guidelines to provide:
1. Summary of current clinical recommendations
2. Level of evidence for each recommendation
3. Areas of consensus
4. Areas of uncertainty or conflicting evidence
5. Key citations to support recommendations

Focus on evidence-based recommendations only."""
            }
        ]
        
        interpretation = await self._call_llm(messages)
        return interpretation
    
    def _extract_citations(self, search_results: List) -> List[Dict]:
        """Extract citation information"""
        
        citations = []
        
        for result in search_results:
            doc = result.document
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "title": doc.metadata.get("title", "Unknown"),
                "evidence_level": doc.metadata.get("evidence_level", "Unknown"),
                "relevance_score": round(result.score, 3)
            })
        
        return citations
    
    def _calculate_confidence(self, search_results: List) -> float:
        """Calculate confidence based on search results"""
        
        if not search_results:
            return 0.0
        
        # Confidence factors:
        # - Average relevance score
        # - Number of high-quality results
        # - Consistency across sources
        
        avg_score = sum(r.score for r in search_results) / len(search_results)
        
        high_quality = sum(
            1 for r in search_results
            if r.document.metadata.get("evidence_level") in ["high", "strong"]
        )
        
        quality_ratio = high_quality / len(search_results)
        
        # Weighted confidence
        confidence = (avg_score * 0.7) + (quality_ratio * 0.3)
        
        return round(confidence, 2)