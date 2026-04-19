"""
Summarization Agent

Summarizes documents and conversations for quick understanding
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    """
    Agent for summarizing enterprise documents and conversations
    """
    
    def __init__(self):
        super().__init__(
            name="Summarization Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for summarization"""
        return """You are an expert at summarizing enterprise documents and conversations.

Your role:
1. Create concise, accurate summaries of documents
2. Extract key points and action items
3. Identify main stakeholders and decisions
4. Highlight important dates and deadlines
5. Note any follow-up items or dependencies

Guidelines:
- Maintain factual accuracy - never add information not in the source
- Use clear, professional language
- Organize information logically
- Include relevant context
- Note document type (meeting notes, report, proposal, etc.)
- Preserve important details (numbers, dates, names)

Summary structure:
- Executive summary (2-3 sentences)
- Key points (bullet points)
- Action items (if applicable)
- Stakeholders/participants
- Important dates/deadlines
- Follow-up items"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process summarization request
        
        Args:
            input_data: {
                "content": str,
                "document_type": str (optional),
                "max_length": int (optional),
                "focus_areas": List[str] (optional),
                "metadata": Dict (optional)
            }
        
        Returns:
            AgentResponse with summary
        """
        
        content = input_data.get("content")
        document_type = input_data.get("document_type", "document")
        max_length = input_data.get("max_length", 500)
        focus_areas = input_data.get("focus_areas", [])
        metadata = input_data.get("metadata", {})
        
        if not content:
            raise ValueError("Content is required for summarization")
        
        logger.info(f"Summarizing {document_type} ({len(content)} chars)")
        
        # Generate summary based on type
        if document_type == "meeting_notes":
            summary = await self._summarize_meeting(content, max_length)
        elif document_type == "email_thread":
            summary = await self._summarize_email_thread(content, max_length)
        elif document_type == "slack_conversation":
            summary = await self._summarize_conversation(content, max_length)
        elif document_type == "technical_document":
            summary = await self._summarize_technical_doc(content, max_length, focus_areas)
        else:
            summary = await self._summarize_general(content, max_length)
        
        # Extract structured information
        structured = self._extract_structured_info(summary)
        
        return AgentResponse(
            content=summary,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "document_type": document_type,
                "original_length": len(content),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(content), 2),
                "structured_info": structured,
                "source_metadata": metadata
            },
            confidence=0.9  # High confidence in summarization
        )
    
    async def _summarize_meeting(
        self,
        content: str,
        max_length: int
    ) -> str:
        """Summarize meeting notes"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Summarize these meeting notes in {max_length} words or less:

{content}

Include:
1. Executive summary
2. Key decisions made
3. Action items with owners
4. Next steps
5. Important dates"""
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    async def _summarize_email_thread(
        self,
        content: str,
        max_length: int
    ) -> str:
        """Summarize email thread"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Summarize this email thread in {max_length} words or less:

{content}

Include:
1. Main topic/purpose
2. Key participants and their positions
3. Current status or outcome
4. Any pending items or decisions needed"""
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    async def _summarize_conversation(
        self,
        content: str,
        max_length: int
    ) -> str:
        """Summarize Slack/chat conversation"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Summarize this conversation in {max_length} words or less:

{content}

Include:
1. Main topic discussed
2. Key points or conclusions
3. Any decisions or action items
4. Unresolved questions"""
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    async def _summarize_technical_doc(
        self,
        content: str,
        max_length: int,
        focus_areas: List[str]
    ) -> str:
        """Summarize technical document"""
        
        focus_text = f"\nFocus on: {', '.join(focus_areas)}" if focus_areas else ""
        
        messages = [
            {
                "role": "user",
                "content": f"""Summarize this technical document in {max_length} words or less:

{content}{focus_text}

Include:
1. Purpose and scope
2. Key technical details
3. Architecture or approach
4. Requirements or constraints
5. Next steps or recommendations"""
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    async def _summarize_general(
        self,
        content: str,
        max_length: int
    ) -> str:
        """Summarize general document"""
        
        messages = [
            {
                "role": "user",
                "content": f"""Summarize this document in {max_length} words or less:

{content}

Provide a clear, concise summary that captures the main points and key information."""
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    def _extract_structured_info(self, summary: str) -> Dict[str, Any]:
        """Extract structured information from summary"""
        
        # Simplified extraction
        # In production, use more sophisticated NLP or structured output
        
        structured = {
            "has_action_items": "action item" in summary.lower() or "todo" in summary.lower(),
            "has_deadlines": any(word in summary.lower() for word in ["deadline", "due", "by"]),
            "has_decisions": "decision" in summary.lower() or "decided" in summary.lower(),
            "mentions_stakeholders": any(word in summary.lower() for word in ["team", "stakeholder", "participant"])
        }
        
        return structured