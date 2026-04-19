"""
Wellness Recommendation Agent

Generates personalized wellness recommendations based on genetic analysis and clinical guidelines
"""

from typing import Dict, Any, List
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class WellnessRecommendationAgent(BaseAgent):
    """
    Agent for generating personalized wellness recommendations
    """
    
    def __init__(self):
        super().__init__(
            name="Wellness Recommendation Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.4  # Moderate temperature for balanced creativity and accuracy
        )
    
    def _get_system_prompt(self) -> str:
        """System prompt for wellness recommendations"""
        return """You are a wellness advisor creating personalized recommendations based on genetic data and clinical guidelines.

Your role:
1. Synthesize genetic analysis and clinical guidelines
2. Generate actionable, evidence-based wellness recommendations
3. Prioritize recommendations by impact and feasibility
4. Consider patient preferences and constraints
5. Provide clear explanations for each recommendation

Guidelines:
- Base recommendations ONLY on provided genetic and clinical evidence
- Distinguish between strong evidence and preliminary research
- Include specific, actionable steps
- Note any contraindications or precautions
- Recommend consulting healthcare providers for medical decisions
- Never diagnose conditions or prescribe medications
- Focus on nutrition, lifestyle, and preventive measures

Recommendation categories:
1. Nutrition and diet
2. Supplements (if supported by evidence)
3. Physical activity
4. Lifestyle modifications
5. Monitoring and screening
6. When to consult healthcare providers

Output format:
- Clear, prioritized recommendations
- Evidence basis for each recommendation
- Practical implementation steps
- Precautions and contraindications
- Timeline for implementation"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate wellness recommendations
        
        Args:
            input_data: {
                "analysis": AgentResponse from DNA analysis,
                "guidelines": AgentResponse from clinical guidelines,
                "patient_context": Dict (age, conditions, medications, etc.),
                "focus_areas": List[str] (optional)
            }
        
        Returns:
            AgentResponse with wellness recommendations
        """
        analysis = input_data.get("analysis")
        guidelines = input_data.get("guidelines")
        patient_context = input_data.get("patient_context", {})
        focus_areas = input_data.get("focus_areas", [
            "nutrition", "supplements", "lifestyle", "monitoring"
        ])
        
        logger.info(f"Generating wellness recommendations for patient")
        
        # Validate inputs
        if not analysis or not guidelines:
            raise ValueError("Both DNA analysis and clinical guidelines required")
        
        # Extract key information
        variants = analysis.metadata.get("variants", [])
        citations = guidelines.metadata.get("citations", [])
        
        # Prepare context
        context = self._prepare_recommendation_context(
            variants=variants,
            guidelines_content=guidelines.content,
            citations=citations,
            patient_context=patient_context
        )
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(context, focus_areas)
        
        # Structure recommendations
        structured_recommendations = self._structure_recommendations(recommendations)
        
        # Safety check
        safety_check = await self._safety_check(structured_recommendations, patient_context)
        
        return AgentResponse(
            content=recommendations,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "structured_recommendations": structured_recommendations,
                "focus_areas": focus_areas,
                "safety_check": safety_check,
                "patient_context": patient_context,
                "variant_count": len(variants),
                "citation_count": len(citations)
            },
            confidence=self._calculate_confidence(analysis, guidelines),
            sources=guidelines.sources
        )
    
    def _prepare_recommendation_context(
        self,
        variants: List[Dict],
        guidelines_content: str,
        citations: List[Dict],
        patient_context: Dict[str, Any]
    ) -> str:
        """Prepare context for recommendation generation"""
        
        # Format variants
        variant_summary = "\n".join([
            f"- {v['gene']} {v['variant']} ({v['zygosity']}): {v.get('interpretation', 'N/A')}"
            for v in variants
        ])
        
        # Format patient context
        age = patient_context.get("age", "Unknown")
        conditions = patient_context.get("conditions", [])
        medications = patient_context.get("medications", [])
        allergies = patient_context.get("allergies", [])
        
        context = f"""
GENETIC ANALYSIS:
{variant_summary}

CLINICAL GUIDELINES:
{guidelines_content}

PATIENT CONTEXT:
Age: {age}
Existing Conditions: {', '.join(conditions) if conditions else 'None reported'}
Current Medications: {', '.join(medications) if medications else 'None reported'}
Allergies: {', '.join(allergies) if allergies else 'None reported'}

SUPPORTING CITATIONS:
{self._format_citations(citations)}
"""
        return context
    
    def _format_citations(self, citations: List[Dict]) -> str:
        """Format citations for context"""
        
        formatted = []
        for i, cite in enumerate(citations, 1):
            formatted.append(
                f"{i}. {cite.get('source', 'Unknown')} ({cite.get('year', 'N/A')}) - "
                f"Evidence Level: {cite.get('evidence_level', 'Unknown')}"
            )
        
        return "\n".join(formatted)
    
    async def _generate_recommendations(
        self,
        context: str,
        focus_areas: List[str]
    ) -> str:
        """Generate personalized recommendations"""
        
        focus_areas_text = ", ".join(focus_areas)
        
        messages = [
            {
                "role": "user",
                "content": f"""{context}

Based on the genetic analysis and clinical guidelines, generate personalized wellness recommendations.

Focus on these areas: {focus_areas_text}

Provide:
1. Specific, actionable recommendations
2. Evidence basis (cite the clinical guidelines)
3. Priority level (high, medium, low)
4. Implementation timeline
5. Any precautions or contraindications
6. When to consult healthcare providers

Remember: Focus on wellness and prevention, not diagnosis or treatment."""
            }
        ]
        
        recommendations = await self._call_llm(messages)
        return recommendations
    
    def _structure_recommendations(self, recommendations: str) -> Dict[str, List[Dict]]:
        """
        Structure recommendations into categories
        
        In production, use structured output or parsing
        For now, return simplified structure
        """
        
        # Simplified structure
        # In production, parse LLM output into structured format
        return {
            "nutrition": [],
            "supplements": [],
            "lifestyle": [],
            "monitoring": [],
            "medical_consultation": []
        }
    
    async def _safety_check(
        self,
        recommendations: Dict[str, List[Dict]],
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform safety check on recommendations
        """
        
        # Check for contraindications
        medications = patient_context.get("medications", [])
        allergies = patient_context.get("allergies", [])
        conditions = patient_context.get("conditions", [])
        
        safety_flags = []
        
        # Basic safety checks
        # In production, implement comprehensive drug interaction checking
        # and contraindication detection
        
        if "warfarin" in [m.lower() for m in medications]:
            safety_flags.append({
                "type": "drug_interaction",
                "severity": "high",
                "message": "Patient on warfarin - vitamin K recommendations require medical review"
            })
        
        if "pregnancy" in [c.lower() for c in conditions]:
            safety_flags.append({
                "type": "special_population",
                "severity": "high",
                "message": "Pregnant patient - all recommendations require OB approval"
            })
        
        return {
            "passed": len(safety_flags) == 0,
            "flags": safety_flags,
            "requires_review": len(safety_flags) > 0
        }
    
    def _calculate_confidence(
        self,
        analysis: AgentResponse,
        guidelines: AgentResponse
    ) -> float:
        """Calculate confidence in recommendations"""
        
        # Confidence based on:
        # - DNA analysis confidence
        # - Clinical guidelines confidence
        # - Number of supporting citations
        
        analysis_confidence = analysis.confidence or 0.5
        guidelines_confidence = guidelines.confidence or 0.5
        
        # Weighted average
        confidence = (analysis_confidence * 0.4) + (guidelines_confidence * 0.6)
        
        return round(confidence, 2)