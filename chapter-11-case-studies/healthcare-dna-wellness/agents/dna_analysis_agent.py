"""
DNA Analysis Agent

Analyzes genetic variants and interprets their clinical significance
"""

from typing import Dict, Any, List
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class DNAAnalysisAgent(BaseAgent):
    """
    Agent for analyzing DNA variants and determining clinical significance
    """
    
    def __init__(self):
        super().__init__(
            name="DNA Analysis Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3  # Lower temperature for medical accuracy
        )
        
        # Variant databases (simplified for demo)
        self.variant_database = self._load_variant_database()
    
    def _load_variant_database(self) -> Dict[str, Dict]:
        """
        Load known variant database
        In production, integrate with ClinVar, dbSNP, etc.
        """
        return {
            "MTHFR:C677T": {
                "gene": "MTHFR",
                "variant": "C677T",
                "rsid": "rs1801133",
                "clinical_significance": "common polymorphism",
                "frequency": 0.32,
                "impact": "reduced enzyme activity",
                "conditions": ["elevated homocysteine", "folate metabolism"]
            },
            "APOE:E4": {
                "gene": "APOE",
                "variant": "E4 allele",
                "rsid": "rs429358",
                "clinical_significance": "risk factor",
                "frequency": 0.14,
                "impact": "increased Alzheimer's risk",
                "conditions": ["Alzheimer's disease", "cardiovascular disease"]
            },
            "CYP2C19:*2": {
                "gene": "CYP2C19",
                "variant": "*2 allele",
                "rsid": "rs4244285",
                "clinical_significance": "poor metabolizer",
                "frequency": 0.15,
                "impact": "reduced drug metabolism",
                "conditions": ["clopidogrel resistance", "PPI metabolism"]
            }
        }
    
    def _get_system_prompt(self) -> str:
        """System prompt for DNA analysis"""
        return """You are a medical genetics expert analyzing DNA variants.

Your role:
1. Interpret genetic variants and their clinical significance
2. Identify potential health implications
3. Determine zygosity effects (homozygous vs heterozygous)
4. Flag variants requiring additional medical consultation
5. Provide evidence-based interpretations

Guidelines:
- Use only established scientific evidence
- Clearly distinguish common polymorphisms from pathogenic variants
- Note when variants have incomplete penetrance
- Indicate confidence levels for interpretations
- Flag any variants requiring immediate medical attention
- Never diagnose - only provide genetic information

Output format:
- Variant interpretation
- Clinical significance
- Potential health implications
- Confidence level (high/medium/low)
- Recommendation for follow-up"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process genetic variants
        
        Args:
            input_data: {
                "patient_id": str,
                "genetic_variants": List[{
                    "gene": str,
                    "variant": str,
                    "zygosity": str,
                    "quality_score": float (optional)
                }]
            }
        
        Returns:
            AgentResponse with variant analysis
        """
        patient_id = input_data.get("patient_id")
        variants = input_data.get("genetic_variants", [])
        
        logger.info(f"Analyzing {len(variants)} variants for patient {patient_id}")
        
        # Analyze each variant
        analyzed_variants = []
        
        for variant in variants:
            variant_key = f"{variant['gene']}:{variant['variant']}"
            
            # Check known database
            if variant_key in self.variant_database:
                db_info = self.variant_database[variant_key]
                
                # Prepare context for LLM
                context = self._prepare_variant_context(variant, db_info)
                
                # Get LLM interpretation
                interpretation = await self._interpret_variant(context)
                
                analyzed_variants.append({
                    "gene": variant["gene"],
                    "variant": variant["variant"],
                    "zygosity": variant.get("zygosity", "unknown"),
                    "database_info": db_info,
                    "interpretation": interpretation,
                    "quality_score": variant.get("quality_score", 1.0)
                })
            else:
                # Unknown variant - flag for review
                analyzed_variants.append({
                    "gene": variant["gene"],
                    "variant": variant["variant"],
                    "zygosity": variant.get("zygosity", "unknown"),
                    "interpretation": "Unknown variant - requires genetic counselor review",
                    "quality_score": variant.get("quality_score", 1.0),
                    "requires_review": True
                })
        
        # Generate summary
        summary = await self._generate_summary(analyzed_variants)
        
        return AgentResponse(
            content=summary,
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "patient_id": patient_id,
                "variants_analyzed": len(analyzed_variants),
                "variants": analyzed_variants,
                "requires_review": any(v.get("requires_review") for v in analyzed_variants)
            },
            confidence=self._calculate_confidence(analyzed_variants)
        )
    
    def _prepare_variant_context(
        self,
        variant: Dict[str, Any],
        db_info: Dict[str, Any]
    ) -> str:
        """Prepare context for variant interpretation"""
        
        context = f"""
Genetic Variant Analysis:

Gene: {variant['gene']}
Variant: {variant['variant']}
Zygosity: {variant.get('zygosity', 'unknown')}

Database Information:
- rsID: {db_info.get('rsid', 'N/A')}
- Clinical Significance: {db_info['clinical_significance']}
- Population Frequency: {db_info['frequency']}
- Functional Impact: {db_info['impact']}
- Associated Conditions: {', '.join(db_info['conditions'])}

Quality Score: {variant.get('quality_score', 'N/A')}
"""
        return context
    
    async def _interpret_variant(self, context: str) -> str:
        """Get LLM interpretation of variant"""
        
        messages = [
            {
                "role": "user",
                "content": f"{context}\n\nProvide a clinical interpretation of this variant, including:\n1. What this variant means\n2. Potential health implications\n3. Zygosity effects\n4. Recommendations for follow-up"
            }
        ]
        
        interpretation = await self._call_llm(messages)
        return interpretation
    
    async def _generate_summary(self, analyzed_variants: List[Dict]) -> str:
        """Generate overall summary of analysis"""
        
        summary_context = f"""
Analyzed {len(analyzed_variants)} genetic variants.

Variants:
{self._format_variants_for_summary(analyzed_variants)}

Generate a concise summary highlighting:
1. Most significant findings
2. Variants requiring medical follow-up
3. Common polymorphisms noted
4. Overall genetic profile summary
"""
        
        messages = [
            {
                "role": "user",
                "content": summary_context
            }
        ]
        
        summary = await self._call_llm(messages)
        return summary
    
    def _format_variants_for_summary(self, variants: List[Dict]) -> str:
        """Format variants for summary generation"""
        
        formatted = []
        for v in variants:
            formatted.append(f"- {v['gene']} {v['variant']} ({v['zygosity']})")
        
        return "\n".join(formatted)
    
    def _calculate_confidence(self, analyzed_variants: List[Dict]) -> float:
        """Calculate overall confidence score"""
        
        if not analyzed_variants:
            return 0.0
        
        # Factors affecting confidence:
        # - Quality scores
        # - Known vs unknown variants
        # - Database coverage
        
        quality_scores = [
            v.get("quality_score", 0.5)
            for v in analyzed_variants
        ]
        
        known_variants = [
            v for v in analyzed_variants
            if not v.get("requires_review", False)
        ]
        
        quality_confidence = sum(quality_scores) / len(quality_scores)
        coverage_confidence = len(known_variants) / len(analyzed_variants)
        
        # Weighted average
        confidence = (quality_confidence * 0.6) + (coverage_confidence * 0.4)
        
        return round(confidence, 2)