"""
API routes for healthcare DNA wellness
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from .models import (
    DNAAnalysisRequest,
    DNAAnalysisResponse,
    ClinicalGuidelineQuery,
    GuidelinesResponse,
    WellnessPlanRequest,
    WellnessPlanResponse,
    ErrorResponse,
    AnalyzedVariant,
    ClinicalGuideline,
    Recommendation
)
from agents.dna_analysis_agent import DNAAnalysisAgent
from agents.clinical_guidelines_agent import ClinicalGuidelinesAgent
from agents.wellness_recommendation_agent import WellnessRecommendationAgent
from agents.compliance_audit_agent import ComplianceAuditAgent
from evaluation.medical_accuracy_validator import MedicalAccuracyValidator
from evaluation.safety_checker import SafetyChecker
from evaluation.metrics import HealthcareMetrics

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize agents
dna_agent = DNAAnalysisAgent()
clinical_agent = ClinicalGuidelinesAgent()
wellness_agent = WellnessRecommendationAgent()
compliance_agent = ComplianceAuditAgent()
accuracy_validator = MedicalAccuracyValidator()
safety_checker = SafetyChecker()
metrics = HealthcareMetrics()


@router.post("/analyze-dna", response_model=DNAAnalysisResponse)
async def analyze_dna(request: DNAAnalysisRequest):
    """
    Analyze genetic variants
    
    Processes genetic variants to determine clinical significance
    and potential health implications.
    """
    
    try:
        # Convert request to agent input format
        agent_input = {
            "patient_id": request.patient_id,
            "genetic_variants": [
                {
                    "gene": v.gene,
                    "variant": v.variant,
                    "zygosity": v.zygosity,
                    "quality_score": v.quality_score or 1.0
                }
                for v in request.genetic_variants
            ]
        }
        
        # Process with DNA analysis agent
        result = await dna_agent.process(agent_input)
        
        # Log compliance
        await compliance_agent.process({
            "action": "analyze_dna",
            "user_id": "api_user",  # Would come from auth in production
            "patient_id": request.patient_id,
            "data_accessed": {"genetic_variants": len(request.genetic_variants)}
        })
        
        # Convert to response format
        analyzed_variants = [
            AnalyzedVariant(
                gene=v["gene"],
                variant=v["variant"],
                zygosity=v["zygosity"],
                clinical_significance=v.get("database_info", {}).get("clinical_significance", "unknown"),
                interpretation=v.get("interpretation", ""),
                confidence=v.get("quality_score", 0.5)
            )
            for v in result.metadata["variants"]
        ]
        
        return DNAAnalysisResponse(
            status="success",
            patient_id=request.patient_id,
            summary=result.content,
            variants=analyzed_variants,
            requires_review=result.metadata.get("requires_review", False),
            confidence_score=result.confidence or 0.5,
            timestamp=result.timestamp
        )
    
    except Exception as e:
        logger.error(f"DNA analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-guidelines", response_model=GuidelinesResponse)
async def search_guidelines(request: ClinicalGuidelineQuery):
    """
    Search clinical guidelines
    
    Retrieves relevant clinical guidelines from medical literature
    based on genetic variants or medical conditions.
    """
    
    try:
        # Construct query
        if request.query:
            query = request.query
        elif request.gene and request.variant:
            query = f"{request.gene} {request.variant} clinical guidelines"
        elif request.condition:
            query = f"{request.condition} clinical guidelines management"
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either query, gene+variant, or condition"
            )
        
        # Search guidelines
        agent_input = {
            "query": query,
            "top_k": request.top_k
        }
        
        result = await clinical_agent.process(agent_input)
        
        # Convert citations to guidelines
        guidelines = [
            ClinicalGuideline(
                source=cite["source"],
                title=cite.get("title", ""),
                content="",  # Content in full result
                year=cite.get("year"),
                evidence_level=cite.get("evidence_level"),
                relevance_score=cite.get("relevance_score", 0.0)
            )
            for cite in result.metadata.get("citations", [])
        ]
        
        return GuidelinesResponse(
            status="success",
            query=query,
            guidelines=guidelines,
            summary=result.content,
            confidence_score=result.confidence or 0.5,
            timestamp=result.timestamp
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guideline search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-wellness-plan", response_model=WellnessPlanResponse)
async def generate_wellness_plan(request: WellnessPlanRequest):
    """
    Generate personalized wellness plan
    
    Creates evidence-based wellness recommendations based on
    genetic analysis and clinical guidelines.
    """
    
    try:
        # First, analyze DNA
        dna_input = {
            "patient_id": request.patient_id,
            "genetic_variants": [
                {
                    "gene": v.gene,
                    "variant": v.variant,
                    "zygosity": v.zygosity,
                    "quality_score": v.quality_score or 1.0
                }
                for v in request.genetic_variants
            ]
        }
        
        dna_result = await dna_agent.process(dna_input)
        
        # Get clinical guidelines
        guidelines_input = {
            "variants": dna_result.metadata["variants"],
            "conditions": request.patient_context.conditions
        }
        
        guidelines_result = await clinical_agent.process(guidelines_input)
        
        # Generate recommendations
        wellness_input = {
            "analysis": dna_result,
            "guidelines": guidelines_result,
            "patient_context": {
                "age": request.patient_context.age,
                "conditions": request.patient_context.conditions,
                "medications": request.patient_context.medications,
                "allergies": request.patient_context.allergies
            },
            "focus_areas": request.focus_areas
        }
        
        wellness_result = await wellness_agent.process(wellness_input)
        
        # Validate medical accuracy
        accuracy_validation = await accuracy_validator.validate_recommendations(
            wellness_result.content,
            wellness_input["patient_context"]
        )
        
        # Check safety
        safety_check = await safety_checker.check_safety(
            wellness_result.content,
            wellness_input["patient_context"]
        )
        
        # Log compliance
        await compliance_agent.process({
            "action": "generate_wellness_plan",
            "user_id": "api_user",
            "patient_id": request.patient_id,
            "data_accessed": {
                "genetic_variants": len(request.genetic_variants),
                "clinical_data": True
            },
            "output_data": wellness_result.content
        })
        
        # Record metrics
        metrics.record_recommendation(
            recommendation_id=f"rec_{datetime.utcnow().timestamp()}",
            patient_id=request.patient_id,
            processing_time=2.5,  # Would track actual time
            accuracy_score=accuracy_validation["accuracy_score"],
            safety_score=safety_check["safety_score"],
            confidence_score=wellness_result.confidence or 0.5
        )
        
        # Parse recommendations (simplified)
        recommendations = [
            Recommendation(
                category=area,
                recommendation=f"Recommendations for {area}",
                priority="medium",
                evidence_basis="Clinical guidelines",
                implementation_steps=["Step 1", "Step 2"],
                precautions=["Consult healthcare provider"]
            )
            for area in request.focus_areas
        ]
        
        return WellnessPlanResponse(
            status="success",
            patient_id=request.patient_id,
            recommendations=recommendations,
            safety_check=safety_check,
            accuracy_validation=accuracy_validation,
            requires_physician_review=(
                not safety_check["safe"] or
                not accuracy_validation["overall_valid"] or
                safety_check["requires_physician_review"]
            ),
            confidence_score=wellness_result.confidence or 0.5,
            timestamp=wellness_result.timestamp
        )
    
    except Exception as e:
        logger.error(f"Wellness plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    
    try:
        summary = metrics.get_summary_metrics(days=7)
        quality = metrics.get_quality_metrics()
        
        return {
            "summary": summary,
            "quality": quality,
            "timestamp": datetime.utcnow()
        }
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))