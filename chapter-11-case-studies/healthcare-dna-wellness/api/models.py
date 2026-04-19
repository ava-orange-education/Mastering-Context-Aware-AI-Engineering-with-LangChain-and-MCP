"""
API models for healthcare DNA wellness
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class GeneticVariant(BaseModel):
    """Genetic variant input"""
    gene: str = Field(..., description="Gene name (e.g., MTHFR)")
    variant: str = Field(..., description="Variant identifier (e.g., C677T)")
    zygosity: str = Field(..., description="Zygosity (homozygous, heterozygous)")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Quality score")
    
    @validator('zygosity')
    def validate_zygosity(cls, v):
        allowed = ['homozygous', 'heterozygous', 'hemizygous', 'unknown']
        if v.lower() not in allowed:
            raise ValueError(f"Zygosity must be one of: {allowed}")
        return v.lower()


class PatientContext(BaseModel):
    """Patient context for recommendations"""
    age: Optional[int] = Field(None, ge=0, le=120)
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    gender: Optional[str] = None


class DNAAnalysisRequest(BaseModel):
    """Request for DNA analysis"""
    patient_id: str = Field(..., description="Patient identifier")
    genetic_variants: List[GeneticVariant] = Field(..., min_items=1)
    patient_context: Optional[PatientContext] = None
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("Patient ID must be at least 3 characters")
        return v


class ClinicalGuidelineQuery(BaseModel):
    """Request for clinical guideline search"""
    query: Optional[str] = None
    gene: Optional[str] = None
    variant: Optional[str] = None
    condition: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)
    evidence_level: Optional[str] = None
    
    @validator('evidence_level')
    def validate_evidence_level(cls, v):
        if v and v.lower() not in ['high', 'moderate', 'low']:
            raise ValueError("Evidence level must be: high, moderate, or low")
        return v.lower() if v else None


class WellnessPlanRequest(BaseModel):
    """Request for wellness plan generation"""
    patient_id: str
    genetic_variants: List[GeneticVariant]
    patient_context: PatientContext
    focus_areas: List[str] = Field(
        default=['nutrition', 'supplements', 'lifestyle', 'monitoring']
    )
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        allowed = ['nutrition', 'supplements', 'lifestyle', 'monitoring', 'exercise']
        for area in v:
            if area.lower() not in allowed:
                raise ValueError(f"Focus area must be one of: {allowed}")
        return [area.lower() for area in v]


class AnalyzedVariant(BaseModel):
    """Analyzed genetic variant"""
    gene: str
    variant: str
    zygosity: str
    clinical_significance: str
    interpretation: str
    confidence: float


class DNAAnalysisResponse(BaseModel):
    """Response from DNA analysis"""
    status: str
    patient_id: str
    summary: str
    variants: List[AnalyzedVariant]
    requires_review: bool
    confidence_score: float
    timestamp: datetime


class ClinicalGuideline(BaseModel):
    """Clinical guideline result"""
    source: str
    title: str
    content: str
    year: Optional[str]
    evidence_level: Optional[str]
    relevance_score: float


class GuidelinesResponse(BaseModel):
    """Response from guideline search"""
    status: str
    query: str
    guidelines: List[ClinicalGuideline]
    summary: str
    confidence_score: float
    timestamp: datetime


class Recommendation(BaseModel):
    """Individual wellness recommendation"""
    category: str
    recommendation: str
    priority: str
    evidence_basis: str
    implementation_steps: List[str]
    precautions: List[str]


class WellnessPlanResponse(BaseModel):
    """Response from wellness plan generation"""
    status: str
    patient_id: str
    recommendations: List[Recommendation]
    safety_check: Dict[str, Any]
    accuracy_validation: Dict[str, Any]
    requires_physician_review: bool
    confidence_score: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = "error"
    error_type: str
    error_message: str
    detail: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)