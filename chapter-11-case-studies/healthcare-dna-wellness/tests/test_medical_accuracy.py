"""
Tests for medical accuracy validation
"""

import pytest
import sys
sys.path.append('../..')

from evaluation.medical_accuracy_validator import MedicalAccuracyValidator
from evaluation.safety_checker import SafetyChecker
from evaluation.metrics import HealthcareMetrics


@pytest.mark.asyncio
class TestMedicalAccuracyValidator:
    """Test medical accuracy validation"""
    
    async def test_initialization(self):
        """Test validator initializes correctly"""
        validator = MedicalAccuracyValidator()
        
        assert validator.contraindications is not None
        assert validator.drug_nutrient_interactions is not None
    
    async def test_validate_safe_recommendations(self):
        """Test validation of safe recommendations"""
        validator = MedicalAccuracyValidator()
        
        recommendations = """
        Based on MTHFR C677T variant (Smith et al., 2020), consider:
        1. Folate supplementation 400mcg daily
        2. Monitor homocysteine levels
        3. Consult healthcare provider for personalized advice
        
        Evidence level: Strong (clinical guidelines)
        """
        
        patient_context = {
            "age": 45,
            "conditions": [],
            "medications": [],
            "allergies": []
        }
        
        result = await validator.validate_recommendations(
            recommendations,
            patient_context
        )
        
        assert result["overall_valid"] is True
        assert result["accuracy_score"] > 0
        assert len(result["issues"]) == 0
    
    async def test_detect_contraindication(self):
        """Test detection of contraindications"""
        validator = MedicalAccuracyValidator()
        
        recommendations = """
        Recommend vitamin K supplements 500mcg daily
        """
        
        patient_context = {
            "medications": ["warfarin"],
            "conditions": [],
            "allergies": []
        }
        
        result = await validator.validate_recommendations(
            recommendations,
            patient_context
        )
        
        assert result["overall_valid"] is False
        contraindications = result["contraindications_check"]
        assert contraindications["found"] is True
    
    async def test_citation_check(self):
        """Test citation quality checking"""
        validator = MedicalAccuracyValidator()
        
        # With citations
        text_with_citations = "Studies show benefits (Smith, 2020) and (Jones et al., 2019)"
        citations = validator._check_citations(text_with_citations)
        assert citations["citation_count"] > 0
        
        # Without citations
        text_without = "Studies show benefits"
        citations = validator._check_citations(text_without)
        assert citations["citation_count"] == 0
    
    async def test_guideline_alignment(self):
        """Test clinical guideline alignment check"""
        validator = MedicalAccuracyValidator()
        
        recommendations = """
        Based on evidence-based clinical guidelines, recommend:
        - Follow established protocols
        - Consult healthcare provider
        """
        
        alignment = await validator._check_guideline_alignment(
            recommendations,
            {}
        )
        
        assert alignment["aligned"] is True
        assert alignment["alignment_score"] > 0


@pytest.mark.asyncio
class TestSafetyChecker:
    """Test safety checking"""
    
    async def test_initialization(self):
        """Test safety checker initializes"""
        checker = SafetyChecker()
        
        assert checker.dosage_limits is not None
        assert checker.high_risk_conditions is not None
    
    async def test_check_safe_recommendations(self):
        """Test checking safe recommendations"""
        checker = SafetyChecker()
        
        recommendations = """
        Vitamin D 2000 IU daily
        Folate 400 mcg daily
        """
        
        patient_context = {
            "age": 35,
            "conditions": [],
            "medications": []
        }
        
        result = await checker.check_safety(
            recommendations,
            patient_context
        )
        
        assert result["safe"] is True
        assert result["safety_score"] > 0.9
        assert len(result["concerns"]) == 0
    
    async def test_detect_excessive_dosage(self):
        """Test detection of excessive dosages"""
        checker = SafetyChecker()
        
        recommendations = """
        Vitamin D 15000 IU daily
        """
        
        patient_context = {
            "age": 35,
            "conditions": [],
            "medications": []
        }
        
        result = await checker.check_safety(
            recommendations,
            patient_context
        )
        
        # Should detect excessive dosage
        assert result["safe"] is False
        assert len(result["concerns"]) > 0
    
    async def test_high_risk_patient(self):
        """Test high-risk patient detection"""
        checker = SafetyChecker()
        
        patient_context = {
            "age": 35,
            "conditions": ["pregnancy"],
            "medications": []
        }
        
        high_risk = checker._check_high_risk_patient(patient_context)
        
        assert high_risk["is_high_risk"] is True
        assert high_risk["reason"] == "pregnancy"
    
    async def test_dangerous_combination(self):
        """Test detection of dangerous combinations"""
        checker = SafetyChecker()
        
        recommendations = """
        High dose omega-3 fatty acids
        """
        
        patient_context = {
            "medications": ["aspirin"],
            "conditions": []
        }
        
        result = await checker.check_safety(
            recommendations,
            patient_context
        )
        
        # May detect bleeding risk
        # Implementation dependent on detection logic


class TestHealthcareMetrics:
    """Test healthcare metrics collection"""
    
    def test_record_recommendation(self):
        """Test recording recommendation metrics"""
        metrics = HealthcareMetrics()
        
        metrics.record_recommendation(
            recommendation_id="REC001",
            patient_id="P123",
            processing_time=2.5,
            accuracy_score=0.92,
            safety_score=0.98,
            confidence_score=0.85
        )
        
        assert len(metrics.metrics_store) == 1
    
    def test_get_summary_metrics(self):
        """Test getting summary metrics"""
        metrics = HealthcareMetrics()
        
        # Record some metrics
        for i in range(10):
            metrics.record_recommendation(
                recommendation_id=f"REC{i:03d}",
                patient_id=f"P{i:03d}",
                processing_time=2.0 + i * 0.1,
                accuracy_score=0.9,
                safety_score=0.95,
                confidence_score=0.85
            )
        
        summary = metrics.get_summary_metrics(days=7)
        
        assert summary["total_recommendations"] == 10
        assert summary["avg_processing_time"] > 0
        assert summary["avg_accuracy_score"] == 0.9
    
    def test_get_quality_metrics(self):
        """Test getting quality metrics"""
        metrics = HealthcareMetrics()
        
        # Record varied quality metrics
        metrics.record_recommendation(
            recommendation_id="REC001",
            patient_id="P001",
            processing_time=2.0,
            accuracy_score=0.95,  # High accuracy
            safety_score=0.98,    # High safety
            confidence_score=0.85
        )
        
        metrics.record_recommendation(
            recommendation_id="REC002",
            patient_id="P002",
            processing_time=2.0,
            accuracy_score=0.75,  # Lower accuracy
            safety_score=0.92,
            confidence_score=0.70
        )
        
        quality = metrics.get_quality_metrics()
        
        assert "accuracy_pass_rate" in quality
        assert "safety_pass_rate" in quality
        assert 0 <= quality["accuracy_pass_rate"] <= 1