"""
Medical Accuracy Validator

Validates medical accuracy of AI-generated recommendations
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MedicalAccuracyValidator:
    """
    Validator for medical accuracy of recommendations
    """
    
    def __init__(self):
        # Known contraindications database
        self.contraindications = self._load_contraindications()
        
        # Drug-nutrient interactions
        self.drug_nutrient_interactions = self._load_drug_interactions()
    
    def _load_contraindications(self) -> Dict[str, List[str]]:
        """
        Load known contraindications
        In production, integrate with comprehensive medical database
        """
        return {
            "warfarin": [
                "vitamin_k_supplements",
                "high_vitamin_k_foods"
            ],
            "pregnancy": [
                "high_dose_vitamin_a",
                "certain_herbal_supplements"
            ],
            "kidney_disease": [
                "potassium_supplements",
                "high_protein_diet"
            ]
        }
    
    def _load_drug_interactions(self) -> Dict[str, List[str]]:
        """Load drug-nutrient interactions"""
        return {
            "warfarin": ["vitamin_k"],
            "levothyroxine": ["calcium", "iron"],
            "metformin": ["vitamin_b12"]
        }
    
    async def validate_recommendations(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate medical recommendations
        
        Args:
            recommendations: Generated recommendations text
            patient_context: Patient context (conditions, medications, etc.)
        
        Returns:
            Validation result
        """
        
        validation_results = {
            "overall_valid": True,
            "accuracy_score": 0.0,
            "issues": [],
            "warnings": [],
            "contraindications_check": {},
            "citation_check": {},
            "clinical_guidelines_alignment": {}
        }
        
        # Check for contraindications
        contraindication_results = await self._check_contraindications(
            recommendations,
            patient_context
        )
        validation_results["contraindications_check"] = contraindication_results
        
        if contraindication_results["found"]:
            validation_results["overall_valid"] = False
            validation_results["issues"].extend(
                contraindication_results["contraindications"]
            )
        
        # Check citation quality
        citation_results = self._check_citations(recommendations)
        validation_results["citation_check"] = citation_results
        
        if citation_results["citation_count"] == 0:
            validation_results["warnings"].append(
                "No citations found - recommendations should be evidence-based"
            )
        
        # Check guideline alignment
        guideline_alignment = await self._check_guideline_alignment(
            recommendations,
            patient_context
        )
        validation_results["clinical_guidelines_alignment"] = guideline_alignment
        
        # Calculate accuracy score
        accuracy_score = self._calculate_accuracy_score(validation_results)
        validation_results["accuracy_score"] = accuracy_score
        
        # Overall validity
        validation_results["overall_valid"] = (
            accuracy_score >= 0.8 and
            len(validation_results["issues"]) == 0
        )
        
        logger.info(
            f"Validation complete: valid={validation_results['overall_valid']}, "
            f"score={accuracy_score:.2f}"
        )
        
        return validation_results
    
    async def _check_contraindications(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for contraindications"""
        
        medications = patient_context.get("medications", [])
        conditions = patient_context.get("conditions", [])
        
        found_contraindications = []
        
        # Check medication contraindications
        for medication in medications:
            medication_lower = medication.lower()
            
            if medication_lower in self.contraindications:
                contraindicated_items = self.contraindications[medication_lower]
                
                for item in contraindicated_items:
                    if item.replace("_", " ") in recommendations.lower():
                        found_contraindications.append({
                            "type": "medication_contraindication",
                            "medication": medication,
                            "contraindicated_item": item,
                            "severity": "high",
                            "message": f"{item} is contraindicated with {medication}"
                        })
        
        # Check condition contraindications
        for condition in conditions:
            condition_lower = condition.lower()
            
            if condition_lower in self.contraindications:
                contraindicated_items = self.contraindications[condition_lower]
                
                for item in contraindicated_items:
                    if item.replace("_", " ") in recommendations.lower():
                        found_contraindications.append({
                            "type": "condition_contraindication",
                            "condition": condition,
                            "contraindicated_item": item,
                            "severity": "high",
                            "message": f"{item} is contraindicated with {condition}"
                        })
        
        return {
            "found": len(found_contraindications) > 0,
            "contraindications": found_contraindications,
            "count": len(found_contraindications)
        }
    
    def _check_citations(self, recommendations: str) -> Dict[str, Any]:
        """Check citation quality"""
        
        import re
        
        # Look for citation patterns
        # (Author, Year), [1], etc.
        citation_patterns = [
            r'\([A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}\)',  # (Smith et al., 2020)
            r'\([A-Z][a-z]+,?\s+\d{4}\)',  # (Smith, 2020)
            r'\[\d+\]',  # [1]
        ]
        
        citations_found = []
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, recommendations)
            citations_found.extend(matches)
        
        return {
            "citation_count": len(citations_found),
            "citations": citations_found,
            "properly_cited": len(citations_found) > 0
        }
    
    async def _check_guideline_alignment(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check alignment with clinical guidelines
        
        In production, integrate with clinical guideline databases
        """
        
        # Simplified check
        # In production, retrieve relevant guidelines and compare
        
        key_phrases = [
            "evidence-based",
            "clinical guidelines",
            "recommended",
            "consult healthcare provider"
        ]
        
        alignment_indicators = sum(
            1 for phrase in key_phrases
            if phrase in recommendations.lower()
        )
        
        return {
            "aligned": alignment_indicators >= 2,
            "alignment_score": alignment_indicators / len(key_phrases),
            "key_phrases_found": alignment_indicators
        }
    
    def _calculate_accuracy_score(
        self,
        validation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall accuracy score"""
        
        score_components = []
        
        # No contraindications: 0.4 weight
        if not validation_results["contraindications_check"]["found"]:
            score_components.append(0.4)
        
        # Has citations: 0.3 weight
        if validation_results["citation_check"]["properly_cited"]:
            score_components.append(0.3)
        
        # Guideline alignment: 0.3 weight
        alignment_score = validation_results["clinical_guidelines_alignment"]["alignment_score"]
        score_components.append(alignment_score * 0.3)
        
        return sum(score_components)
    
    async def validate_genetic_interpretation(
        self,
        variant: Dict[str, Any],
        interpretation: str
    ) -> Dict[str, Any]:
        """
        Validate genetic variant interpretation
        
        Args:
            variant: Genetic variant data
            interpretation: AI-generated interpretation
        
        Returns:
            Validation result
        """
        
        # Check interpretation quality
        quality_indicators = [
            "clinical significance",
            "zygosity",
            "population frequency",
            "evidence"
        ]
        
        indicators_present = sum(
            1 for indicator in quality_indicators
            if indicator in interpretation.lower()
        )
        
        # Check for uncertainty acknowledgment
        uncertainty_phrases = [
            "may",
            "might",
            "possible",
            "preliminary",
            "consult genetic counselor"
        ]
        
        acknowledges_uncertainty = any(
            phrase in interpretation.lower()
            for phrase in uncertainty_phrases
        )
        
        return {
            "valid": indicators_present >= 3,
            "quality_score": indicators_present / len(quality_indicators),
            "acknowledges_uncertainty": acknowledges_uncertainty,
            "requires_review": not acknowledges_uncertainty
        }