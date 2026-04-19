"""
Safety Checker

Ensures recommendations are safe for the patient
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SafetyChecker:
    """
    Checker for safety of medical recommendations
    """
    
    def __init__(self):
        # Safety thresholds
        self.dosage_limits = self._load_dosage_limits()
        
        # High-risk conditions
        self.high_risk_conditions = [
            "pregnancy",
            "kidney_disease",
            "liver_disease",
            "bleeding_disorder",
            "immunocompromised"
        ]
    
    def _load_dosage_limits(self) -> Dict[str, Dict[str, Any]]:
        """Load safe dosage limits for supplements"""
        return {
            "vitamin_d": {
                "daily_limit": 4000,
                "unit": "IU",
                "upper_limit": 10000
            },
            "vitamin_a": {
                "daily_limit": 3000,
                "unit": "mcg",
                "upper_limit": 10000
            },
            "folate": {
                "daily_limit": 1000,
                "unit": "mcg",
                "upper_limit": 5000
            },
            "vitamin_b12": {
                "daily_limit": 1000,
                "unit": "mcg",
                "upper_limit": 5000
            }
        }
    
    async def check_safety(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive safety check
        
        Args:
            recommendations: Generated recommendations
            patient_context: Patient context
        
        Returns:
            Safety check results
        """
        
        safety_results = {
            "safe": True,
            "safety_score": 1.0,
            "concerns": [],
            "warnings": [],
            "requires_physician_review": False
        }
        
        # Check for high-risk patient
        high_risk_check = self._check_high_risk_patient(patient_context)
        if high_risk_check["is_high_risk"]:
            safety_results["requires_physician_review"] = True
            safety_results["warnings"].append(
                f"High-risk patient: {high_risk_check['reason']}"
            )
        
        # Check dosage recommendations
        dosage_check = self._check_dosages(recommendations)
        if dosage_check["has_issues"]:
            safety_results["safe"] = False
            safety_results["concerns"].extend(dosage_check["issues"])
        
        # Check for dangerous combinations
        combination_check = self._check_combinations(
            recommendations,
            patient_context
        )
        if combination_check["has_issues"]:
            safety_results["safe"] = False
            safety_results["concerns"].extend(combination_check["issues"])
        
        # Check for absolute contraindications
        contraindication_check = self._check_absolute_contraindications(
            recommendations,
            patient_context
        )
        if contraindication_check["found"]:
            safety_results["safe"] = False
            safety_results["concerns"].extend(
                contraindication_check["contraindications"]
            )
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(safety_results)
        safety_results["safety_score"] = safety_score
        
        logger.info(
            f"Safety check: safe={safety_results['safe']}, "
            f"score={safety_score:.2f}, "
            f"concerns={len(safety_results['concerns'])}"
        )
        
        return safety_results
    
    def _check_high_risk_patient(
        self,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if patient is high-risk"""
        
        conditions = [c.lower() for c in patient_context.get("conditions", [])]
        
        for high_risk_condition in self.high_risk_conditions:
            if high_risk_condition in conditions:
                return {
                    "is_high_risk": True,
                    "reason": high_risk_condition
                }
        
        # Check age
        age = patient_context.get("age")
        if age and (age < 18 or age > 65):
            return {
                "is_high_risk": True,
                "reason": f"age {age} (pediatric or elderly)"
            }
        
        return {
            "is_high_risk": False,
            "reason": None
        }
    
    def _check_dosages(self, recommendations: str) -> Dict[str, Any]:
        """Check supplement dosages"""
        
        import re
        
        issues = []
        
        for supplement, limits in self.dosage_limits.items():
            # Look for dosage recommendations
            # Pattern: "500 mg vitamin_d" or "vitamin_d 500 mg"
            supplement_name = supplement.replace("_", " ")
            
            pattern = rf'(\d+)\s*({limits["unit"]})?\s*{supplement_name}|{supplement_name}\s*(\d+)\s*({limits["unit"]})?'
            matches = re.findall(pattern, recommendations.lower(), re.IGNORECASE)
            
            for match in matches:
                # Extract dosage number
                dosage = int(match[0]) if match[0] else int(match[2])
                
                if dosage > limits["upper_limit"]:
                    issues.append({
                        "type": "excessive_dosage",
                        "supplement": supplement,
                        "recommended_dosage": dosage,
                        "safe_limit": limits["daily_limit"],
                        "upper_limit": limits["upper_limit"],
                        "severity": "high",
                        "message": f"{supplement} dosage {dosage} {limits['unit']} exceeds safe upper limit"
                    })
                elif dosage > limits["daily_limit"]:
                    issues.append({
                        "type": "high_dosage",
                        "supplement": supplement,
                        "recommended_dosage": dosage,
                        "safe_limit": limits["daily_limit"],
                        "severity": "medium",
                        "message": f"{supplement} dosage {dosage} {limits['unit']} exceeds recommended daily limit"
                    })
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues
        }
    
    def _check_combinations(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for dangerous supplement/medication combinations"""
        
        issues = []
        
        # Known dangerous combinations
        dangerous_combinations = {
            ("warfarin", "vitamin_k"): "Vitamin K interferes with warfarin",
            ("aspirin", "high_dose_omega3"): "Increased bleeding risk",
            ("immunosuppressants", "echinacea"): "May reduce immunosuppressant effectiveness"
        }
        
        medications = [m.lower() for m in patient_context.get("medications", [])]
        
        for (med, supplement), warning in dangerous_combinations.items():
            if med in medications and supplement.replace("_", " ") in recommendations.lower():
                issues.append({
                    "type": "dangerous_combination",
                    "medication": med,
                    "supplement": supplement,
                    "severity": "high",
                    "message": warning
                })
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues
        }
    
    def _check_absolute_contraindications(
        self,
        recommendations: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for absolute contraindications"""
        
        contraindications = []
        
        conditions = [c.lower() for c in patient_context.get("conditions", [])]
        
        # Pregnancy contraindications
        if "pregnancy" in conditions:
            if "vitamin a" in recommendations.lower() and "high dose" in recommendations.lower():
                contraindications.append({
                    "type": "absolute_contraindication",
                    "condition": "pregnancy",
                    "contraindicated_item": "high_dose_vitamin_a",
                    "severity": "critical",
                    "message": "High-dose vitamin A is teratogenic during pregnancy"
                })
        
        # Kidney disease contraindications
        if "kidney disease" in conditions or "renal" in " ".join(conditions):
            if "potassium" in recommendations.lower():
                contraindications.append({
                    "type": "absolute_contraindication",
                    "condition": "kidney_disease",
                    "contraindicated_item": "potassium_supplements",
                    "severity": "critical",
                    "message": "Potassium supplements contraindicated in kidney disease"
                })
        
        return {
            "found": len(contraindications) > 0,
            "contraindications": contraindications
        }
    
    def _calculate_safety_score(self, safety_results: Dict[str, Any]) -> float:
        """Calculate overall safety score"""
        
        score = 1.0
        
        # Deduct for concerns
        for concern in safety_results["concerns"]:
            severity = concern.get("severity", "medium")
            
            if severity == "critical":
                score -= 0.5
            elif severity == "high":
                score -= 0.3
            elif severity == "medium":
                score -= 0.1
        
        # Minimum score is 0
        return max(0.0, score)