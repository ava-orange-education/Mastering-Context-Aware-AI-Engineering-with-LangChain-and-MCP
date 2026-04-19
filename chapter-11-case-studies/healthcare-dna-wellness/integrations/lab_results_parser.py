"""
Lab Results Parser

Parses lab results from various formats into standardized structure
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class LabResultsParser:
    """
    Parser for lab results from different sources
    """
    
    def __init__(self):
        # Reference ranges for common genetic markers
        self.reference_ranges = {
            "homocysteine": {"low": 5.0, "high": 15.0, "unit": "µmol/L"},
            "folate": {"low": 2.7, "high": 17.0, "unit": "ng/mL"},
            "vitamin_b12": {"low": 200, "high": 900, "unit": "pg/mL"},
            "vitamin_d": {"low": 30, "high": 100, "unit": "ng/mL"},
            "crp": {"low": 0, "high": 3.0, "unit": "mg/L"},
        }
    
    def parse_fhir_observation(
        self,
        observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse FHIR Observation resource
        
        Args:
            observation: FHIR Observation resource
        
        Returns:
            Standardized lab result
        """
        # Extract code
        code_obj = observation.get("code", {})
        coding = code_obj.get("coding", [{}])[0]
        
        # Extract value
        value_data = self._extract_value(observation)
        
        # Extract date
        effective_date = observation.get("effectiveDateTime")
        
        # Determine status
        status = observation.get("status", "unknown")
        
        # Get interpretation
        interpretation = self._get_interpretation(observation)
        
        return {
            "test_name": code_obj.get("text") or coding.get("display"),
            "loinc_code": coding.get("code") if coding.get("system") == "http://loinc.org" else None,
            "value": value_data.get("value"),
            "unit": value_data.get("unit"),
            "reference_range": self._get_reference_range(observation),
            "status": status,
            "interpretation": interpretation,
            "date": effective_date,
            "source": "fhir_observation"
        }
    
    def _extract_value(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract value from FHIR observation"""
        
        if "valueQuantity" in observation:
            return {
                "value": observation["valueQuantity"].get("value"),
                "unit": observation["valueQuantity"].get("unit")
            }
        elif "valueString" in observation:
            return {
                "value": observation["valueString"],
                "unit": None
            }
        elif "valueCodeableConcept" in observation:
            coding = observation["valueCodeableConcept"].get("coding", [{}])[0]
            return {
                "value": coding.get("display"),
                "unit": None
            }
        else:
            return {"value": None, "unit": None}
    
    def _get_reference_range(
        self,
        observation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract reference range from observation"""
        
        ref_range = observation.get("referenceRange", [])
        if ref_range:
            range_obj = ref_range[0]
            low = range_obj.get("low", {})
            high = range_obj.get("high", {})
            
            return {
                "low": low.get("value"),
                "high": high.get("value"),
                "unit": low.get("unit") or high.get("unit")
            }
        
        return None
    
    def _get_interpretation(
        self,
        observation: Dict[str, Any]
    ) -> Optional[str]:
        """Get interpretation (normal, high, low, etc.)"""
        
        interpretation = observation.get("interpretation", [])
        if interpretation:
            coding = interpretation[0].get("coding", [{}])[0]
            return coding.get("code")
        
        return None
    
    def parse_hl7_message(self, hl7_message: str) -> List[Dict[str, Any]]:
        """
        Parse HL7 v2 message (ORU^R01)
        
        Args:
            hl7_message: HL7 v2 message string
        
        Returns:
            List of lab results
        """
        try:
            from hl7apy.parser import parse_message
            
            message = parse_message(hl7_message)
            results = []
            
            # Extract OBX segments (observations)
            for obx in message.children:
                if obx.name == "OBX":
                    result = self._parse_obx_segment(obx)
                    if result:
                        results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to parse HL7 message: {e}")
            return []
    
    def _parse_obx_segment(self, obx) -> Optional[Dict[str, Any]]:
        """Parse single OBX (observation) segment"""
        
        try:
            return {
                "test_name": str(obx.obx_3.ce_2),  # Observation identifier text
                "value": str(obx.obx_5),  # Observation value
                "unit": str(obx.obx_6),  # Units
                "reference_range": str(obx.obx_7) if obx.obx_7 else None,
                "status": str(obx.obx_11),  # Observation result status
                "date": str(obx.obx_14) if obx.obx_14 else None,
                "source": "hl7_message"
            }
        except Exception as e:
            logger.error(f"Failed to parse OBX segment: {e}")
            return None
    
    def parse_csv_results(
        self,
        csv_content: str,
        format_type: str = "standard"
    ) -> List[Dict[str, Any]]:
        """
        Parse CSV lab results
        
        Args:
            csv_content: CSV content as string
            format_type: Format type (standard, labcorp, quest, etc.)
        
        Returns:
            List of lab results
        """
        import csv
        from io import StringIO
        
        results = []
        reader = csv.DictReader(StringIO(csv_content))
        
        for row in reader:
            if format_type == "standard":
                result = self._parse_standard_csv_row(row)
            elif format_type == "labcorp":
                result = self._parse_labcorp_csv_row(row)
            elif format_type == "quest":
                result = self._parse_quest_csv_row(row)
            else:
                result = self._parse_standard_csv_row(row)
            
            if result:
                results.append(result)
        
        return results
    
    def _parse_standard_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Parse standard CSV format"""
        
        return {
            "test_name": row.get("Test Name") or row.get("test_name"),
            "value": self._parse_numeric_value(row.get("Value") or row.get("value")),
            "unit": row.get("Unit") or row.get("unit"),
            "reference_range": row.get("Reference Range") or row.get("reference_range"),
            "status": row.get("Status") or row.get("status", "final"),
            "date": row.get("Date") or row.get("date"),
            "source": "csv_standard"
        }
    
    def _parse_labcorp_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Parse LabCorp-specific CSV format"""
        
        # LabCorp may use different column names
        return {
            "test_name": row.get("TestName"),
            "value": self._parse_numeric_value(row.get("Result")),
            "unit": row.get("Units"),
            "reference_range": row.get("RefRange"),
            "status": "final",
            "date": row.get("CollectionDate"),
            "source": "csv_labcorp"
        }
    
    def _parse_quest_csv_row(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Parse Quest Diagnostics-specific CSV format"""
        
        return {
            "test_name": row.get("Test"),
            "value": self._parse_numeric_value(row.get("ResultValue")),
            "unit": row.get("ResultUnit"),
            "reference_range": row.get("ReferenceRange"),
            "status": "final",
            "date": row.get("SpecimenDate"),
            "source": "csv_quest"
        }
    
    def _parse_numeric_value(self, value_str: Optional[str]) -> Optional[float]:
        """Parse numeric value from string"""
        
        if not value_str:
            return None
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[<>≤≥]', '', str(value_str))
        cleaned = cleaned.strip()
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    def interpret_result(
        self,
        test_name: str,
        value: float,
        unit: str
    ) -> Dict[str, Any]:
        """
        Interpret lab result against reference ranges
        
        Args:
            test_name: Name of test
            value: Test value
            unit: Unit of measurement
        
        Returns:
            Interpretation with status
        """
        # Normalize test name
        normalized_name = test_name.lower().replace(" ", "_")
        normalized_name = re.sub(r'[^a-z0-9_]', '', normalized_name)
        
        # Check if we have reference range
        ref_range = self.reference_ranges.get(normalized_name)
        
        if not ref_range:
            return {
                "status": "unknown",
                "message": "No reference range available",
                "severity": "info"
            }
        
        # Check unit compatibility
        if unit != ref_range["unit"]:
            return {
                "status": "unit_mismatch",
                "message": f"Unit mismatch: expected {ref_range['unit']}, got {unit}",
                "severity": "warning"
            }
        
        # Interpret value
        if value < ref_range["low"]:
            severity = "low" if (ref_range["low"] - value) / ref_range["low"] < 0.2 else "very_low"
            return {
                "status": "low",
                "message": f"Below normal range ({ref_range['low']}-{ref_range['high']} {ref_range['unit']})",
                "severity": severity,
                "reference_range": ref_range
            }
        
        elif value > ref_range["high"]:
            severity = "high" if (value - ref_range["high"]) / ref_range["high"] < 0.2 else "very_high"
            return {
                "status": "high",
                "message": f"Above normal range ({ref_range['low']}-{ref_range['high']} {ref_range['unit']})",
                "severity": severity,
                "reference_range": ref_range
            }
        
        else:
            return {
                "status": "normal",
                "message": f"Within normal range ({ref_range['low']}-{ref_range['high']} {ref_range['unit']})",
                "severity": "normal",
                "reference_range": ref_range
            }
    
    def correlate_with_genetics(
        self,
        lab_results: List[Dict[str, Any]],
        genetic_variants: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Correlate lab results with genetic variants
        
        Args:
            lab_results: List of lab results
            genetic_variants: List of genetic variants
        
        Returns:
            Correlations found
        """
        correlations = []
        
        # Define known correlations
        genetic_lab_correlations = {
            "MTHFR:C677T": ["homocysteine", "folate"],
            "APOE:E4": ["cholesterol", "ldl", "apob"],
            "CYP2C19:*2": ["clopidogrel_response"],
        }
        
        for variant in genetic_variants:
            variant_key = f"{variant['gene']}:{variant['variant']}"
            related_tests = genetic_lab_correlations.get(variant_key, [])
            
            for lab in lab_results:
                test_name_normalized = lab["test_name"].lower().replace(" ", "_")
                
                for related_test in related_tests:
                    if related_test in test_name_normalized:
                        correlations.append({
                            "variant": variant_key,
                            "lab_test": lab["test_name"],
                            "lab_value": lab["value"],
                            "lab_status": lab.get("status"),
                            "clinical_significance": self._get_correlation_significance(
                                variant_key, lab
                            )
                        })
        
        return correlations
    
    def _get_correlation_significance(
        self,
        variant_key: str,
        lab_result: Dict[str, Any]
    ) -> str:
        """Get clinical significance of variant-lab correlation"""
        
        # Simplified significance determination
        # In production, use comprehensive clinical database
        
        if variant_key == "MTHFR:C677T" and "homocysteine" in lab_result["test_name"].lower():
            if lab_result.get("value", 0) > 15:
                return "Elevated homocysteine with MTHFR variant - consider folate supplementation"
            else:
                return "Normal homocysteine despite MTHFR variant - adequate folate status"
        
        return "Genetic variant may influence lab values - consult healthcare provider"