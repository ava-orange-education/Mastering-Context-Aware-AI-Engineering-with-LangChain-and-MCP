"""
EHR Connector

Integrates with Electronic Health Records using HL7 FHIR standard
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import httpx
import sys
sys.path.append('../..')

from shared.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EHRConnector:
    """
    Connector for HL7 FHIR-compliant EHR systems
    """
    
    def __init__(self):
        self.base_url = settings.healthcare_ehr_endpoint
        self.api_key = settings.healthcare_ehr_api_key
        self.client = None
        
        logger.info(f"Initialized EHR connector: {self.base_url}")
    
    async def initialize(self) -> None:
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/fhir+json"
            },
            timeout=30.0
        )
        
        logger.info("EHR connector initialized")
    
    async def close(self) -> None:
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
    
    async def get_patient_record(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient record using FHIR Patient resource
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            FHIR Patient resource
        """
        try:
            response = await self.client.get(f"/Patient/{patient_id}")
            response.raise_for_status()
            
            patient = response.json()
            
            logger.info(f"Retrieved patient record: {patient_id}")
            return patient
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get patient record: {e}")
            raise
    
    async def get_lab_results(
        self,
        patient_id: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Get lab results using FHIR Observation resources
        
        Args:
            patient_id: Patient identifier
            days: Number of days to look back
        
        Returns:
            List of FHIR Observation resources
        """
        try:
            # FHIR search parameters
            params = {
                "patient": patient_id,
                "category": "laboratory",
                "_sort": "-date",
                "_count": 100
            }
            
            response = await self.client.get("/Observation", params=params)
            response.raise_for_status()
            
            bundle = response.json()
            observations = bundle.get("entry", [])
            
            logger.info(f"Retrieved {len(observations)} lab results for {patient_id}")
            return [entry["resource"] for entry in observations]
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get lab results: {e}")
            raise
    
    async def get_genetic_observations(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get genetic observations/test results
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            List of genetic observations
        """
        try:
            params = {
                "patient": patient_id,
                "code": "http://loinc.org|81247-9",  # LOINC code for genetic variant
                "_count": 100
            }
            
            response = await self.client.get("/Observation", params=params)
            response.raise_for_status()
            
            bundle = response.json()
            observations = bundle.get("entry", [])
            
            logger.info(f"Retrieved {len(observations)} genetic observations")
            return [entry["resource"] for entry in observations]
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get genetic observations: {e}")
            raise
    
    async def get_conditions(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get patient conditions using FHIR Condition resources
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            List of FHIR Condition resources
        """
        try:
            params = {
                "patient": patient_id,
                "clinical-status": "active",
                "_count": 100
            }
            
            response = await self.client.get("/Condition", params=params)
            response.raise_for_status()
            
            bundle = response.json()
            conditions = bundle.get("entry", [])
            
            logger.info(f"Retrieved {len(conditions)} conditions")
            return [entry["resource"] for entry in conditions]
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get conditions: {e}")
            raise
    
    async def get_medications(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get patient medications using FHIR MedicationStatement resources
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            List of FHIR MedicationStatement resources
        """
        try:
            params = {
                "patient": patient_id,
                "status": "active",
                "_count": 100
            }
            
            response = await self.client.get("/MedicationStatement", params=params)
            response.raise_for_status()
            
            bundle = response.json()
            medications = bundle.get("entry", [])
            
            logger.info(f"Retrieved {len(medications)} medications")
            return [entry["resource"] for entry in medications]
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get medications: {e}")
            raise
    
    async def get_allergies(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get patient allergies using FHIR AllergyIntolerance resources
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            List of FHIR AllergyIntolerance resources
        """
        try:
            params = {
                "patient": patient_id,
                "clinical-status": "active",
                "_count": 100
            }
            
            response = await self.client.get("/AllergyIntolerance", params=params)
            response.raise_for_status()
            
            bundle = response.json()
            allergies = bundle.get("entry", [])
            
            logger.info(f"Retrieved {len(allergies)} allergies")
            return [entry["resource"] for entry in allergies]
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to get allergies: {e}")
            raise
    
    async def create_observation(
        self,
        patient_id: str,
        observation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create new observation (e.g., genetic analysis result)
        
        Args:
            patient_id: Patient identifier
            observation_data: FHIR Observation resource
        
        Returns:
            Created observation
        """
        try:
            # Ensure patient reference
            observation_data["subject"] = {
                "reference": f"Patient/{patient_id}"
            }
            
            response = await self.client.post(
                "/Observation",
                json=observation_data
            )
            response.raise_for_status()
            
            observation = response.json()
            
            logger.info(f"Created observation: {observation.get('id')}")
            return observation
        
        except httpx.HTTPError as e:
            logger.error(f"Failed to create observation: {e}")
            raise
    
    def parse_patient_demographics(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse FHIR Patient resource into simplified format
        
        Args:
            patient: FHIR Patient resource
        
        Returns:
            Simplified patient demographics
        """
        name = patient.get("name", [{}])[0]
        
        return {
            "patient_id": patient.get("id"),
            "first_name": name.get("given", [""])[0],
            "last_name": name.get("family", ""),
            "birth_date": patient.get("birthDate"),
            "gender": patient.get("gender"),
            "active": patient.get("active", True)
        }
    
    def parse_observation_value(self, observation: Dict[str, Any]) -> Any:
        """
        Parse observation value from FHIR format
        
        Args:
            observation: FHIR Observation resource
        
        Returns:
            Observation value
        """
        # FHIR supports multiple value types
        if "valueQuantity" in observation:
            return {
                "value": observation["valueQuantity"].get("value"),
                "unit": observation["valueQuantity"].get("unit"),
                "type": "quantity"
            }
        elif "valueString" in observation:
            return {
                "value": observation["valueString"],
                "type": "string"
            }
        elif "valueCodeableConcept" in observation:
            coding = observation["valueCodeableConcept"].get("coding", [{}])[0]
            return {
                "value": coding.get("display"),
                "code": coding.get("code"),
                "type": "coded"
            }
        else:
            return None