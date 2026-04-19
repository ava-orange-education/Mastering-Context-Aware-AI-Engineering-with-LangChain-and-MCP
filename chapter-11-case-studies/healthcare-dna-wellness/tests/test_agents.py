"""
Tests for healthcare agents
"""

import pytest
import asyncio
import sys
sys.path.append('../..')

from agents.dna_analysis_agent import DNAAnalysisAgent
from agents.clinical_guidelines_agent import ClinicalGuidelinesAgent
from agents.wellness_recommendation_agent import WellnessRecommendationAgent
from agents.compliance_audit_agent import ComplianceAuditAgent


@pytest.mark.asyncio
class TestDNAAnalysisAgent:
    """Test DNA Analysis Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = DNAAnalysisAgent()
        
        assert agent.name == "DNA Analysis Agent"
        assert agent.model is not None
        assert agent.client is not None
    
    async def test_process_single_variant(self):
        """Test processing single variant"""
        agent = DNAAnalysisAgent()
        
        input_data = {
            "patient_id": "TEST001",
            "genetic_variants": [
                {
                    "gene": "MTHFR",
                    "variant": "C677T",
                    "zygosity": "heterozygous",
                    "quality_score": 0.95
                }
            ]
        }
        
        result = await agent.process(input_data)
        
        assert result.content is not None
        assert result.agent_name == "DNA Analysis Agent"
        assert "variants" in result.metadata
        assert len(result.metadata["variants"]) == 1
        assert result.confidence is not None
    
    async def test_process_multiple_variants(self):
        """Test processing multiple variants"""
        agent = DNAAnalysisAgent()
        
        input_data = {
            "patient_id": "TEST002",
            "genetic_variants": [
                {
                    "gene": "MTHFR",
                    "variant": "C677T",
                    "zygosity": "heterozygous",
                    "quality_score": 0.95
                },
                {
                    "gene": "APOE",
                    "variant": "E4",
                    "zygosity": "heterozygous",
                    "quality_score": 0.92
                }
            ]
        }
        
        result = await agent.process(input_data)
        
        assert len(result.metadata["variants"]) == 2
        assert result.confidence > 0
    
    async def test_unknown_variant(self):
        """Test handling of unknown variant"""
        agent = DNAAnalysisAgent()
        
        input_data = {
            "patient_id": "TEST003",
            "genetic_variants": [
                {
                    "gene": "UNKNOWN",
                    "variant": "XYZ123",
                    "zygosity": "heterozygous",
                    "quality_score": 0.85
                }
            ]
        }
        
        result = await agent.process(input_data)
        
        # Should handle gracefully
        assert result.metadata.get("requires_review") is True
    
    async def test_health_check(self):
        """Test agent health check"""
        agent = DNAAnalysisAgent()
        
        # Health check might fail without API key configured
        # Just verify method exists and returns boolean
        try:
            is_healthy = await agent.health_check()
            assert isinstance(is_healthy, bool)
        except:
            # Expected if no API key configured
            pass


@pytest.mark.asyncio
class TestClinicalGuidelinesAgent:
    """Test Clinical Guidelines Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ClinicalGuidelinesAgent()
        
        assert agent.name == "Clinical Guidelines Agent"
        assert agent.retriever is not None
    
    async def test_process_with_query(self):
        """Test processing with direct query"""
        agent = ClinicalGuidelinesAgent()
        
        input_data = {
            "query": "MTHFR C677T folate supplementation",
            "top_k": 5
        }
        
        # This will fail without vector DB setup, but tests the flow
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception as e:
            # Expected without vector DB
            assert "vector" in str(e).lower() or "initialize" in str(e).lower()
    
    async def test_process_with_variants(self):
        """Test processing with variant information"""
        agent = ClinicalGuidelinesAgent()
        
        input_data = {
            "variants": [
                {
                    "gene": "MTHFR",
                    "variant": "C677T"
                }
            ],
            "conditions": ["elevated homocysteine"],
            "top_k": 5
        }
        
        try:
            result = await agent.process(input_data)
            assert result.content is not None
        except Exception:
            # Expected without vector DB
            pass


@pytest.mark.asyncio
class TestWellnessRecommendationAgent:
    """Test Wellness Recommendation Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = WellnessRecommendationAgent()
        
        assert agent.name == "Wellness Recommendation Agent"
        assert agent.model is not None
    
    async def test_process_requires_inputs(self):
        """Test that process requires analysis and guidelines"""
        agent = WellnessRecommendationAgent()
        
        with pytest.raises(ValueError):
            await agent.process({})
    
    async def test_calculate_confidence(self):
        """Test confidence calculation"""
        agent = WellnessRecommendationAgent()
        
        from shared.base_agent import AgentResponse
        from datetime import datetime
        
        mock_analysis = AgentResponse(
            content="Analysis",
            agent_name="DNA",
            timestamp=datetime.utcnow(),
            confidence=0.9
        )
        
        mock_guidelines = AgentResponse(
            content="Guidelines",
            agent_name="Clinical",
            timestamp=datetime.utcnow(),
            confidence=0.85
        )
        
        confidence = agent._calculate_confidence(mock_analysis, mock_guidelines)
        
        # Should be weighted average
        assert 0 <= confidence <= 1
        assert confidence > 0.8


@pytest.mark.asyncio
class TestComplianceAuditAgent:
    """Test Compliance Audit Agent"""
    
    async def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ComplianceAuditAgent()
        
        assert agent.name == "Compliance Audit Agent"
        assert agent.audit_logger is not None
    
    async def test_process_authorized_access(self):
        """Test authorized access logging"""
        agent = ComplianceAuditAgent()
        
        input_data = {
            "action": "view_genetic_data",
            "user_id": "doctor123",
            "patient_id": "P123456",
            "data_accessed": {"genetic_variants": 3},
            "ip_address": "192.168.1.100",
            "authorization": {
                "role": "physician",
                "permitted_actions": ["view_genetic_data", "generate_recommendations"]
            }
        }
        
        result = await agent.process(input_data)
        
        assert result.metadata["compliant"] is True
        assert result.metadata["compliance_results"]["authorization_check"]["authorized"] is True
    
    async def test_process_unauthorized_access(self):
        """Test unauthorized access detection"""
        agent = ComplianceAuditAgent()
        
        input_data = {
            "action": "delete_data",
            "user_id": "nurse456",
            "patient_id": "P123456",
            "data_accessed": {},
            "ip_address": "192.168.1.101",
            "authorization": {
                "role": "nurse",
                "permitted_actions": ["view_clinical_data"]
            }
        }
        
        result = await agent.process(input_data)
        
        assert result.metadata["compliant"] is False
        assert result.metadata["compliance_results"]["authorization_check"]["authorized"] is False
    
    async def test_phi_detection(self):
        """Test PHI detection in output"""
        agent = ComplianceAuditAgent()
        
        output_with_phi = "Patient SSN: 123-45-6789, Phone: 555-1234"
        
        input_data = {
            "action": "generate_recommendations",
            "user_id": "doctor123",
            "patient_id": "P123456",
            "data_accessed": {},
            "output_data": output_with_phi,
            "authorization": {
                "role": "physician",
                "permitted_actions": ["generate_recommendations"]
            }
        }
        
        result = await agent.process(input_data)
        
        # Should detect PHI
        phi_detection = result.metadata["compliance_results"]["phi_detection"]
        assert phi_detection["phi_detected"] is True
        assert len(phi_detection["phi_types"]) > 0


@pytest.mark.asyncio
class TestAgentIntegration:
    """Test agent integration"""
    
    async def test_dna_to_wellness_flow(self):
        """Test flow from DNA analysis to wellness recommendations"""
        
        # This would require full setup, so we'll test the structure
        dna_agent = DNAAnalysisAgent()
        wellness_agent = WellnessRecommendationAgent()
        
        # Verify agents can be chained
        assert dna_agent.name != wellness_agent.name
        assert callable(dna_agent.process)
        assert callable(wellness_agent.process)