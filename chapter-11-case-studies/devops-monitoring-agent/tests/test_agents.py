"""
Tests for DevOps Agents
"""

import pytest
import asyncio
from datetime import datetime
import sys
sys.path.append('..')

from agents.incident_detection_agent import IncidentDetectionAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.root_cause_analysis_agent import RootCauseAnalysisAgent
from agents.remediation_agent import RemediationAgent
from agents.monitoring_agent import MonitoringAgent
from agents.learning_agent import LearningAgent


class TestIncidentDetectionAgent:
    """Test incident detection agent"""
    
    @pytest.fixture
    def agent(self):
        return IncidentDetectionAgent()
    
    @pytest.mark.asyncio
    async def test_detect_high_severity_incident(self, agent):
        """Test detection of high severity incident"""
        
        input_data = {
            "metrics": {
                "cpu_usage": 0.95,
                "error_rate": 0.15,
                "response_time": 3000
            },
            "alerts": [
                "High CPU usage",
                "Error rate above threshold"
            ],
            "logs": [
                "ERROR: Database connection failed",
                "ERROR: Service timeout"
            ]
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata is not None
        assert response.metadata.get("severity") in ["high", "critical"]
        assert len(response.metadata.get("affected_components", [])) > 0
    
    @pytest.mark.asyncio
    async def test_no_incident_detected(self, agent):
        """Test when no incident is detected"""
        
        input_data = {
            "metrics": {
                "cpu_usage": 0.45,
                "error_rate": 0.001,
                "response_time": 150
            },
            "alerts": [],
            "logs": []
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        # Should indicate no incident or low severity
        severity = response.metadata.get("severity", "none")
        assert severity in ["none", "low"]


class TestAnomalyDetectionAgent:
    """Test anomaly detection agent"""
    
    @pytest.fixture
    def agent(self):
        return AnomalyDetectionAgent()
    
    @pytest.mark.asyncio
    async def test_detect_metric_anomaly(self, agent):
        """Test detection of metric anomaly"""
        
        input_data = {
            "metrics": {
                "cpu_usage": 0.95,  # Anomalously high
                "memory_usage": 0.60
            },
            "baseline": {
                "cpu_usage": {"mean": 0.50, "stdev": 0.10},
                "memory_usage": {"mean": 0.55, "stdev": 0.05}
            }
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata.get("anomalies_detected") is not None
        
        # If anomalies detected, verify structure
        if response.metadata.get("anomalies_detected"):
            anomalies = response.metadata.get("anomalies", [])
            assert len(anomalies) > 0
            assert "metric" in anomalies[0]
            assert "severity" in anomalies[0]


class TestRootCauseAnalysisAgent:
    """Test RCA agent"""
    
    @pytest.fixture
    def agent(self):
        return RootCauseAnalysisAgent()
    
    @pytest.mark.asyncio
    async def test_perform_rca(self, agent):
        """Test RCA analysis"""
        
        input_data = {
            "incident": {
                "title": "High Error Rate",
                "severity": "high",
                "affected_components": ["api-server"]
            },
            "metrics": {
                "error_rate": 0.15,
                "cpu_usage": 0.85
            },
            "logs": [
                "ERROR: Database timeout",
                "ERROR: Connection pool exhausted"
            ],
            "recent_changes": [
                {
                    "type": "deployment",
                    "description": "Deployed v2.0.0",
                    "timestamp": "2024-01-15T14:00:00Z"
                }
            ]
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata.get("root_cause") is not None
        assert isinstance(response.metadata.get("contributing_factors", []), list)
        assert response.confidence > 0


class TestRemediationAgent:
    """Test remediation agent"""
    
    @pytest.fixture
    def agent(self):
        return RemediationAgent()
    
    @pytest.mark.asyncio
    async def test_generate_remediation_plan(self, agent):
        """Test remediation plan generation"""
        
        input_data = {
            "incident": {
                "title": "High Memory Usage",
                "severity": "high",
                "type": "resource",
                "affected_components": ["api-server-1"]
            },
            "root_cause": "Memory leak in request handler",
            "system_state": {
                "replicas": 3,
                "memory_usage": 0.95
            },
            "allow_automatic": True
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata.get("actions") is not None
        
        actions = response.metadata.get("actions", [])
        assert len(actions) > 0
        
        # Verify action structure
        for action in actions:
            assert "action" in action
            assert "safe_to_automate" in action
            assert "requires_approval" in action


class TestMonitoringAgent:
    """Test monitoring agent"""
    
    @pytest.fixture
    def agent(self):
        return MonitoringAgent()
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test system health check"""
        
        input_data = {
            "services": ["api-server", "database", "cache"],
            "metrics": {
                "api_server_error_rate": 0.01,
                "api_server_p99_latency": 200,
                "database_availability": 0.999
            },
            "slos": {
                "api_server_max_error_rate": 0.05,
                "api_server_max_latency": 1000
            }
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata.get("overall_health") is not None
        assert response.metadata.get("service_health") is not None
        assert response.metadata.get("slo_compliance") is not None


class TestLearningAgent:
    """Test learning agent"""
    
    @pytest.fixture
    def agent(self):
        return LearningAgent()
    
    @pytest.mark.asyncio
    async def test_learn_from_incident(self, agent):
        """Test learning from resolved incident"""
        
        input_data = {
            "incident": {
                "title": "Database Connection Pool Exhausted",
                "type": "resource",
                "severity": "high",
                "affected_components": ["api-server"]
            },
            "resolution": {
                "root_cause": "Insufficient connection pool size",
                "method": "Increased pool size",
                "time_to_resolve": 15
            },
            "actions_taken": [
                {"action": "scale_deployment", "success": True},
                {"action": "update_config", "success": True}
            ],
            "outcome": "resolved"
        }
        
        response = await agent.process(input_data)
        
        assert response is not None
        assert response.metadata.get("patterns_recorded") >= 0
        assert response.metadata.get("learning_applied") is True
    
    def test_action_effectiveness_tracking(self, agent):
        """Test tracking of action effectiveness"""
        
        # Record some actions
        agent._record_action_effectiveness(
            [{"action": "restart_pod"}],
            "resolved"
        )
        
        agent._record_action_effectiveness(
            [{"action": "restart_pod"}],
            "resolved"
        )
        
        agent._record_action_effectiveness(
            [{"action": "scale_deployment"}],
            "failed"
        )
        
        stats = agent.get_action_effectiveness_stats()
        
        assert "restart_pod" in stats
        assert stats["restart_pod"]["success_rate"] > 0
        assert stats["restart_pod"]["total_uses"] == 2


@pytest.mark.asyncio
async def test_agent_health_checks():
    """Test health checks for all agents"""
    
    agents = [
        IncidentDetectionAgent(),
        AnomalyDetectionAgent(),
        RootCauseAnalysisAgent(),
        RemediationAgent(),
        MonitoringAgent(),
        LearningAgent()
    ]
    
    for agent in agents:
        health = await agent.health_check()
        assert health is True, f"{agent.name} health check failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])