"""
Monitoring Agent

Continuously monitors system health and metrics
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class MonitoringAgent(BaseAgent):
    """
    Agent for continuous system monitoring
    """
    
    def __init__(self):
        super().__init__(
            name="Monitoring Agent",
            model="claude-sonnet-4-20250514",
            temperature=0.3
        )
        
        # Health check intervals
        self.check_intervals = {
            "critical": 30,    # seconds
            "high": 60,
            "medium": 300,
            "low": 600
        }
    
    def _get_system_prompt(self) -> str:
        """System prompt for monitoring"""
        return """You are an expert DevOps monitoring system.

Your role:
1. Continuously monitor system health
2. Detect anomalies and trends
3. Assess system performance
4. Identify potential issues before they become incidents
5. Track SLIs and SLOs
6. Generate health reports

Monitoring principles:
- Proactive: Catch issues before they impact users
- Comprehensive: Monitor all critical components
- Contextual: Consider business context and SLOs
- Actionable: Provide clear next steps
- Trend-aware: Detect degrading patterns

Health assessment criteria:
- Service availability
- Response times and latency
- Error rates
- Resource utilization
- Dependency health
- SLO compliance

Output format:
- Overall health status
- Component-level health
- Anomalies detected
- Trends and patterns
- SLO status
- Recommended actions"""
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Perform health check
        
        Args:
            input_data: {
                "services": List[str],
                "metrics": Dict[str, Any],
                "slos": Optional[Dict[str, float]],
                "baseline": Optional[Dict[str, Any]]
            }
        
        Returns:
            AgentResponse with health assessment
        """
        
        services = input_data.get("services", [])
        metrics = input_data.get("metrics", {})
        slos = input_data.get("slos", {})
        baseline = input_data.get("baseline", {})
        
        logger.info(f"Monitoring {len(services)} services")
        
        # Analyze health
        health_assessment = await self._assess_health(
            services=services,
            metrics=metrics,
            slos=slos,
            baseline=baseline
        )
        
        return AgentResponse(
            content=health_assessment["summary"],
            agent_name=self.name,
            timestamp=datetime.utcnow(),
            metadata={
                "overall_health": health_assessment["overall_health"],
                "service_health": health_assessment["service_health"],
                "anomalies": health_assessment["anomalies"],
                "slo_compliance": health_assessment["slo_compliance"],
                "recommendations": health_assessment["recommendations"]
            },
            confidence=0.85
        )
    
    async def _assess_health(
        self,
        services: List[str],
        metrics: Dict[str, Any],
        slos: Dict[str, float],
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess system health"""
        
        service_health = {}
        anomalies = []
        
        # Check each service
        for service in services:
            health = self._check_service_health(
                service=service,
                metrics=metrics,
                slos=slos,
                baseline=baseline
            )
            service_health[service] = health
            
            if health["status"] != "healthy":
                anomalies.append({
                    "service": service,
                    "issue": health["issue"],
                    "severity": health["severity"]
                })
        
        # Calculate overall health
        health_scores = [h["score"] for h in service_health.values()]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 1.0
        
        if avg_health >= 0.9:
            overall_health = "healthy"
        elif avg_health >= 0.7:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"
        
        # Check SLO compliance
        slo_compliance = self._check_slo_compliance(metrics, slos)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            service_health=service_health,
            anomalies=anomalies,
            slo_compliance=slo_compliance
        )
        
        # Generate summary
        summary = self._generate_summary(
            overall_health=overall_health,
            service_health=service_health,
            anomalies=anomalies,
            slo_compliance=slo_compliance
        )
        
        return {
            "overall_health": overall_health,
            "service_health": service_health,
            "anomalies": anomalies,
            "slo_compliance": slo_compliance,
            "recommendations": recommendations,
            "summary": summary
        }
    
    def _check_service_health(
        self,
        service: str,
        metrics: Dict[str, Any],
        slos: Dict[str, float],
        baseline: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check health of a single service"""
        
        health = {
            "status": "healthy",
            "score": 1.0,
            "issue": None,
            "severity": None
        }
        
        # Check error rate
        error_rate_key = f"{service}_error_rate"
        if error_rate_key in metrics:
            error_rate = metrics[error_rate_key]
            slo_error_rate = slos.get(f"{service}_max_error_rate", 0.01)
            
            if error_rate > slo_error_rate:
                health["status"] = "unhealthy"
                health["score"] = 0.5
                health["issue"] = f"High error rate: {error_rate:.2%}"
                health["severity"] = "high"
                return health
        
        # Check latency
        latency_key = f"{service}_p99_latency"
        if latency_key in metrics:
            latency = metrics[latency_key]
            slo_latency = slos.get(f"{service}_max_latency", 1000)
            
            if latency > slo_latency:
                health["status"] = "degraded"
                health["score"] = 0.7
                health["issue"] = f"High latency: {latency}ms"
                health["severity"] = "medium"
                return health
        
        # Check availability
        availability_key = f"{service}_availability"
        if availability_key in metrics:
            availability = metrics[availability_key]
            
            if availability < 0.99:
                health["status"] = "unhealthy"
                health["score"] = 0.4
                health["issue"] = f"Low availability: {availability:.2%}"
                health["severity"] = "critical"
                return health
        
        return health
    
    def _check_slo_compliance(
        self,
        metrics: Dict[str, Any],
        slos: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check SLO compliance"""
        
        compliance = {
            "compliant": True,
            "violations": []
        }
        
        for slo_name, slo_target in slos.items():
            if slo_name in metrics:
                actual = metrics[slo_name]
                
                # Check if metric meets SLO
                # Assumes higher is better for most SLOs
                if "error" in slo_name.lower() or "latency" in slo_name.lower():
                    # Lower is better
                    if actual > slo_target:
                        compliance["compliant"] = False
                        compliance["violations"].append({
                            "slo": slo_name,
                            "target": slo_target,
                            "actual": actual,
                            "gap": actual - slo_target
                        })
                else:
                    # Higher is better
                    if actual < slo_target:
                        compliance["compliant"] = False
                        compliance["violations"].append({
                            "slo": slo_name,
                            "target": slo_target,
                            "actual": actual,
                            "gap": slo_target - actual
                        })
        
        return compliance
    
    def _generate_recommendations(
        self,
        service_health: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        slo_compliance: Dict[str, Any]
    ) -> List[str]:
        """Generate monitoring recommendations"""
        
        recommendations = []
        
        # Recommendations based on unhealthy services
        unhealthy_services = [
            name for name, health in service_health.items()
            if health["status"] != "healthy"
        ]
        
        if unhealthy_services:
            recommendations.append(
                f"Investigate {len(unhealthy_services)} unhealthy service(s): "
                f"{', '.join(unhealthy_services)}"
            )
        
        # Recommendations based on SLO violations
        if not slo_compliance["compliant"]:
            recommendations.append(
                f"Address {len(slo_compliance['violations'])} SLO violation(s)"
            )
        
        # Recommendations based on anomalies
        critical_anomalies = [a for a in anomalies if a["severity"] == "critical"]
        if critical_anomalies:
            recommendations.append(
                f"Immediate action needed for {len(critical_anomalies)} critical issue(s)"
            )
        
        return recommendations
    
    def _generate_summary(
        self,
        overall_health: str,
        service_health: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        slo_compliance: Dict[str, Any]
    ) -> str:
        """Generate health summary"""
        
        summary_parts = [
            f"System Health: {overall_health.upper()}",
            ""
        ]
        
        # Service breakdown
        healthy_count = sum(1 for h in service_health.values() if h["status"] == "healthy")
        total_count = len(service_health)
        
        summary_parts.append(f"Services: {healthy_count}/{total_count} healthy")
        
        # Anomalies
        if anomalies:
            summary_parts.append(f"\nAnomalies Detected: {len(anomalies)}")
            for anomaly in anomalies[:3]:
                summary_parts.append(
                    f"  - {anomaly['service']}: {anomaly['issue']} "
                    f"(severity: {anomaly['severity']})"
                )
        
        # SLO compliance
        if not slo_compliance["compliant"]:
            summary_parts.append(
                f"\nSLO Violations: {len(slo_compliance['violations'])}"
            )
        
        return "\n".join(summary_parts)