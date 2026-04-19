"""
API Models

Pydantic models for DevOps monitoring API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class IncidentSeverity(str, Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentType(str, Enum):
    """Incident types"""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    ERROR_RATE = "error_rate"
    RESOURCE = "resource"
    SECURITY = "security"
    OTHER = "other"


class IncidentRequest(BaseModel):
    """Request to create incident"""
    title: str = Field(..., description="Incident title")
    description: str = Field(..., description="Incident description")
    severity: IncidentSeverity = Field(..., description="Incident severity")
    type: IncidentType = Field(..., description="Incident type")
    affected_components: List[str] = Field(default=[], description="Affected components")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Relevant metrics")


class IncidentResponse(BaseModel):
    """Incident response"""
    id: str
    title: str
    description: str
    severity: str
    type: str
    status: str
    affected_components: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    root_cause: Optional[str] = None
    actions_taken: List[str] = []


class MetricsRequest(BaseModel):
    """Request to submit metrics"""
    metrics: Dict[str, float] = Field(..., description="Metric name to value mapping")
    timestamp: Optional[datetime] = Field(default=None, description="Metric timestamp")
    labels: Optional[Dict[str, str]] = Field(default=None, description="Metric labels")


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    metrics: Dict[str, float] = Field(..., description="Current metrics")
    baseline: Optional[Dict[str, Any]] = Field(default=None, description="Baseline statistics")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class AnomalyDetectionResponse(BaseModel):
    """Anomaly detection response"""
    anomalies_detected: bool
    anomaly_count: int
    anomalies: List[Dict[str, Any]]
    analysis: str
    confidence: float


class RemediationRequest(BaseModel):
    """Request for remediation plan"""
    incident_id: str = Field(..., description="Incident identifier")
    allow_automatic: bool = Field(default=False, description="Allow automatic execution")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Constraints")


class RemediationAction(BaseModel):
    """Remediation action"""
    action: str
    description: str
    safe_to_automate: bool
    requires_approval: bool
    impact: str
    execution_steps: List[str]
    rollback_plan: str


class RemediationResponse(BaseModel):
    """Remediation response"""
    incident_id: str
    actions: List[RemediationAction]
    safe_actions: List[RemediationAction]
    supervised_actions: List[RemediationAction]
    plan: str


class ActionExecutionRequest(BaseModel):
    """Request to execute action"""
    action: str = Field(..., description="Action name")
    platform: str = Field(default="kubernetes", description="Platform")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    dry_run: bool = Field(default=False, description="Dry run mode")


class ActionExecutionResponse(BaseModel):
    """Action execution response"""
    execution_id: str
    action: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, bool]
    version: str = "1.0.0"


class AlertRequest(BaseModel):
    """Request to create alert"""
    title: str = Field(..., description="Alert title")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    source: str = Field(default="devops-agent", description="Alert source")
    custom_details: Optional[Dict[str, Any]] = Field(default=None, description="Custom details")


class AlertResponse(BaseModel):
    """Alert response"""
    alert_id: str
    status: str
    message: str


class RCARequest(BaseModel):
    """Request for root cause analysis"""
    incident_id: str = Field(..., description="Incident identifier")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Metrics data")
    logs: Optional[List[str]] = Field(default=None, description="Log entries")
    recent_changes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Recent changes")


class RCAResponse(BaseModel):
    """Root cause analysis response"""
    incident_id: str
    root_cause: str
    contributing_factors: List[str]
    evidence: List[str]
    timeline: List[str]
    recommendations: List[str]
    confidence: float
    analysis: str


class MonitoringStatus(BaseModel):
    """Monitoring status"""
    overall_health: str
    service_health: Dict[str, Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    slo_compliance: Dict[str, Any]
    recommendations: List[str]


class MetricsResponse(BaseModel):
    """Metrics response"""
    metrics: Dict[str, float]
    timestamp: datetime
    statistics: Optional[Dict[str, Any]] = None