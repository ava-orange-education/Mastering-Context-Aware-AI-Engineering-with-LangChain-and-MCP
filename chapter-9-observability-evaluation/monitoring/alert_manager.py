"""
Manage alerts for monitoring thresholds.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    condition: Callable[[Any], bool]
    severity: str  # info, warning, critical
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5  # Minimum time between alerts
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    severity: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manage monitoring alerts"""
    
    def __init__(self):
        """Initialize alert manager"""
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        self.alert_handlers: List[Callable] = []
    
    def add_rule(self, rule_id: str, name: str, condition: Callable,
                severity: str = "warning", description: str = "",
                cooldown_minutes: int = 5) -> AlertRule:
        """
        Add alert rule
        
        Args:
            rule_id: Unique rule ID
            name: Rule name
            condition: Condition function that returns bool
            severity: Alert severity
            description: Rule description
            cooldown_minutes: Cooldown period
            
        Returns:
            Created alert rule
        """
        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            condition=condition,
            severity=severity,
            description=description,
            cooldown_minutes=cooldown_minutes
        )
        
        self.rules[rule_id] = rule
        
        logger.info(f"Alert rule added: {name} ({rule_id})")
        
        return rule
    
    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Alert rule removed: {rule_id}")
    
    def check_rules(self, metrics: Dict[str, Any]):
        """
        Check all rules against current metrics
        
        Args:
            metrics: Current metrics to check
        """
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                from datetime import timedelta
                cooldown = timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() - rule.last_triggered < cooldown:
                    continue
            
            # Evaluate condition
            try:
                if rule.condition(metrics):
                    self._trigger_alert(rule, metrics)
            except Exception as e:
                logger.error(f"Error checking rule {rule.rule_id}: {e}")
    
    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger an alert"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            message=f"{rule.name}: {rule.description}",
            metadata=metrics
        )
        
        self.alerts.append(alert)
        rule.last_triggered = datetime.now()
        
        # Log based on severity
        if rule.severity == "critical":
            logger.critical(f"ALERT: {alert.message}")
        elif rule.severity == "warning":
            logger.warning(f"ALERT: {alert.message}")
        else:
            logger.info(f"ALERT: {alert.message}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        Add alert handler function
        
        Args:
            handler: Function that receives Alert object
        """
        self.alert_handlers.append(handler)
        logger.info("Alert handler added")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """
        Get unacknowledged alerts
        
        Args:
            severity: Filter by severity
            
        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts if not a.acknowledged]
        
        if severity:
            active = [a for a in active if a.severity == severity]
        
        return active
    
    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return
        
        logger.warning(f"Alert not found: {alert_id}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()
        
        by_severity = {}
        for alert in active_alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'by_severity': by_severity,
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled)
        }


def create_default_rules(alert_manager: AlertManager):
    """Create default alert rules"""
    
    # High error rate
    alert_manager.add_rule(
        rule_id="high_error_rate",
        name="High Error Rate",
        condition=lambda m: m.get('error_rate', 0) > 0.1,  # 10% error rate
        severity="critical",
        description="Error rate exceeds 10%"
    )
    
    # High latency
    alert_manager.add_rule(
        rule_id="high_latency",
        name="High Latency",
        condition=lambda m: m.get('avg_latency', 0) > 5.0,  # 5 seconds
        severity="warning",
        description="Average latency exceeds 5 seconds"
    )
    
    # Low throughput
    alert_manager.add_rule(
        rule_id="low_throughput",
        name="Low Throughput",
        condition=lambda m: m.get('throughput', 100) < 10,  # < 10 requests
        severity="warning",
        description="Throughput dropped below threshold"
    )
    
    logger.info("Default alert rules created")