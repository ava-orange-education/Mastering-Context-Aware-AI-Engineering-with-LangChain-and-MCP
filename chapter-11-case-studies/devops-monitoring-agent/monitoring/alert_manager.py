"""
Alert Manager

Manages alerts and notifications
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alerts and notifications
    """
    
    def __init__(self):
        # Active alerts
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        
        # Alert rules
        self.rules: Dict[str, Dict[str, Any]] = {}
        
        # Notification channels
        self.channels: Dict[str, Any] = {}
        
        # Alert suppression
        self.suppressions: Dict[str, datetime] = {}
        self.suppression_window = 300  # 5 minutes
    
    def add_rule(
        self,
        name: str,
        condition: str,
        severity: str,
        notification_channels: List[str],
        cooldown: int = 300
    ) -> None:
        """
        Add an alert rule
        
        Args:
            name: Rule name
            condition: Condition expression
            severity: Alert severity
            notification_channels: Channels to notify
            cooldown: Cooldown period in seconds
        """
        
        self.rules[name] = {
            "condition": condition,
            "severity": severity,
            "channels": notification_channels,
            "cooldown": cooldown,
            "enabled": True
        }
        
        logger.info(f"Added alert rule: {name}")
    
    def register_channel(
        self,
        name: str,
        channel_type: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a notification channel
        
        Args:
            name: Channel name
            channel_type: Type (slack, email, pagerduty, etc.)
            config: Channel configuration
        """
        
        self.channels[name] = {
            "type": channel_type,
            "config": config,
            "enabled": True
        }
        
        logger.info(f"Registered notification channel: {name} ({channel_type})")
    
    def evaluate_rules(
        self,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate alert rules against current metrics
        
        Args:
            metrics: Current metric values
        
        Returns:
            List of triggered alerts
        """
        
        triggered_alerts = []
        
        for rule_name, rule in self.rules.items():
            if not rule["enabled"]:
                continue
            
            # Check if rule is in cooldown
            if self._is_suppressed(rule_name):
                continue
            
            # Evaluate condition
            if self._evaluate_condition(rule["condition"], metrics):
                alert = self._create_alert(
                    rule_name=rule_name,
                    severity=rule["severity"],
                    metrics=metrics
                )
                
                triggered_alerts.append(alert)
                
                # Add to active alerts
                self.active_alerts[alert["id"]] = alert
                
                # Suppress for cooldown period
                self._suppress_alert(rule_name, rule["cooldown"])
        
        return triggered_alerts
    
    def _evaluate_condition(
        self,
        condition: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """Evaluate alert condition"""
        
        # Simple condition evaluation
        # In production, use a proper expression parser
        
        try:
            # Replace metric names with values
            expression = condition
            for metric_name, value in metrics.items():
                expression = expression.replace(metric_name, str(value))
            
            # Evaluate
            result = eval(expression)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def _create_alert(
        self,
        rule_name: str,
        severity: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an alert"""
        
        alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{rule_name}"
        
        alert = {
            "id": alert_id,
            "rule": rule_name,
            "severity": severity,
            "state": "firing",
            "started_at": datetime.utcnow(),
            "metrics": metrics,
            "annotations": {
                "summary": f"Alert: {rule_name}",
                "description": f"Alert triggered by rule: {rule_name}"
            }
        }
        
        # Add to history
        self.alert_history.append(alert.copy())
        
        logger.warning(f"Alert triggered: {rule_name} (severity: {severity})")
        
        return alert
    
    def _is_suppressed(self, rule_name: str) -> bool:
        """Check if alert is suppressed"""
        
        if rule_name not in self.suppressions:
            return False
        
        suppression_end = self.suppressions[rule_name]
        
        if datetime.utcnow() > suppression_end:
            # Suppression expired
            del self.suppressions[rule_name]
            return False
        
        return True
    
    def _suppress_alert(self, rule_name: str, duration: int) -> None:
        """Suppress an alert for duration"""
        
        self.suppressions[rule_name] = datetime.utcnow() + timedelta(seconds=duration)
    
    async def send_notifications(
        self,
        alert: Dict[str, Any]
    ) -> None:
        """
        Send notifications for an alert
        
        Args:
            alert: Alert to notify about
        """
        
        rule_name = alert["rule"]
        
        if rule_name not in self.rules:
            return
        
        channels = self.rules[rule_name]["channels"]
        
        for channel_name in channels:
            if channel_name not in self.channels:
                logger.warning(f"Channel not found: {channel_name}")
                continue
            
            channel = self.channels[channel_name]
            
            if not channel["enabled"]:
                continue
            
            try:
                await self._send_to_channel(channel, alert)
            except Exception as e:
                logger.error(f"Failed to send to {channel_name}: {e}")
    
    async def _send_to_channel(
        self,
        channel: Dict[str, Any],
        alert: Dict[str, Any]
    ) -> None:
        """Send alert to a notification channel"""
        
        channel_type = channel["type"]
        
        if channel_type == "slack":
            await self._send_slack(channel, alert)
        elif channel_type == "email":
            await self._send_email(channel, alert)
        elif channel_type == "pagerduty":
            await self._send_pagerduty(channel, alert)
        else:
            logger.warning(f"Unknown channel type: {channel_type}")
    
    async def _send_slack(
        self,
        channel: Dict[str, Any],
        alert: Dict[str, Any]
    ) -> None:
        """Send to Slack"""
        
        # Placeholder - implement actual Slack webhook
        logger.info(f"Sending to Slack: {alert['rule']}")
    
    async def _send_email(
        self,
        channel: Dict[str, Any],
        alert: Dict[str, Any]
    ) -> None:
        """Send email"""
        
        # Placeholder - implement actual email sending
        logger.info(f"Sending email: {alert['rule']}")
    
    async def _send_pagerduty(
        self,
        channel: Dict[str, Any],
        alert: Dict[str, Any]
    ) -> None:
        """Send to PagerDuty"""
        
        # Placeholder - implement actual PagerDuty integration
        logger.info(f"Sending to PagerDuty: {alert['rule']}")
    
    def resolve_alert(self, alert_id: str) -> None:
        """Resolve an active alert"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert["state"] = "resolved"
            alert["resolved_at"] = datetime.utcnow()
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(
        self,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get active alerts"""
        
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        return alerts
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert["severity"]] += 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "by_severity": dict(severity_counts),
            "total_triggered": len(self.alert_history),
            "suppressed_rules": len(self.suppressions)
        }