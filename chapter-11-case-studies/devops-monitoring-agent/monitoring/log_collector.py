"""
Log Collector

Collects and processes logs from various sources
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class LogCollector:
    """
    Collects logs from multiple sources
    """
    
    def __init__(self):
        # Log sources
        self.sources = {}
        
        # Log buffer
        self.log_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 10000
        
        # Log patterns for error detection
        self.error_patterns = [
            r"error",
            r"exception",
            r"failed",
            r"fatal",
            r"critical",
            r"panic",
            r"traceback"
        ]
        
        # Compiled patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.error_patterns
        ]
    
    def register_source(
        self,
        name: str,
        source_type: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a log source
        
        Args:
            name: Source name
            source_type: Type (kubernetes, elasticsearch, cloudwatch, etc.)
            config: Source configuration
        """
        
        self.sources[name] = {
            "type": source_type,
            "config": config,
            "enabled": True,
            "last_collected": None
        }
        
        logger.info(f"Registered log source: {name} ({source_type})")
    
    async def collect_logs(
        self,
        sources: Optional[List[str]] = None,
        timeframe: str = "5m",
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect logs from sources
        
        Args:
            sources: List of source names (None = all)
            timeframe: Time window for logs
            filters: Optional filters (service, severity, etc.)
        
        Returns:
            List of log entries
        """
        
        # Determine which sources to collect from
        if sources:
            source_names = sources
        else:
            source_names = [name for name, src in self.sources.items() if src["enabled"]]
        
        all_logs = []
        
        # Collect from each source
        for source_name in source_names:
            if source_name not in self.sources:
                logger.warning(f"Unknown log source: {source_name}")
                continue
            
            source = self.sources[source_name]
            
            try:
                logs = await self._collect_from_source(
                    source_name=source_name,
                    source=source,
                    timeframe=timeframe,
                    filters=filters
                )
                
                all_logs.extend(logs)
                source["last_collected"] = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Failed to collect from {source_name}: {e}")
        
        # Add to buffer
        self._add_to_buffer(all_logs)
        
        return all_logs
    
    async def _collect_from_source(
        self,
        source_name: str,
        source: Dict[str, Any],
        timeframe: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect logs from a single source"""
        
        source_type = source["type"]
        
        if source_type == "kubernetes":
            return await self._collect_kubernetes_logs(source, timeframe, filters)
        elif source_type == "elasticsearch":
            return await self._collect_elasticsearch_logs(source, timeframe, filters)
        elif source_type == "cloudwatch":
            return await self._collect_cloudwatch_logs(source, timeframe, filters)
        else:
            logger.warning(f"Unsupported log source type: {source_type}")
            return []
    
    async def _collect_kubernetes_logs(
        self,
        source: Dict[str, Any],
        timeframe: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect logs from Kubernetes"""
        
        # Placeholder - would use actual Kubernetes client
        logs = []
        
        # Simulated log entries
        if filters and "pod" in filters:
            pod_name = filters["pod"]
            logs.append({
                "timestamp": datetime.utcnow(),
                "source": "kubernetes",
                "pod": pod_name,
                "level": "error",
                "message": f"Sample error log from {pod_name}",
                "raw": f"ERROR: Sample error log from {pod_name}"
            })
        
        return logs
    
    async def _collect_elasticsearch_logs(
        self,
        source: Dict[str, Any],
        timeframe: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect logs from Elasticsearch"""
        
        # Placeholder - would use actual Elasticsearch client
        logs = []
        
        # In production, query Elasticsearch with:
        # - Time range based on timeframe
        # - Filters for service, severity, etc.
        # - Parse and normalize log format
        
        return logs
    
    async def _collect_cloudwatch_logs(
        self,
        source: Dict[str, Any],
        timeframe: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Collect logs from CloudWatch"""
        
        # Placeholder - would use boto3
        logs = []
        
        return logs
    
    def _add_to_buffer(self, logs: List[Dict[str, Any]]) -> None:
        """Add logs to buffer"""
        
        self.log_buffer.extend(logs)
        
        # Trim buffer if too large
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]
    
    def filter_error_logs(
        self,
        logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter logs for errors"""
        
        error_logs = []
        
        for log in logs:
            # Check severity level
            if log.get("level") in ["error", "critical", "fatal"]:
                error_logs.append(log)
                continue
            
            # Check message content
            message = log.get("message", "") + " " + log.get("raw", "")
            
            for pattern in self.compiled_patterns:
                if pattern.search(message):
                    error_logs.append(log)
                    break
        
        return error_logs
    
    def aggregate_by_pattern(
        self,
        logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate logs by error pattern"""
        
        patterns = {}
        
        for log in logs:
            message = log.get("message", "")
            
            # Extract pattern (simplified)
            # In production, use more sophisticated pattern extraction
            pattern_key = self._extract_pattern(message)
            
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            
            patterns[pattern_key].append(log)
        
        return patterns
    
    def _extract_pattern(self, message: str) -> str:
        """Extract log pattern from message"""
        
        # Simple pattern extraction
        # Replace numbers, IDs, timestamps with placeholders
        
        pattern = message
        
        # Replace UUIDs
        pattern = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '<UUID>',
            pattern,
            flags=re.IGNORECASE
        )
        
        # Replace numbers
        pattern = re.sub(r'\b\d+\b', '<NUM>', pattern)
        
        # Replace timestamps
        pattern = re.sub(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            '<TIMESTAMP>',
            pattern
        )
        
        return pattern
    
    def search_logs(
        self,
        query: str,
        timeframe: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search logs in buffer
        
        Args:
            query: Search query
            timeframe: Optional timeframe
            limit: Maximum results
        
        Returns:
            Matching logs
        """
        
        results = []
        
        # Parse timeframe
        if timeframe:
            cutoff = datetime.utcnow() - self._parse_timeframe(timeframe)
        else:
            cutoff = None
        
        # Search in buffer
        for log in reversed(self.log_buffer):
            # Check timeframe
            if cutoff and log.get("timestamp", datetime.min) < cutoff:
                continue
            
            # Check if query matches
            message = log.get("message", "") + " " + log.get("raw", "")
            
            if query.lower() in message.lower():
                results.append(log)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def _parse_timeframe(self, timeframe: str) -> timedelta:
        """Parse timeframe string to timedelta"""
        
        # Simple parser: "5m" -> 5 minutes, "1h" -> 1 hour
        if timeframe.endswith("m"):
            return timedelta(minutes=int(timeframe[:-1]))
        elif timeframe.endswith("h"):
            return timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith("d"):
            return timedelta(days=int(timeframe[:-1]))
        else:
            return timedelta(minutes=5)
    
    def get_log_statistics(
        self,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Get log statistics"""
        
        cutoff = datetime.utcnow() - self._parse_timeframe(timeframe)
        
        recent_logs = [
            log for log in self.log_buffer
            if log.get("timestamp", datetime.min) >= cutoff
        ]
        
        # Count by level
        by_level = {}
        for log in recent_logs:
            level = log.get("level", "unknown")
            by_level[level] = by_level.get(level, 0) + 1
        
        # Count by source
        by_source = {}
        for log in recent_logs:
            source = log.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            "total_logs": len(recent_logs),
            "timeframe": timeframe,
            "by_level": by_level,
            "by_source": by_source,
            "buffer_size": len(self.log_buffer)
        }