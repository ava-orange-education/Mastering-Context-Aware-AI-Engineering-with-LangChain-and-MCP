"""
Deployment-specific configuration
"""

from typing import Dict


class DeploymentConfig:
    """Deployment configuration for different environments"""
    
    DEVELOPMENT = {
        "replicas": 1,
        "cpu_request": "100m",
        "cpu_limit": "500m",
        "memory_request": "256Mi",
        "memory_limit": "512Mi",
        "autoscaling_enabled": False,
    }
    
    STAGING = {
        "replicas": 2,
        "cpu_request": "200m",
        "cpu_limit": "1000m",
        "memory_request": "512Mi",
        "memory_limit": "1Gi",
        "autoscaling_enabled": True,
        "min_replicas": 2,
        "max_replicas": 5,
        "target_cpu_utilization": 70,
    }
    
    PRODUCTION = {
        "replicas": 3,
        "cpu_request": "500m",
        "cpu_limit": "2000m",
        "memory_request": "1Gi",
        "memory_limit": "2Gi",
        "autoscaling_enabled": True,
        "min_replicas": 3,
        "max_replicas": 10,
        "target_cpu_utilization": 60,
    }
    
    @classmethod
    def get_config(cls, environment: str) -> Dict:
        """Get configuration for specific environment"""
        return getattr(cls, environment.upper(), cls.DEVELOPMENT)