"""
API-specific configuration
"""

from typing import List


class APIConfig:
    """API configuration and constants"""
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # API Metadata
    API_TITLE: str = "RAG Multi-Agent System API"
    API_DESCRIPTION: str = """
    Production REST API for RAG-enabled multi-agent system.
    
    Features:
    - Document retrieval and analysis
    - Multi-agent query processing
    - Authentication and rate limiting
    - Health monitoring
    """
    
    API_VERSION: str = "1.0.0"
    
    # Request/Response
    MAX_QUERY_LENGTH: int = 1000
    MAX_RESPONSE_LENGTH: int = 5000
    DEFAULT_TOP_K: int = 5
    
    # Timeouts
    AGENT_TIMEOUT: int = 30
    RETRIEVAL_TIMEOUT: int = 10
    
    # Paths
    API_PREFIX: str = "/api/v1"
    HEALTH_PATH: str = "/health"
    METRICS_PATH: str = "/metrics"