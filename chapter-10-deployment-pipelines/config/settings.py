"""
Application configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VectorStoreType(str, Enum):
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    app_name: str = "rag-multi-agent-system"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = False
    
    # Authentication
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # LLM API Keys
    anthropic_api_key: str
    openai_api_key: Optional[str] = None
    
    # Vector Database - Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "rag-embeddings"
    
    # Vector Database - Weaviate
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str = "Document"
    
    # Vector Store Selection
    vector_store: VectorStoreType = VectorStoreType.PINECONE
    
    # Kubernetes
    k8s_namespace: str = "rag-system"
    k8s_cluster_name: str = "production-cluster"
    
    # Docker
    docker_registry: str = "docker.io"
    docker_username: str = "your-username"
    docker_image_prefix: str = "rag-system"
    
    # Monitoring
    prometheus_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Resource Limits
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()