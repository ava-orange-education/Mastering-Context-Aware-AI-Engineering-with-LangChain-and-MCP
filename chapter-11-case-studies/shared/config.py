"""
Configuration management for all case studies
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    anthropic_api_key: str
    
    # Vector Database
    vector_store: str = "pinecone"
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = "case-studies"
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_class_name: str = "Document"
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    debug: bool = False
    
    # Healthcare
    healthcare_ehr_endpoint: Optional[str] = None
    healthcare_ehr_api_key: Optional[str] = None
    healthcare_encryption_key: Optional[str] = None
    healthcare_audit_db_url: Optional[str] = None
    
    # Enterprise
    enterprise_sharepoint_url: Optional[str] = None
    enterprise_sharepoint_client_id: Optional[str] = None
    enterprise_sharepoint_client_secret: Optional[str] = None
    enterprise_confluence_url: Optional[str] = None
    enterprise_confluence_api_token: Optional[str] = None
    enterprise_slack_bot_token: Optional[str] = None
    enterprise_ad_endpoint: Optional[str] = None
    
    # Education
    education_lms_url: Optional[str] = None
    education_lms_api_key: Optional[str] = None
    education_student_db_url: Optional[str] = None
    education_analytics_enabled: bool = True
    
    # DevOps
    devops_prometheus_url: Optional[str] = "http://localhost:9090"
    devops_elasticsearch_url: Optional[str] = "http://localhost:9200"
    devops_kubernetes_config: Optional[str] = None
    devops_pagerduty_api_key: Optional[str] = None
    devops_slack_webhook: Optional[str] = None
    
    # Security
    jwt_secret_key: str = "changeme"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Database
    database_url: Optional[str] = None
    redis_url: Optional[str] = "redis://localhost:6379/0"
    
    # Monitoring
    prometheus_enabled: bool = False
    metrics_port: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()