"""
Inter-agent authentication for secure multi-agent systems.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import secrets
import hmac
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentCredentials:
    """Agent credentials"""
    agent_id: str
    agent_name: str
    secret_key: str
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True


@dataclass
class AgentToken:
    """Agent authentication token"""
    token_id: str
    agent_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str


class AgentAuthenticator:
    """Authenticates agents in multi-agent systems"""
    
    def __init__(self, token_lifetime_minutes: int = 60):
        """
        Initialize agent authenticator
        
        Args:
            token_lifetime_minutes: Token validity period
        """
        self.agents: Dict[str, AgentCredentials] = {}
        self.tokens: Dict[str, AgentToken] = {}
        self.token_lifetime = timedelta(minutes=token_lifetime_minutes)
    
    def register_agent(self, agent_id: str, agent_name: str, 
                      permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Register new agent
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Agent name
            permissions: Agent permissions
            
        Returns:
            Registration result with secret key
        """
        if agent_id in self.agents:
            return {'success': False, 'error': 'Agent already registered'}
        
        # Generate secret key
        secret_key = secrets.token_hex(32)
        
        credentials = AgentCredentials(
            agent_id=agent_id,
            agent_name=agent_name,
            secret_key=secret_key,
            permissions=permissions or []
        )
        
        self.agents[agent_id] = credentials
        
        logger.info(f"Agent registered: {agent_name} ({agent_id})")
        
        return {
            'success': True,
            'agent_id': agent_id,
            'secret_key': secret_key,
            'message': 'Store secret key securely - it will not be shown again'
        }
    
    def authenticate_agent(self, agent_id: str, secret_key: str) -> Dict[str, Any]:
        """
        Authenticate agent and issue token
        
        Args:
            agent_id: Agent ID
            secret_key: Secret key
            
        Returns:
            Authentication result with token
        """
        # Verify credentials
        credentials = self.agents.get(agent_id)
        
        if not credentials:
            return {'success': False, 'error': 'Invalid agent ID'}
        
        if not credentials.active:
            return {'success': False, 'error': 'Agent is deactivated'}
        
        if not hmac.compare_digest(credentials.secret_key, secret_key):
            return {'success': False, 'error': 'Invalid secret key'}
        
        # Generate token
        token = self._generate_token(agent_id)
        
        logger.info(f"Agent authenticated: {credentials.agent_name} ({agent_id})")
        
        return {
            'success': True,
            'token': token.token_id,
            'agent_id': agent_id,
            'agent_name': credentials.agent_name,
            'expires_at': token.expires_at.isoformat()
        }
    
    def verify_token(self, token_id: str) -> Dict[str, Any]:
        """
        Verify agent token
        
        Args:
            token_id: Token to verify
            
        Returns:
            Verification result
        """
        token = self.tokens.get(token_id)
        
        if not token:
            return {'valid': False, 'error': 'Invalid token'}
        
        # Check expiration
        if datetime.now() > token.expires_at:
            del self.tokens[token_id]
            return {'valid': False, 'error': 'Token expired'}
        
        # Verify signature
        expected_signature = self._compute_signature(token.agent_id, token.issued_at)
        
        if not hmac.compare_digest(token.signature, expected_signature):
            return {'valid': False, 'error': 'Invalid token signature'}
        
        # Get agent info
        credentials = self.agents.get(token.agent_id)
        
        if not credentials or not credentials.active:
            return {'valid': False, 'error': 'Agent not active'}
        
        return {
            'valid': True,
            'agent_id': credentials.agent_id,
            'agent_name': credentials.agent_name,
            'permissions': credentials.permissions
        }
    
    def revoke_token(self, token_id: str) -> Dict[str, Any]:
        """
        Revoke agent token
        
        Args:
            token_id: Token to revoke
            
        Returns:
            Revocation result
        """
        if token_id in self.tokens:
            del self.tokens[token_id]
            logger.info(f"Token revoked: {token_id}")
            return {'success': True}
        
        return {'success': False, 'error': 'Token not found'}
    
    def deactivate_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Deactivate agent
        
        Args:
            agent_id: Agent to deactivate
            
        Returns:
            Deactivation result
        """
        credentials = self.agents.get(agent_id)
        
        if not credentials:
            return {'success': False, 'error': 'Agent not found'}
        
        credentials.active = False
        
        # Revoke all tokens for this agent
        tokens_to_revoke = [
            tid for tid, token in self.tokens.items()
            if token.agent_id == agent_id
        ]
        
        for token_id in tokens_to_revoke:
            del self.tokens[token_id]
        
        logger.info(f"Agent deactivated: {credentials.agent_name} ({agent_id})")
        
        return {
            'success': True,
            'tokens_revoked': len(tokens_to_revoke)
        }
    
    def _generate_token(self, agent_id: str) -> AgentToken:
        """Generate authentication token"""
        token_id = f"agent_token_{secrets.token_hex(16)}"
        now = datetime.now()
        expires_at = now + self.token_lifetime
        
        signature = self._compute_signature(agent_id, now)
        
        token = AgentToken(
            token_id=token_id,
            agent_id=agent_id,
            issued_at=now,
            expires_at=expires_at,
            signature=signature
        )
        
        self.tokens[token_id] = token
        
        return token
    
    def _compute_signature(self, agent_id: str, timestamp: datetime) -> str:
        """Compute token signature"""
        credentials = self.agents.get(agent_id)
        
        if not credentials:
            return ""
        
        message = f"{agent_id}:{timestamp.isoformat()}".encode()
        signature = hmac.new(
            credentials.secret_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        
        return signature