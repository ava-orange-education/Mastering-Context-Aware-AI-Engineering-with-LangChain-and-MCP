"""
Authentication and session management for secure AI systems.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import secrets
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    active: bool = True
    mfa_enabled: bool = False


@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    active: bool = True


class AuthenticationManager:
    """Manages user authentication and sessions"""
    
    def __init__(self, session_timeout_minutes: int = 60):
        """
        Initialize authentication manager
        
        Args:
            session_timeout_minutes: Session timeout in minutes
        """
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def register_user(self, username: str, email: str, password: str, 
                     roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Register new user
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            roles: User roles
            
        Returns:
            Registration result
        """
        # Check if username already exists
        if any(u.username == username for u in self.users.values()):
            return {'success': False, 'error': 'Username already exists'}
        
        # Validate password strength
        if not self._validate_password(password):
            return {
                'success': False,
                'error': 'Password must be at least 8 characters with uppercase, lowercase, and numbers'
            }
        
        # Create user
        user_id = self._generate_user_id()
        password_hash = self._hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ['user']
        )
        
        self.users[user_id] = user
        
        logger.info(f"User registered: {username}")
        
        return {
            'success': True,
            'user_id': user_id,
            'username': username
        }
    
    def authenticate(self, username: str, password: str, 
                    ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate user and create session
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            
        Returns:
            Authentication result with session token
        """
        # Check if account is locked
        if self._is_locked_out(username):
            return {
                'success': False,
                'error': 'Account temporarily locked due to failed login attempts'
            }
        
        # Find user
        user = self._find_user_by_username(username)
        
        if not user:
            self._record_failed_attempt(username)
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._record_failed_attempt(username)
            return {'success': False, 'error': 'Invalid credentials'}
        
        # Check if user is active
        if not user.active:
            return {'success': False, 'error': 'Account is deactivated'}
        
        # Clear failed attempts
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        # Create session
        session = self._create_session(user.user_id, ip_address)
        
        # Update last login
        user.last_login = datetime.now()
        
        logger.info(f"User authenticated: {username}")
        
        return {
            'success': True,
            'session_token': session.session_id,
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'expires_at': session.expires_at.isoformat()
        }
    
    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """
        Validate session token
        
        Args:
            session_token: Session token
            
        Returns:
            Validation result with user info
        """
        session = self.sessions.get(session_token)
        
        if not session:
            return {'valid': False, 'error': 'Invalid session token'}
        
        if not session.active:
            return {'valid': False, 'error': 'Session is inactive'}
        
        if datetime.now() > session.expires_at:
            session.active = False
            return {'valid': False, 'error': 'Session expired'}
        
        # Get user
        user = self.users.get(session.user_id)
        
        if not user or not user.active:
            session.active = False
            return {'valid': False, 'error': 'User account invalid'}
        
        return {
            'valid': True,
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'permissions': user.permissions
        }
    
    def logout(self, session_token: str) -> Dict[str, Any]:
        """
        Logout user and invalidate session
        
        Args:
            session_token: Session token
            
        Returns:
            Logout result
        """
        session = self.sessions.get(session_token)
        
        if session:
            session.active = False
            logger.info(f"User logged out: session {session_token}")
            return {'success': True}
        
        return {'success': False, 'error': 'Invalid session'}
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{secrets.token_hex(16)}"
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{pwd_hash}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwd_hash = password_hash.split(':')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == pwd_hash
        except:
            return False
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit
    
    def _create_session(self, user_id: str, ip_address: Optional[str]) -> Session:
        """Create new session"""
        session_id = f"sess_{secrets.token_hex(32)}"
        now = datetime.now()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address
        )
        
        self.sessions[session_id] = session
        
        return session
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now())
        
        # Keep only recent attempts
        cutoff = datetime.now() - self.lockout_duration
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]
    
    def _is_locked_out(self, username: str) -> bool:
        """Check if account is locked out"""
        if username not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > datetime.now() - self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts