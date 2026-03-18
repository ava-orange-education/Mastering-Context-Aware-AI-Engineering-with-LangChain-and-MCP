"""
API routes module
"""

from .health_routes import router as health_router
from .agent_routes import router as agent_router
from .admin_routes import router as admin_router

__all__ = [
    'health_router',
    'agent_router',
    'admin_router',
]