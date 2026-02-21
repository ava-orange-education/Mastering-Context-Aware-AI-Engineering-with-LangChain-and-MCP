"""
MCP-based multi-agent system for multimodal processing.
"""

from .vision_agent import VisionAgent
from .audio_agent import AudioAgent
from .document_agent import DocumentAgent
from .orchestrator import MultiModalOrchestrator
from .mcp_server import MCPServer

__all__ = [
    'VisionAgent',
    'AudioAgent',
    'DocumentAgent',
    'MultiModalOrchestrator',
    'MCPServer'
]