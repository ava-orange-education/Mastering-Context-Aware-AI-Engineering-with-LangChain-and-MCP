#!/usr/bin/env python3
"""
Example script to run the Chapter 5 Multimodal Personal Assistant and MCP server.

Usage:
  1. Keep your API key in a .env file (copy .env.example to .env and set ANTHROPIC_API_KEY).

  2. From this directory (chapter-5-multimodal-ai), run:
       python run_example.py assistant   # Run assistant demo
       python run_example.py mcp          # Start MCP server (placeholder)
       python run_example.py core         # Minimal core agent demo (image + prompt)
"""

import os
import sys

# Load .env so ANTHROPIC_API_KEY (and optional HF_TOKEN) are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use export ANTHROPIC_API_KEY instead

# Suppress "unauthenticated requests to HF Hub" warning when HF_TOKEN is not set
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_VERBOSITY"] = "error"


def run_core_agent_demo():
    """Minimal demo: core multimodal agent with Claude (image + prompt)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in .env (copy .env.example to .env) or export it.")
        return 1

    # Use core agent only (no vision/audio heavy deps)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.multimodal_agent import MultiModalAgent

    agent = MultiModalAgent(api_key=api_key)
    print("Core MultiModalAgent ready. Example usage:")
    print("  result = agent.analyze_image('path/to/image.jpg', 'What is in this image?')")
    print("  print(result)")
    return 0


def run_assistant_demo():
    """Full personal assistant (requires vision/audio stack: torch, whisper, clip, etc.)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in .env (copy .env.example to .env) or export it.")
        return 1

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from personal_assistant import MultiModalPersonalAssistant

    assistant = MultiModalPersonalAssistant(api_key=api_key, enable_cache=True)
    print("MultiModalPersonalAssistant ready. Example:")
    print("  result = assistant.process_request({")
    print("      'task': 'visual_qa', 'image_path': 'image.jpg', 'question': 'Describe this'")
    print("  })")
    return 0


def run_mcp_server():
    """Start MCP server (placeholder – prints server info)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY in .env (copy .env.example to .env) or export it.")
        return 1

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mcp_agents import MCPServer

    server = MCPServer(api_key=api_key, host="localhost", port=8080)
    server.start()
    return 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    cmd = sys.argv[1].lower()
    if cmd == "core":
        return run_core_agent_demo()
    if cmd == "assistant":
        return run_assistant_demo()
    if cmd == "mcp":
        return run_mcp_server()
    print("Unknown command. Use: assistant | mcp | core")
    return 1


if __name__ == "__main__":
    sys.exit(main())
