# How to Run Chapter 5 – Multimodal AI

This folder contains a **multimodal personal assistant** and **MCP (Model Context Protocol) server** with vision, audio, and document agents.

## 1. Prerequisites

- **Python 3.10+**
- **Anthropic API key** (for Claude). Get one at [console.anthropic.com](https://console.anthropic.com).
- **pytesseract**: OCR used by document processing. Install the Tesseract binary:
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`

## 2. Install Dependencies

From the `chapter-5-multimodal-ai` directory:

```bash
cd /path/to/Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/chapter-5-multimodal-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install **CLIP** (needed for vision agent image search / classification):

```bash
pip install git+https://github.com/openai/CLIP.git
```

Optional: for **text-to-speech** in the voice assistant:

```bash
pip install gtts
```

## 3. Set API Key

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 4. Run

### Option A: Core agent only (minimal – no Whisper/CLIP/BLIP)

Uses only `anthropic` and `Pillow` for image + text with Claude:

```bash
python run_example.py core
```

Then in Python:

```python
from core.multimodal_agent import MultiModalAgent
import os
agent = MultiModalAgent(api_key=os.environ["ANTHROPIC_API_KEY"])
result = agent.analyze_image("path/to/image.jpg", "What is in this image?")
print(result)
```

### Option B: Full personal assistant

Runs the full stack (vision, audio, document agents). Requires all dependencies including `torch`, `transformers`, `openai-whisper`, and CLIP:

```bash
python run_example.py assistant
```

Example request:

```python
from personal_assistant import MultiModalPersonalAssistant
import os
assistant = MultiModalPersonalAssistant(api_key=os.environ["ANTHROPIC_API_KEY"])
result = assistant.analyze_image("image.jpg", "Describe this image")
# or
result = assistant.process_request({
    "task": "visual_qa",
    "image_path": "image.jpg",
    "question": "What is in this image?"
})
```

### Option C: MCP server

Starts the MCP server (current implementation prints server info; no real network server yet):

```bash
python run_example.py mcp
```

## 5. Project layout (short)

| Path | Purpose |
|------|--------|
| `core/` | Base multimodal agent (Claude + image) |
| `personal_assistant/` | High-level assistant with cache, validation, capabilities |
| `mcp_agents/` | Vision, audio, document agents + orchestrator + MCP server |
| `llm_integration/` | Claude, GPT-4 Vision, Gemini integrations |
| `vision/` | CLIP, BLIP, Grounding DINO, image preprocessing |
| `audio/` | Whisper, voice assistant, preprocessing |
| `text/` | PDF + OCR document processing |
| `utils/` | Validation, file and base64 helpers |
| `best_practices/` | Error handling, cost optimization, monitoring |

## 6. Troubleshooting

- **ImportError for `whisper` or `clip`**: Install `openai-whisper` and CLIP (see step 2). Use `run_example.py core` if you want to avoid heavy vision/audio deps.
- **pytesseract not found**: Install the Tesseract binary (see Prerequisites).
- **CUDA/GPU**: PyTorch will use GPU if available; otherwise CPU is used.
