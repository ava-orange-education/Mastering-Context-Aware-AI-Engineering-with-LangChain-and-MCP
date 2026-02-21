# Chapter 5: Multi-Modal AI Integration

A comprehensive multi-modal AI system featuring vision, audio, and document processing capabilities with MCP-based agent orchestration. Includes a **multimodal personal assistant** and **MCP (Model Context Protocol) server** for document Q&A, image understanding, audio transcription, meeting summarization, receipt analysis, and cross-modal reasoning.

---

## 🌟 Features

- **Vision Processing**
  - CLIP for zero-shot classification and semantic search
  - BLIP/BLIP-2 for image captioning and VQA
  - Grounding DINO for open-vocabulary object detection
  - Claude Vision for advanced image analysis

- **Audio Processing**
  - Whisper for transcription and translation
  - Multi-language support
  - Voice assistant capabilities

- **Document Processing**
  - PDF, DOCX, image text extraction
  - Document Q&A
  - Multi-page analysis

- **Multi-Agent System**
  - MCP-based orchestration
  - Specialized agents for each modality
  - Cross-modal reasoning
  - Intelligent query routing

- **Personal Assistant**
  - Receipt analysis
  - Meeting transcription and summarization
  - Document verification
  - Comprehensive caching and monitoring

---

## 📋 Prerequisites

- **Python 3.10+** (3.8+ may work; 3.10+ recommended)
- **CUDA-capable GPU** (recommended for vision models; CPU supported)
- **Anthropic API key** – used by the assistant and MCP server for Claude. Get one at [console.anthropic.com](https://console.anthropic.com).
- **Tesseract** (for document OCR; required for PDF/image text extraction):
  - **Linux (Ubuntu/Debian):** `sudo apt install tesseract-ocr`
  - **macOS:** `brew install tesseract`
  - **Windows:** Install from [GitHub – tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **`.env` file** – create from `.env.example` and add your API key (see [Tokens and API keys](#tokens-and-api-keys) below).
- **Optional API keys:** OpenAI (GPT-4 Vision), Google (Gemini).

### Tokens and API keys

| Variable | Required | Purpose |
|----------|----------|---------|
| **`ANTHROPIC_API_KEY`** | Yes | Claude API access for the personal assistant and MCP server. Set in `.env` or export in the terminal. |
| **`HF_TOKEN`** | No | Hugging Face token for downloading BLIP/transformers models. Enables higher rate limits and avoids “unauthenticated requests” warnings. Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Add to `.env` or export. |

**Example `.env`:**

```bash
# Required for assistant and MCP
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: fewer HF download warnings and higher rate limits
HF_TOKEN=your_huggingface_token_here
```

Do not commit `.env`; it is listed in `.gitignore`.

---

## 🚀 Quick start

### 1. Clone/navigate to the chapter folder

```bash
git clone https://github.com/yourusername/chapter-5-multimodal-ai.git  # or use your repo URL
cd chapter-5-multimodal-ai
# Or if already in the book repo:
cd Mastering-Context-Aware-AI-Engineering-with-LangChain-and-MCP/chapter-5-multimodal-ai
```

### 2. Create and activate a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install **CLIP** (required for vision agent image search/classification):

```bash
pip install git+https://github.com/openai/CLIP.git
```

**Optional** – install the package in editable mode (if you have `setup.py`):

```bash
pip install -e .
```

**Optional** – text-to-speech for the voice assistant:

```bash
pip install gtts
```

**Optional** – Gemini integration:

```bash
pip install google-generativeai
```

### 4. Set up tokens in a `.env` file

Create a `.env` file in this directory (see [Tokens and API keys](#tokens-and-api-keys) for the full list):

```bash
cp .env.example .env
# Edit .env and add at least ANTHROPIC_API_KEY=your_anthropic_api_key_here
# Optionally: HF_TOKEN=your_huggingface_token_here
```

The run script and examples load `.env` automatically. **Do not commit `.env`** (it is in `.gitignore`).

If you prefer not to use `.env`, set the key in the terminal before running:

- **Linux / macOS:** `export ANTHROPIC_API_KEY=your_key_here`
- **Windows PowerShell:** `$env:ANTHROPIC_API_KEY = "your_key_here"`
- **Windows CMD:** `set ANTHROPIC_API_KEY=your_key_here`

### 5. Run

| Command | Description |
|--------|-------------|
| `python run_example.py core` | Core agent only (minimal deps: Claude + image) |
| `python run_example.py assistant` | Full personal assistant (vision, audio, document) |
| `python run_example.py mcp` | Start MCP server (prints server info) |

With no arguments, the script prints usage:

```bash
python run_example.py
```

---

## Basic usage

```python
import os
from dotenv import load_dotenv
load_dotenv()

from personal_assistant import MultiModalPersonalAssistant

assistant = MultiModalPersonalAssistant(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    enable_cache=True
)

# Analyze an image
result = assistant.analyze_image(
    image_path="path/to/image.jpg",
    query="What's in this image?"
)
print(result.get("result"))
```

---

## 📚 Examples

The `examples/` directory contains end-to-end examples:

| Example | Description |
|--------|-------------|
| **01_basic_vision.py** | Basic vision analysis with Claude |
| **02_image_captioning.py** | Image captioning with BLIP |
| **03_audio_transcription.py** | Audio transcription with Whisper |
| **04_document_qa.py** | Document question answering |
| **05_clip_search.py** | Semantic image search with CLIP |
| **06_grounding_dino.py** | Object detection |
| **07_mcp_agents.py** | Multi-agent coordination |
| **08_personal_assistant.py** | Personal assistant features |
| **09_cross_modal_reasoning.py** | Cross-modal analysis |
| **10_full_workflow.py** | Complete end-to-end workflow |

Run any example (ensure `.env` is set):

```bash
python examples/01_basic_vision.py
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Personal Assistant Layer                 │
│  (Capabilities, Validation, Caching)             │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│        Multi-Agent Orchestrator                  │
│     (Query Routing, Result Fusion)               │
└─────────┬───────────┬──────────────┬─────────────┘
          │           │              │
    ┌─────▼─────┐ ┌───▼────┐  ┌─────▼─────┐
    │  Vision   │ │ Audio  │  │ Document  │
    │  Agent    │ │ Agent  │  │  Agent    │
    └───────────┘ └────────┘  └──────────┘
```

---

## Usage examples (detailed)

### Core agent (minimal)

Uses only `anthropic` and `Pillow`. No Whisper/CLIP/BLIP.

```bash
python run_example.py core
```

```python
import os
from dotenv import load_dotenv
load_dotenv()
from core.multimodal_agent import MultiModalAgent

agent = MultiModalAgent(api_key=os.environ["ANTHROPIC_API_KEY"])
result = agent.analyze_image("path/to/image.jpg", "What is in this image?")
print(result)
```

### Full personal assistant

Requires full stack: `torch`, `transformers`, `openai-whisper`, CLIP.

```bash
python run_example.py assistant
```

```python
import os
from dotenv import load_dotenv
load_dotenv()
from personal_assistant import MultiModalPersonalAssistant

assistant = MultiModalPersonalAssistant(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    enable_cache=True
)

# Visual Q&A
result = assistant.analyze_image("photo.jpg", "Describe this image.")

# Document Q&A
result = assistant.analyze_document("document.pdf", ["What is the main topic?"])

# Audio transcription
result = assistant.transcribe_audio("meeting.wav", language="en")

# Image search, receipt analysis, meeting transcription, cross-modal reasoning
result = assistant.process_request({
    "capability": "cross_modal_reasoning",
    "image_path": "slide.png",
    "document_path": "notes.pdf",
    "query": "Summarize how the image and document relate."
})
```

### MCP server

```bash
python run_example.py mcp
```

```python
import os
from dotenv import load_dotenv
load_dotenv()
from mcp_agents import MCPServer

server = MCPServer(api_key=os.environ["ANTHROPIC_API_KEY"], host="localhost", port=8080)
server.start()
```

---

## 🔧 Configuration

Configuration files are in `configs/`:

| File | Purpose |
|------|---------|
| **`model_config.yaml`** | Model settings and parameters |
| **`mcp_config.yaml`** | MCP server and agent configuration |
| **`assistant_config.yaml`** | Personal assistant settings |

---

## 📊 Performance monitoring

```python
from best_practices.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(assistant)

# Process with monitoring
result = monitor.monitored_request(request)

# Get performance report
report = monitor.get_performance_report()
print(report)
```

---

## 💰 Cost tracking

```python
from best_practices.cost_optimization import CostOptimizedAssistant

assistant = CostOptimizedAssistant(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    budget_limit=10.0  # $10 limit
)

result = assistant.process_request(request)
summary = assistant.get_cost_summary()
print(summary)
```

---

## Project layout

| Path | Purpose |
|------|--------|
| `core/` | Base multimodal agent (Claude + image) |
| `personal_assistant/` | Assistant with cache, validation, capabilities |
| `mcp_agents/` | Vision, audio, document agents, orchestrator, MCP server |
| `llm_integration/` | Claude, GPT-4 Vision, Gemini |
| `vision/` | CLIP, BLIP, Grounding DINO, image preprocessing |
| `audio/` | Whisper, voice assistant, preprocessing |
| `text/` | PDF + OCR document processing |
| `utils/` | Validation, file and base64 helpers |
| `best_practices/` | Error handling, cost optimization, monitoring |
| `configs/` | Model, MCP, and assistant configuration |
| `examples/` | Example scripts (01–10) |
| `tests/` | Test suite |
| `run_example.py` | Entry script: `core` \| `assistant` \| `mcp` |
| `requirements.txt` | Python dependencies |

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_vision.py
pytest tests/test_personal_assistant.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ImportError: No module named 'whisper'` or `'clip'` | Install: `pip install openai-whisper` and `pip install git+https://github.com/openai/CLIP.git`. Or use `python run_example.py core` to skip vision/audio. |
| `pytesseract.pytesseract.TesseractNotFoundError` | Install the Tesseract binary (see Prerequisites). |
| `ANTHROPIC_API_KEY` not set | Create `.env` from `.env.example` and set your key, or export it in the terminal. |
| Slow or OOM on CPU | Use smaller models (e.g. Whisper `tiny`/`base`) or run `python run_example.py core` for minimal usage. |
| GPU usage | PyTorch uses CUDA if available; otherwise CPU. |

---

## Requirements summary

- **Core:** `anthropic`, `Pillow`, `requests`, `numpy`, `PyPDF2`, `pytesseract`, `python-dotenv`
- **Vision/Audio:** `torch`, `torchaudio`, `transformers`, `openai-whisper`, CLIP (see install command above)
- **Optional:** `google-generativeai`, `gtts`; Grounding DINO from [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

See `requirements.txt` for versions and comments.

---

## 🔐 Security considerations

- API keys are stored in `.env` (never commit this file).
- Input validation is performed on all user inputs.
- File size limits are enforced.
- Supported file formats are restricted.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Anthropic for Claude API
- OpenAI for CLIP and Whisper
- Salesforce for BLIP/BLIP-2
- IDEA Research for Grounding DINO

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🗺️ Roadmap

- [ ] Add video processing capabilities
- [ ] Implement streaming audio transcription
- [ ] Add support for more LLM providers
- [ ] Create web interface
- [ ] Add batch processing utilities
- [ ] Implement advanced caching strategies

---

## 📖 Documentation

When available, full documentation may be provided in a `docs/` directory:

- Architecture Guide
- Vision Models Guide
- Audio Processing Guide
- MCP Setup Guide
- API Reference
