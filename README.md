# ARA

Privacy-first, local-first, voice-first personal AI assistant.

Runs entirely on RTX 4090 + Ryzen 9 + 64GB RAM on Windows. No cloud dependencies. No telemetry. Your data stays on your machine.

## Features

- **Voice-first interface** — wake word detection, streaming STT, natural TTS
- **Deep personal memory** — 4+ years of diary data indexed in vector DB + knowledge graph
- **Sarcastic personality** — warm humor with a configurable dial (1-10)
- **Agent orchestration** — intent routing, RAG, web search, desktop automation
- **Vision** — image analysis via Moondream 2B + Qwen-VL
- **Full-stack dashboard** — Next.js 16 with chat, memory browser, settings
- **Fine-tuning pipeline** — QLoRA training infrastructure for personality injection

## Architecture

```
Voice Input ─── Wake Word ─── VAD ─── STT (Moonshine v2)
                                         │
                                    Orchestrator (LangGraph)
                                    ├── Intent Router
                                    ├── RAG Agent (LanceDB + Knowledge Graph)
                                    ├── Memory Agent
                                    ├── Web Search Agent
                                    ├── Desktop Tool Agent
                                    └── Vision Agent
                                         │
                                    LLM (Qwen3-30B-A3B via Ollama)
                                         │
                                    TTS (Kokoro-82M) ─── Speaker
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CPU | AMD Ryzen 9 or equivalent |
| RAM | 64GB DDR5 |
| Storage | NVMe SSD (50GB+ free for models and data) |
| OS | Windows 10/11 (native, no WSL) |

## Tech Stack

| Component | Choice |
|-----------|--------|
| Core LLM | Qwen3-30B-A3B (MoE, Q4_K_M) via Ollama |
| STT | Moonshine v2 Medium (CPU, ONNX) |
| TTS | Kokoro-82M (CPU, ONNX) |
| Wake Word | openWakeWord |
| VAD | Silero VAD |
| Embeddings | nomic-embed-text-v1.5 (CPU) |
| Vector DB | LanceDB (hybrid search) |
| Reranker | bge-reranker-v2-m3 (CPU) |
| Knowledge Graph | NetworkX + SQLite |
| Vision | Moondream 2B (4-bit) + Qwen-VL |
| Orchestration | LangGraph + LlamaIndex |
| API | FastAPI |
| Frontend | Next.js 16 + Tailwind + shadcn/ui |
| Fine-tuning | QLoRA via Unsloth + TRL |

## Quick Start

### Prerequisites

1. [Python 3.11+](https://python.org)
2. [uv](https://docs.astral.sh/uv/) (Python package manager)
3. [Ollama](https://ollama.com/download) (LLM inference)
4. [Node.js 20+](https://nodejs.org) (for frontend)
5. NVIDIA GPU with CUDA drivers

### Setup

```bash
# Clone the repository
git clone https://github.com/0xdenjii/ai-ara.git
cd ai-ara

# Run the automated setup
uv run python scripts/setup.py

# Or manual setup:
uv sync --extra dev              # Install Python dependencies
cp .env.example .env             # Create environment config
ollama pull qwen3:30b-a3b        # Download primary LLM (~18GB)
ollama pull nomic-embed-text     # Download embedding model
cd frontend && npm install       # Install frontend dependencies
```

### Running

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start API server
uv run uvicorn ara.api.app:app --host 127.0.0.1 --port 8000

# Terminal 3: Start frontend
cd frontend && npm run dev

# Open http://localhost:3000
```

### CLI Chat

```bash
uv run python scripts/chat.py             # Interactive chat with memory
uv run python scripts/chat.py --no-memory  # Chat without memory retrieval
```

## Project Structure

```
ai-ara/
├── src/ara/
│   ├── config/          # Pydantic settings, model configs
│   ├── core/            # LLM client, streaming, prompts
│   ├── voice/           # Audio I/O, VAD, STT, TTS, wake word, pipeline
│   ├── memory/          # Chunking, embeddings, vector store, knowledge graph
│   ├── agents/          # Orchestrator, router, RAG/memory/tool/web agents
│   ├── tools/           # Desktop automation, system tools, MCP bridge
│   ├── vision/          # Moondream, Qwen-VL, vision pipeline
│   ├── web/             # Search, browser, research, connectivity
│   ├── fine_tuning/     # Dataset generation, QLoRA training, evaluation
│   ├── api/             # FastAPI app, routes, WebSocket, schemas
│   └── utils/           # Error hierarchy, logging, types
├── frontend/            # Next.js 16 dashboard
├── scripts/             # CLI tools (chat, ingest, benchmark, setup)
├── tests/
│   ├── unit/            # 1,211+ unit tests
│   └── integration/     # 79 integration tests
├── data/                # Vector DB, conversations, models (gitignored)
└── dairy/               # Personal diary data (gitignored)
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/chat.py` | Interactive CLI chat |
| `scripts/ingest.py` | Ingest diary files into memory |
| `scripts/preprocess_diary.py` | Scan and classify diary data |
| `scripts/analyze_diary.py` | Explore diary data statistics |
| `scripts/generate_dataset.py` | Fine-tuning dataset generation |
| `scripts/benchmark_llm.py` | LLM inference benchmarks |
| `scripts/benchmark.py` | Combined benchmark suite |
| `scripts/test_voice_e2e.py` | Interactive voice pipeline test |
| `scripts/setup.py` | One-command environment bootstrap |
| `scripts/autostart.py` | Windows autostart configuration |

## Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/unit/test_api.py -v

# Run integration tests
uv run python -m pytest tests/integration/ -v

# Lint
uv run ruff check src/ tests/ scripts/
```

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and customize:

```bash
# Key settings
ARA_PRIMARY_MODEL=qwen3:30b-a3b    # Main LLM
ARA_HUMOR_DIAL=7                    # 1 (supportive) to 10 (unfiltered)
ARA_INTERNET_ENABLED=true           # Enable web search
ARA_DESKTOP_ENABLED=true            # Enable desktop automation
```

See `.env.example` for all available settings with documentation.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests with coverage
uv run python -m pytest tests/ --cov=ara --cov-report=term-missing

# Build frontend
cd frontend && npx next build

# Run benchmarks
uv run python scripts/benchmark.py
```

## License

Private. All rights reserved.
