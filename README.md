# Voxium

Voxium is a **100% local, privacy-first voice assistant and dictation engine**. Built to run entirely offline, it leverages state-of-the-art open-source AI models to offer real-time voice-to-text processing and intelligent command execution without ever sending your data to the cloud.

## Key Features

- **100% Offline & Private:** Your voice, data, and memory never leave your device.
- **Dual Pipeline Architecture:**
  - **Dictation Mode:** Seamless real-time voice-to-text transcription with auto-cleanup, punctuation, and formatting powered by local LLMs.
  - **Agent Mode:** Voice-activated AI assistant capable of interpreting natural language intents and executing local tools (e.g., managing the clipboard, querying memory).
- **LangGraph State Machine:** A robust, deterministic pipeline routing Voice Activity Detection (VAD), Speech-to-Text (STT), Intent Reasoning, and Tool Execution for maximum stability.
- **Persistent Graph-RAG Memory:** Leverages ChromaDB and Graphify to continuously learn from your conversations and instructions, allowing the agent to recall past context and user preferences over time.
- **Hardware Agnostic & Optimized:** Runs natively on CPU and GPU with optimized GGUF inference (via `llama-cpp-python`).
- **High-Fidelity Audio Stack:**
  - *Wake-Word & VAD:* Tiered Silero VAD dropping background noise and silence before expensive STT processing.
  - *Transcription (STT):* Lightning-fast offline transcription via Faster-Whisper.
  - *Synthesis (TTS):* Instant vocal responses powered by Piper TTS.

## Installation

### 1. Prerequisites
- Python 3.10+
- FFmpeg (Make sure it is installed and available in your system PATH)

### 2. Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/RKBehera223048/Voxium.git
cd Voxium

python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Model Downloads
Voxium requires offline models to function. Create a `models/` directory in the root and download your preferred models:
- **LLM:** Download a GGUF model (e.g., Mistral-7B-Instruct) into `models/llm/`.
- **TTS:** Piper ONNX voices into `models/tts/`.
- **STT:** Faster-Whisper models will be downloaded automatically on first run, but can be managed manually in `models/stt/`.

Create a `.env` file from the example and configure your model paths:
```bash
cp .env.example .env
```
Edit `.env` to point `LLM_MODEL_PATH` to your downloaded GGUF file.

## Usage

Start the backend server:
```bash
python app.py
```

The REST API and WebSocket endpoints will be available at `http://127.0.0.1:5000`. 
Voxium acts as the orchestrator backend. You can interface with it by streaming audio chunks to the `/api/transcribe` endpoint or through a dedicated desktop client.

## Testing

Voxium comes with an API test suite to verify graph node routing and model pipelines:
```bash
pytest tests/test_api.py -v
```

## Architecture
- `ai_pipelines/`: Core AI logic (transcription, reasoning).
- `voxgraph/`: LangGraph declarative state graph orchestrating the pipeline.
- `core/`: State management and memory graph retrieval.
- `audio/`: VAD, STT, and TTS engines.
- `memory/`: Vector and Graph-based local storage (ChromaDB + SQLite).
