# 🎙️ Voxium

**Voxium** is a next-generation, 100% offline, privacy-first voice assistant and dictation engine. Designed for complete data sovereignty, Voxium leverages state-of-the-art open-source AI models to offer real-time transcription, intelligent intent parsing, and autonomous command execution—all without ever sending a single byte of your data to the cloud.

---

## 🌟 Core Features

### 🔒 100% Offline & Private
Your voice, your data, and your memory never leave your device. Voxium requires **no API keys, no internet connection, and no subscriptions**. It is built for environments where privacy and security are non-negotiable.

### 🔀 Dual-Pipeline Architecture
Voxium intelligently routes your voice input based on your intent:
1. **Dictation Mode:** Seamlessly dictates your speech into highly accurate text. An integrated LLM cleanup node automatically fixes punctuation, formats paragraphs, and corrects grammar before copying it to your clipboard or editor.
2. **Agent Mode:** A fully autonomous voice assistant. Ask Voxium questions or give it commands, and it will reason through your request, invoke local tools, and reply to you using natural text-to-speech.

### 🧠 LangGraph State Machine Orchestrator
At the heart of Voxium is a robust, deterministic state graph. Complex processing is seamlessly orchestrated through a pipeline:
`Audio Stream ➔ Voice Activity Detection (VAD) ➔ Speech-to-Text (STT) ➔ Intent Routing ➔ LLM Reasoning / Cleanup ➔ Tool Execution ➔ Text-to-Speech (TTS)`
This ensures maximum stability, easy extensibility, and rapid short-circuiting (e.g., dropping background noise before expensive STT processing).

### 🕸️ Persistent Graph-RAG Memory
Voxium remembers. Using **ChromaDB** coupled with a custom **Graphify memory engine**, Voxium builds a persistent knowledge graph of your conversations, instructions, and preferences. Over time, the agent learns context about you, allowing for highly personalized assistance.

### 💻 Hardware Agnostic LLM Execution
Voxium is optimized for the hardware you already own. By leveraging `llama-cpp-python` and the **GGUF** model format, Voxium runs highly optimized LLMs natively on standard CPUs, Apple Silicon (Metal), or NVIDIA GPUs (CUDA).

### 🎵 High-Fidelity Local Audio Stack
- **Noise Rejection (VAD):** A multi-tier pipeline using Silero Neural VAD filters out background noise and silence, ensuring the transcription engine is only triggered by actual human speech.
- **Lightning-Fast Transcription (STT):** Powered by **Faster-Whisper**, delivering near-instantaneous offline transcription.
- **Natural Synthesis (TTS):** Instant, conversational vocal responses powered by **Piper TTS**.

### 🛠️ Extensible Tool System
Voxium's Agent can autonomously invoke local tools based on your intent. Built-in tools include:
- **Clipboard Manager:** Read from or write to your system clipboard.
- **File & Document Manager:** Create, edit, and read local documents and markdown files.
- **Local Search:** Retrieve information from your personal knowledge base.

### 🌐 Rich Local Frontend
Voxium includes a beautifully crafted, dynamic Web UI (HTML/JS/CSS) featuring a dashboard, text editor, meeting notes, and settings page. It communicates with the Python backend in real-time via WebSockets.

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **FFmpeg:** Ensure FFmpeg is installed and added to your system `PATH` (required for audio chunk processing).

### 2. Environment Setup
Clone the repository and install the dependencies in a virtual environment:
```bash
git clone https://github.com/RKBehera223048/Voxium.git
cd Voxium

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Downloads
Voxium requires offline models to function. Create a `models/` directory in the root of the project and download your preferred open-source models:
- **LLM:** Download a GGUF model (e.g., *Mistral-7B-Instruct* or *Llama-3-8B*) and place it in `models/llm/`.
- **TTS:** Download a Piper ONNX voice model into `models/tts/`.
- **STT:** Faster-Whisper models will be downloaded automatically to `models/stt/` on first run.

Create your environment configuration file:
```bash
cp .env.example .env
```
Edit the `.env` file to point `LLM_MODEL_PATH` to your downloaded GGUF file.

---

## 🖥️ Usage

Start the backend orchestrator:
```bash
python app.py
```

- The REST API and WebSocket endpoints will boot up at `http://127.0.0.1:5000`. 
- Open `http://127.0.0.1:5000` in your browser to access the Voxium Dashboard and interface directly with the assistant.
- Developers can interface with Voxium by streaming audio chunks to the `/api/transcribe` endpoint.

---

## 🧪 Testing

Voxium comes with an extensive Pytest suite to verify graph node routing, conditional edges, and model pipelines without requiring hardware microphones:

```bash
pytest tests/test_api.py -v
```

---

## 🏗️ Project Architecture

- `ai_pipelines/` & `llm/`: Core AI reasoning, prompt management, and inference logic.
- `voxgraph/`: The declarative LangGraph state machine orchestrating the end-to-end pipeline.
- `core/`: Application state management, event bus, and intent routing.
- `audio/`: VAD, STT, and TTS engines.
- `memory/`: Vector and Graph-based local storage (ChromaDB + SQLite).
- `tools/`: Extensible tool definitions invoked by the Agent.
- `frontend/`: The local Web UI and WebSocket client logic.
