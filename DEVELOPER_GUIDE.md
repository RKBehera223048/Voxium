# Voxium Developer & Debugger Guide

Welcome to the Voxium codebase! This document is a comprehensive guide designed for collaborators and debuggers to understand the architecture, data flow, and file structure of the project completely. 

Voxium is a 100% local, privacy-first Voice Assistant. It operates entirely on the CPU using Flask and WebSockets, merging OpenWhispr's robust audio routing and Graphify's topology-based knowledge management.

---

## 📂 Codebase Directory Structure

```text
Voxium/
├── app.py                      # Main entry point (Flask, Routes, SocketIO)
├── ai_pipelines/               # Core machine learning & AI modules
│   ├── ingestion.py            # VAD Pipeline (RMS + Silero)
│   ├── transcription.py        # STT Engine (Whisper & Parakeet ONNX)
│   ├── diarization.py          # Speaker Identification & Embedding Manager
│   ├── reasoning.py            # LLM inference via llama-cpp
│   └── synthesis.py            # TTS processing
├── core/                       # Core orchestration and memory systems
│   ├── orchestrator.py         # Main event loop, pipelines, & intent routing
│   ├── state_manager.py        # Session state management
│   ├── memory_graph.py         # Graph-RAG memory graph (NetworkX)
│   ├── entity_extractor.py     # Regex/Hash based entity extraction
│   ├── graph_clustering.py     # Community detection algorithms
│   └── prompts.py              # Prompt engineering for LLM/Cleanup
├── data/db/                    # Local storage layer
│   └── speaker_db.py           # SQLite DB for persistent speaker profiles
├── mcp_tools/                  # Agent tool implementations
│   ├── document_editor.py      # Deterministic AST parsing
│   ├── calendar_tool.py        # Calendar automations
│   └── web_automation.py       # Web automation tools
└── utils/                      # Utilities and helpers
    ├── audio_utils.py          # PCM conversions, Audio gating
    └── text_cleaner.py         # Text normalization post-transcription
```

---

## 🧠 Architectural Deep-Dive & Data Flow

### 1. The Orchestrator (`core/orchestrator.py`)
This is the heart of Voxium. It replaces traditional IPC mechanisms with a robust Python `asyncio.Queue` system.

**The Audio Lifecycle (`process_event`):**
1. **Event Ingestion:** Audio bytes arrive via WebSocket (`/audio_chunk`) or REST (`/api/transcribe`) and are queued as an `AudioEvent`.
2. **VAD Pre-Filter:** `ai_pipelines/ingestion.py` processes the audio. First, it hits a lightweight RMS gate (sub-ms latency) to strip pure silence. If speech is detected, it hits a Silero VAD model (512-sample window) to confirm speech. **(Anti-Hallucination feature)**
3. **Transcription:** Audio is sent to `ai_pipelines/transcription.py`. It uses `sherpa-onnx` bindings (Parakeet/Whisper) running multi-threaded on the CPU to convert speech to text.
4. **Intent Routing:** `resolve_route_kind()` determines the text path:
   - `AGENT`: If the hotkey was "voice_agent" or a wake word ("Hey Voxium") was detected.
   - `CLEANUP`: If it's a standard dictation, the LLM will rewrite/format it.
   - `SKIP`: Raw dictation is returned.
5. **Execution & Result:** Depending on the route, it queries the `LocalReasoningEngine` (`ai_pipelines/reasoning.py`), updates session history, and emits the final payload via WebSocket.

### 2. Local Memory Graph (`core/memory_graph.py`)
Voxium relies on a deterministic **Graph-RAG Memory**.
- Rather than expensive vector embeddings, `core/entity_extractor.py` uses Regex and NLP heuristics to find entities (names, dates, concepts) and relationships.
- These nodes and edges are stored in a `NetworkX` graph.
- Every conversation turn is appended to this graph via `StateManager.add_turn()`.
- **Querying:** When the user asks a question, BFS traversal injects neighbor context directly into the LLM prompt.
- **Clustering:** `core/graph_clustering.py` groups the nodes into distinct topics over time using Leiden/Louvain modularity optimization.

### 3. Speaker Diarization (`ai_pipelines/diarization.py`)
- Real-time speaker extraction calculates the centroid of audio embeddings.
- It compares the current embedding against profiles stored permanently in SQLite via `data/db/speaker_db.py`.
- Constants like `MATCH_THRESHOLD = 0.65` control the sensitivity of recognizing a known speaker.

### 4. Zero-LLM Tool Parsing (`mcp_tools/document_editor.py`)
- To save LLM compute during coding and file manipulation, Voxium uses the native Python `ast` module.
- It calculates line ranges deterministically to slice relevant context for the LLM before any AI action occurs.

---

## 🐞 Debugging Guide

### Log Configuration
Voxium uses native Python `logging`. 
- To enable ultra-verbose logging, ensure `VOXIUM_DEBUG=true` is set in the `.env` file. 
- You will see logs tagged as `[DEBUG] voxium: ...`

### Common Failure Points & Debugging Steps
1. **Transcription Hallucinations:**
   - *Symptom:* The STT engine starts spitting out random text like "Thanks for watching."
   - *Fix:* Check `utils/audio_utils.py` and tweak the thresholds. Specifically, increase `SILENCE_RMS` or `SPEECH_WINDOW_PEAK`. Check VAD rejection logs in the orchestrator.
2. **WebSocket Timeouts:**
   - *Symptom:* Frontend says "Processing..." forever.
   - *Fix:* Ensure the `asyncio` loop running the Orchestrator hasn't deadlocked. Add `logging.debug()` inside `_process_event()` in `core/orchestrator.py` to trace if VAD, STT, or LLM reasoning is hanging.
3. **Agent Route not Triggering:**
   - *Symptom:* Saying "Hey Voxium, edit this file" just outputs dictation.
   - *Fix:* Inspect `core/prompts.py` -> `detect_agent_invocation()`. The wake word regex might be too strict.

### Key REST Endpoints for Debugging
- `GET /api/status`: Check if the Orchestrator loop is actively running and engines are loaded.
- `GET /api/graph/stats`: See if the memory graph is actively ingesting tokens.

---

## 👩‍💻 Contributing

1. **Adding a new AI Pipeline:** Add it to the `ai_pipelines/` directory. Ensure it initializes gracefully and cleanly loads models into memory only when invoked to keep idle RAM usage low.
2. **Modifying Routes:** Any new prompt paths or intents must be updated in `core/orchestrator.py` -> `resolve_route_kind()`.
3. **Environment Additions:** Ensure any new toggle features (like enabling/disabling a model) are added to the `.env` template and mapped in `app.py`.
