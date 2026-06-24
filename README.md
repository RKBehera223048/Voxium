# Voxium

Voxium is a 100% local, privacy-first Voice Assistant backend powered by a Flask and WebSocket architecture. It seamlessly integrates a robust speech-processing pipeline with an advanced local Graph-RAG memory layer, ensuring zero external API calls and complete data sovereignty.

## Core Philosophy
- **100% Local**: All processing—from Voice Activity Detection (VAD) and Speech-to-Text (STT) to Large Language Model (LLM) reasoning and memory clustering—happens on your machine.
- **Privacy-First**: No data leaves your device. Speaker profiles, transcription history, and the memory graph are all stored locally.

---

## 🚀 Features & Architecture

Voxium's architecture is a fusion of advanced speech processing techniques (ported from OpenWhispr) and topology-based knowledge management (ported from Graphify).

### 1. Advanced Speech Processing Pipeline

#### Strict Acoustic Gate (Anti-Hallucination)
- **How it works**: Audio goes through a dual-tier validation process before reaching the neural network. 
  - **Tier 1**: A lightweight windowed RMS/peak accumulator (sub-millisecond latency) filters out pure silence.
  - **Tier 2**: Silero VAD validates speech presence on 512-sample windows (~50ms latency).
- **Benefit**: Prevents the STT engine from hallucinating text on background noise or silence, saving compute resources.

#### High-Performance STT via NVIDIA Parakeet & Whisper
- **How it works**: Uses local STT engines with `sherpa-onnx` Python bindings. It auto-tunes threads based on CPU count and segments long audio streams for efficient processing.
- **Benefit**: Achieves real-time transcription entirely on CPU, removing the overhead of WebSocket server binaries.

#### Local Speaker Diarization with Persistence
- **How it works**: Identifies and separates different speakers in the audio stream in real-time. It extracts speaker embeddings and matches them against known profiles using cosine similarity.
- **Benefit**: Unlike traditional transient diarization, Voxium uses an **SQLite database** to persist speaker profiles across sessions, meaning the system "remembers" voices over time.

#### Dual-Pipeline Intent Routing
- **How it works**: Once audio is transcribed, Voxium dynamically routes the text based on intent.
  - **Agent Route**: If the voice agent is invoked, the text is routed to the local LLM for reasoning and action execution.
  - **Dictation/Cleanup Route**: If it's pure dictation, it bypasses the agent and can optionally undergo text cleanup/formatting.
- **Benefit**: Ensures low latency for simple dictation while reserving heavy LLM compute for actual agent interactions.

---

### 2. Intelligent Graph-RAG Memory Layer

Voxium features a localized memory engine that extracts relationships and entities from conversations to build a persistent knowledge graph.

#### Local Graph-RAG Engine
- **How it works**: Uses regex heuristics and standard Python libraries (no GPU embeddings required) to extract named entities, relationships, and dates from conversations. Entities are deterministically hashed (SHA-256).
- **Benefit**: The graph is ingested at the end of every turn and queried via Breadth-First Search (BFS) to inject relevant historical context directly into the LLM's prompt.

#### Topology-Based Clustering (Community Detection)
- **How it works**: Automatically groups related memories using Leiden or Louvain community detection algorithms based purely on edge-density (modularity optimization), without dense text embeddings.
- **Benefit**: Discovers hidden structures and topic clusters within the user's conversational history.

#### Deterministic AST Parsing
- **How it works**: Uses Python's built-in `ast` module and regex to map out code documents (classes, functions, headers) deterministically. 
- **Benefit**: Drastically reduces the LLM context window by allowing it to operate only on specific, localized node boundaries during code or document edits.

---

## 🔌 API & Integration

Voxium provides a rich REST and WebSocket API for seamless frontend integration.

- **WebSocket (`/audio_chunk`)**: Real-time audio streaming and live status updates.
- **REST APIs**:
  - `POST /api/transcribe`: Synchronous file transcription.
  - `GET /api/speakers`: Retrieve persistent speaker profiles.
  - `GET /api/status`: System and orchestrator health checks.
- **Graph API Endpoints**:
  - `GET /api/graph`: Returns the full memory graph in `node_link_data` format.
  - `GET /api/graph/clusters`: Community assignments and cohesion scores.
  - `GET /api/graph/search?q=...`: Fuzzy node search.
  - `GET /api/graph/node/<id>`: Node details and neighbors.
  - `GET /api/graph/stats`: Topology statistics.

---

## 🛠️ Setup & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Configure the `.env` file with your preferred engine and thresholds.
   ```env
   STT_ENGINE=whisper
   AGENT_NAME=Voxium
   # Adjust graph, VAD, and diarization thresholds as needed
   ```
3. **Model Placement**:
   Place your Whisper or Parakeet ONNX models inside the `models/stt/` directory.
4. **Run the Server**:
   ```bash
   python app.py
   ```
   *The server will start on `http://127.0.0.1:5000` by default.*
