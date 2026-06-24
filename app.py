"""
Voxium — Flask Application Entry Point
=========================================
Main server that ties together the orchestrator, VAD pipeline,
transcription engines, reasoning, and diarization into a unified
REST + WebSocket API.

Architecture:
    - Flask serves the web UI and REST endpoints
    - Flask-SocketIO handles real-time audio streaming and status updates
    - The VoxiumOrchestrator runs in a background asyncio event loop
    - All AI processing remains 100% local — no external API calls
"""

from __future__ import annotations

import os
import asyncio
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv

# Load environment before any other imports
load_dotenv()

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from core.orchestrator import VoxiumOrchestrator, TriggerType, PipelineResult
from core.state_manager import StateManager

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if os.getenv("VOXIUM_DEBUG", "true").lower() == "true" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voxium")


# =============================================================================
# App Factory
# =============================================================================

def create_app() -> tuple[Flask, SocketIO]:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["SECRET_KEY"] = os.getenv("VOXIUM_SECRET_KEY", "voxium-dev-key")

    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

    # ── State & Orchestrator ────────────────────────────────────────────
    state = StateManager()
    orchestrator = VoxiumOrchestrator(
        state_manager=state,
        on_result=lambda result: _broadcast_result(socketio, result),
    )

    # Store references on app for route access
    app.orchestrator = orchestrator
    app.state = state

    # ── Routes ──────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Serve the main dashboard."""
        return render_template("index.html")

    @app.route("/api/status")
    def status():
        """Get system status."""
        loop = _get_loop()
        state_data = asyncio.run_coroutine_threadsafe(
            state.get_state(), loop
        ).result(timeout=5)

        return jsonify({
            "status": "running",
            "orchestrator": state_data,
            "stt_engine": orchestrator._stt_engine.get_model_info()
            if orchestrator._stt_engine else None,
            "reasoning": orchestrator._reasoning.get_model_info()
            if orchestrator._reasoning else None,
        })

    @app.route("/api/transcribe", methods=["POST"])
    def transcribe():
        """
        Transcribe audio via REST API.

        Accepts audio file upload and runs the full pipeline:
            VAD → Transcription → Route → Cleanup/Agent

        Query params:
            trigger: "dictation" | "voice_agent" (default: "dictation")
            format: audio format hint (default: "webm")
        """
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        audio_bytes = audio_file.read()

        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio file"}), 400

        # Parse trigger type
        trigger_str = request.args.get("trigger", "dictation")
        try:
            trigger = TriggerType(trigger_str)
        except ValueError:
            trigger = TriggerType.DICTATION

        source_format = request.args.get("format", "webm")

        # Run pipeline synchronously
        loop = _get_loop()
        result = asyncio.run_coroutine_threadsafe(
            orchestrator.process_audio_sync(
                audio_bytes=audio_bytes,
                trigger=trigger,
                source_format=source_format,
            ),
            loop,
        ).result(timeout=120)

        return jsonify({
            "success": result.success,
            "route": result.route.value,
            "raw_text": result.raw_text,
            "processed_text": result.processed_text,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "error": result.error,
        })

    @app.route("/api/speakers", methods=["GET"])
    def get_speakers():
        """Get all stored speaker profiles."""
        from data.db.speaker_db import SpeakerProfileDB

        loop = _get_loop()
        db = SpeakerProfileDB()
        profiles = asyncio.run_coroutine_threadsafe(
            db.get_all_profiles(), loop
        ).result(timeout=10)

        return jsonify({
            "speakers": [
                {
                    "id": p.id,
                    "display_name": p.display_name,
                    "embedding_count": p.embedding_count,
                    "created_at": p.created_at,
                    "last_seen_at": p.last_seen_at,
                }
                for p in profiles
            ],
        })

    @app.route("/api/config", methods=["GET"])
    def get_config():
        """Get current configuration."""
        return jsonify({
            "agent_name": os.getenv("AGENT_NAME", "Voxium"),
            "stt_engine": os.getenv("STT_ENGINE", "whisper"),
            "language": os.getenv("PREFERRED_LANGUAGE", "en"),
            "agent_enabled": os.getenv("AGENT_ENABLED", "true").lower() == "true",
            "cleanup_enabled": os.getenv("CLEANUP_ENABLED", "true").lower() == "true",
            "diarization_enabled": os.getenv("DIARIZATION_ENABLED", "false").lower() == "true",
        })

    # ── Graph-RAG Memory API ─────────────────────────────────────────────

    @app.route("/api/graph")
    def graph_data():
        """Get full graph in node_link_data format (matches graphify)."""
        loop = _get_loop()
        data = asyncio.run_coroutine_threadsafe(
            state.memory_graph.get_graph_data(), loop
        ).result(timeout=30)
        return jsonify(data)

    @app.route("/api/graph/clusters")
    def graph_clusters():
        """Get community assignments, labels, and cohesion scores."""
        loop = _get_loop()
        data = asyncio.run_coroutine_threadsafe(
            state.memory_graph.get_clusters(), loop
        ).result(timeout=30)
        return jsonify(data)

    @app.route("/api/graph/search")
    def graph_search():
        """Fuzzy search nodes by label. Query param: ?q=..."""
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"results": [], "error": "Missing query parameter ?q="}), 400
        loop = _get_loop()
        results = asyncio.run_coroutine_threadsafe(
            state.memory_graph.search_nodes(query), loop
        ).result(timeout=10)
        return jsonify({"query": query, "results": results})

    @app.route("/api/graph/node/<node_id>")
    def graph_node_detail(node_id: str):
        """Get full details for a single node + its neighbors."""
        loop = _get_loop()
        detail = asyncio.run_coroutine_threadsafe(
            state.memory_graph.get_node_detail(node_id), loop
        ).result(timeout=10)
        if detail is None:
            return jsonify({"error": f"Node '{node_id}' not found"}), 404
        return jsonify(detail)

    @app.route("/api/graph/stats")
    def graph_stats():
        """Get graph statistics: node/edge/community counts, density."""
        loop = _get_loop()
        stats = asyncio.run_coroutine_threadsafe(
            state.memory_graph.get_stats(), loop
        ).result(timeout=10)
        return jsonify(stats)

    # ── WebSocket Events ────────────────────────────────────────────────

    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected: %s", request.sid)
        emit("status", {"connected": True})

    @socketio.on("disconnect")
    def handle_disconnect():
        logger.info("Client disconnected: %s", request.sid)

    @socketio.on("audio_chunk")
    def handle_audio_chunk(data):
        """
        Receive audio chunks via WebSocket for real-time streaming.

        The frontend sends audio chunks as they're recorded. When the
        recording stops, the frontend sends a 'recording_stop' event.
        """
        if isinstance(data, dict):
            audio_bytes = data.get("audio", b"")
            trigger = data.get("trigger", "dictation")
        else:
            audio_bytes = data
            trigger = "dictation"

        # Queue for processing
        loop = _get_loop()
        asyncio.run_coroutine_threadsafe(
            orchestrator.submit_audio(
                audio_bytes=audio_bytes if isinstance(audio_bytes, bytes) else bytes(audio_bytes),
                trigger=TriggerType(trigger),
            ),
            loop,
        )

    @socketio.on("recording_stop")
    def handle_recording_stop(data=None):
        """Handle recording stop — triggers final processing."""
        emit("processing", {"status": "processing"})

    return app, socketio


# =============================================================================
# Helpers
# =============================================================================

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create the background asyncio event loop."""
    global _loop, _loop_thread

    if _loop is not None and _loop.is_running():
        return _loop

    _loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(_loop)
        _loop.run_forever()

    _loop_thread = threading.Thread(target=run_loop, daemon=True)
    _loop_thread.start()

    return _loop


def _broadcast_result(socketio: SocketIO, result: PipelineResult) -> None:
    """Broadcast a pipeline result to all connected clients via WebSocket."""
    socketio.emit("transcription_result", {
        "success": result.success,
        "route": result.route.value,
        "raw_text": result.raw_text,
        "processed_text": result.processed_text,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "error": result.error,
    })


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Start the Voxium server."""
    app, socketio = create_app()

    # Start the orchestrator in the background loop
    loop = _get_loop()
    asyncio.run_coroutine_threadsafe(
        app.orchestrator.start(), loop
    ).result(timeout=30)

    host = os.getenv("VOXIUM_HOST", "127.0.0.1")
    port = int(os.getenv("VOXIUM_PORT", "5000"))

    logger.info("=" * 60)
    logger.info("  Voxium Voice Assistant")
    logger.info("  http://%s:%d", host, port)
    logger.info("  Engine: %s", os.getenv("STT_ENGINE", "whisper"))
    logger.info("  Agent: %s", os.getenv("AGENT_NAME", "Voxium"))
    logger.info("=" * 60)

    socketio.run(app, host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
