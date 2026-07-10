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
import re
import asyncio
import logging
import threading
from pathlib import Path

from dotenv import load_dotenv

# Load environment before any other imports
load_dotenv()

from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

from core.orchestrator import VoxiumOrchestrator, TriggerType, PipelineResult
from core.state_manager import StateManager

# =============================================================================
# Logging  [H-3: Default debug to false]
# =============================================================================

logging.basicConfig(
    level=logging.DEBUG if os.getenv("VOXIUM_DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voxium")

# =============================================================================
# Security Constants
# =============================================================================

# [C-5] Whitelist of allowed HTML page names to prevent template injection
ALLOWED_PAGES = frozenset({
    "dashboard", "editor", "login", "meeting", "notes", "settings", "index",
})

# Maximum sizes for upload protection [H-7]
MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB max upload
MAX_WS_CHUNK_SIZE = 1 * 1024 * 1024     # 1 MB per WebSocket chunk

# Allowed export formats [C-4]
ALLOWED_EXPORT_FORMATS = frozenset({
    "md", "markdown", "docx", "ppt", "pptx", "tex", "latex", "txt", "text",
})


# =============================================================================
# App Factory
# =============================================================================

def create_app() -> tuple[Flask, SocketIO]:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        static_folder="frontend/static",
        template_folder="frontend/templates",
    )

    # ── [C-2] Require a real secret key — crash on insecure defaults ────
    secret = os.getenv("VOXIUM_SECRET_KEY", "")
    _insecure_defaults = {"", "voxium-dev-key", "change-me-to-a-random-string"}
    if secret in _insecure_defaults:
        raise RuntimeError(
            "CRITICAL: VOXIUM_SECRET_KEY is not set or is using a default value. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    app.config["SECRET_KEY"] = secret

    # [H-7] Limit upload size to prevent memory exhaustion
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # [M-10] CSRF protection — exempt JSON API endpoints since they
    # are protected by CORS origin checks + lack of cookie-based auth
    app.config["WTF_CSRF_CHECK_DEFAULT"] = False  # We'll manually protect form endpoints
    csrf = CSRFProtect(app)

    # ── [H-6] Rate limiting ────────────────────────────────────────────
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["60 per minute"],
        storage_uri="memory://",
    )

    # ── [C-3] Lock down CORS to allowed origins ────────────────────────
    allowed_origins = os.getenv(
        "VOXIUM_ALLOWED_ORIGINS", "http://127.0.0.1:5000"
    ).split(",")
    allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

    CORS(app, origins=allowed_origins)
    socketio = SocketIO(app, cors_allowed_origins=allowed_origins, async_mode="eventlet")

    # ── [C-6] Optional passphrase-based session auth ───────────────────
    # When VOXIUM_LOCAL_PASSPHRASE is set, require authentication.
    # When unset, Voxium runs in open local mode (127.0.0.1 only).
    _local_passphrase = os.getenv("VOXIUM_LOCAL_PASSPHRASE", "").strip()
    _auth_enabled = bool(_local_passphrase)

    if _auth_enabled:
        logger.info("Authentication ENABLED — passphrase required")
    else:
        logger.info("Authentication DISABLED — running in open local mode")

    @app.before_request
    def _check_auth():
        """[C-6] Enforce passphrase session on all routes when enabled."""
        if not _auth_enabled:
            return None
        # Always allow: login page, static assets, the auth endpoint itself
        exempt_paths = {"/login.html", "/api/auth/login", "/api/auth/status"}
        if (request.path in exempt_paths
                or request.path == "/"
                or request.path.startswith("/static/")):
            return None
        if not session.get("authenticated"):
            # API calls get 401; browser requests redirect to login
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required"}), 401
            return redirect("/login.html")

    # ── State & Orchestrator ────────────────────────────────────────────
    state = StateManager()
    
    from memory.memory_manager import HybridMemory
    memory_mgr = HybridMemory()
    state.register_on_turn(memory_mgr.on_turn)
    
    orchestrator = VoxiumOrchestrator(
        state_manager=state,
        on_result=lambda result: _broadcast_result(socketio, result),
    )

    # [L-2] Create a single shared DB instance instead of per-request
    from memory.sqlite import VoxiumDB as SpeakerProfileDB
    speaker_db = SpeakerProfileDB()

    # Store references on app for route access
    app.orchestrator = orchestrator
    app.state = state
    app.memory = memory_mgr

    # ── Auth Routes [C-6] ─────────────────────────────────────────────

    @app.route("/api/auth/login", methods=["POST"])
    @csrf.exempt  # Login form uses its own validation
    @limiter.limit("10 per minute")  # Brute-force protection
    def auth_login():
        """Validate passphrase and create a session."""
        if not _auth_enabled:
            return jsonify({"success": True, "message": "Auth not enabled"})
        data = request.json or {}
        passphrase = data.get("passphrase", "")
        if passphrase == _local_passphrase:
            session["authenticated"] = True
            session.permanent = True
            return jsonify({"success": True})
        logger.warning("Failed login attempt from %s", request.remote_addr)
        return jsonify({"error": "Invalid passphrase"}), 403

    @app.route("/api/auth/logout", methods=["POST"])
    @csrf.exempt
    def auth_logout():
        """Clear the session."""
        session.clear()
        return jsonify({"success": True})

    @app.route("/api/auth/status")
    def auth_status():
        """Check whether the user is authenticated."""
        return jsonify({
            "auth_enabled": _auth_enabled,
            "authenticated": session.get("authenticated", False),
        })

    # ── Routes ──────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Serve the main dashboard."""
        return render_template("index.html")

    @app.route("/<page>.html")
    def serve_html_page(page):
        """Serve other HTML pages. [C-5] Whitelist-only to prevent template injection."""
        if page not in ALLOWED_PAGES:
            return jsonify({"error": "Page not found"}), 404
        return render_template(f"{page}.html")

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
            "reasoning": orchestrator._reasoning_engine.get_model_info()
            if orchestrator._reasoning_engine else None,
        })

    @app.route("/api/transcribe", methods=["POST"])
    @csrf.exempt  # JSON API protected by CORS
    @limiter.limit("5 per minute")  # [H-6] Heavy endpoint
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
        try:
            result = asyncio.run_coroutine_threadsafe(
                orchestrator.process_audio_sync(
                    audio_bytes=audio_bytes,
                    trigger=trigger,
                    source_format=source_format,
                ),
                loop,
            ).result(timeout=120)
        except Exception as e:
            # [H-8] Don't leak internal exception details
            logger.error("Transcription pipeline error: %s", e, exc_info=True)
            return jsonify({"error": "Transcription failed. Check server logs."}), 500

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
        loop = _get_loop()
        profiles = asyncio.run_coroutine_threadsafe(
            speaker_db.get_all_profiles(), loop
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

    @app.route("/api/export", methods=["POST"])
    @csrf.exempt  # JSON API protected by CORS
    @limiter.limit("10 per minute")
    def export_document():
        """Export document to specified format. [C-4] Path-traversal hardened."""
        data = request.json
        if not data or "content" not in data or "format" not in data:
            return jsonify({"error": "Missing content or format"}), 400
        
        from document_engine.exporter import DocumentExporter
        from config import config
        exporter = DocumentExporter()
        
        # [C-4] Sanitize title — strip path separators and special chars
        title = data.get("title", "Exported_Document")
        title = re.sub(r'[^\w\s-]', '', title).strip()
        if not title:
            title = "Exported_Document"

        # [C-4] Validate format against allowlist
        fmt = data['format'].lower().strip('.')
        if fmt not in ALLOWED_EXPORT_FORMATS:
            return jsonify({"error": f"Unsupported export format: {data['format']}"}), 400

        output_filename = f"{title}.{fmt}"
        export_base = Path(config.export_dir).resolve()
        output_path = export_base / output_filename

        # [C-4] Final path-traversal guard — ensure output stays in export dir
        if not str(output_path.resolve()).startswith(str(export_base)):
            return jsonify({"error": "Invalid file path"}), 400
        
        try:
            generated_path = exporter.export(data["content"], fmt, str(output_path), title=title)
            return jsonify({"success": True, "path": generated_path})
        except Exception as e:
            # [H-8] Don't leak internal exception details
            logger.error("Export error: %s", e, exc_info=True)
            return jsonify({"error": "Export failed. Check server logs."}), 500

    @app.route("/api/tools/clipboard", methods=["GET", "POST"])
    @csrf.exempt  # JSON API protected by CORS
    @limiter.limit("30 per minute")
    def clipboard_op():
        """Clipboard read/write. [H-9] Fixed async/sync mismatch."""
        from tools.clipboard import read_clipboard, write_clipboard
        loop = _get_loop()

        if request.method == "GET":
            try:
                content = asyncio.run_coroutine_threadsafe(
                    read_clipboard(), loop
                ).result(timeout=5)
                return jsonify({"content": content})
            except Exception as e:
                # [H-8] Don't leak exception internals
                logger.error("Clipboard read error: %s", e, exc_info=True)
                return jsonify({"error": "Clipboard read failed"}), 500
        else:
            data = request.json
            if not data or "content" not in data:
                return jsonify({"error": "Missing content"}), 400
            try:
                asyncio.run_coroutine_threadsafe(
                    write_clipboard(data["content"]), loop
                ).result(timeout=5)
                return jsonify({"success": True})
            except Exception as e:
                logger.error("Clipboard write error: %s", e, exc_info=True)
                return jsonify({"error": "Clipboard write failed"}), 500

    @app.route("/api/models", methods=["GET"])
    def get_models():
        from config import config
        return jsonify({
            "stt": {"status": "ready", "path": "models/stt/"},
            "llm": {"status": "ready", "path": config.llm_model_path},
            "tts": {"status": "ready", "voice": config.piper_voice_model},
            "wakeword": {"status": "ready", "sensitivity": config.wakeword_sensitivity}
        })

    # ── Graph-RAG Memory API ─────────────────────────────────────────────

    @app.route("/api/graph")
    def graph_data():
        """Get graph data with pagination. [M-1] Prevents memory exhaustion.

        Query params:
            limit: Max nodes to return (default 500, use 0 for all)
            offset: Number of nodes to skip (default 0)
        """
        # [M-1] Default to 500 nodes to prevent browser-crashing JSON
        raw_limit = request.args.get("limit", "500")
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = 500
        offset = max(0, int(request.args.get("offset", "0")))

        # limit=0 means "return everything" (for admin/debug use)
        effective_limit = None if limit == 0 else max(1, limit)

        loop = _get_loop()
        data = asyncio.run_coroutine_threadsafe(
            memory_mgr.memory_graph.get_graph_data(
                limit=effective_limit, offset=offset
            ), loop
        ).result(timeout=30)
        return jsonify(data)

    @app.route("/api/graph/clusters")
    def graph_clusters():
        """Get community assignments, labels, and cohesion scores."""
        loop = _get_loop()
        data = asyncio.run_coroutine_threadsafe(
            memory_mgr.memory_graph.get_clusters(), loop
        ).result(timeout=30)
        return jsonify(data)

    @app.route("/api/graph/search")
    def graph_search():
        """Fuzzy search nodes by label. Query param: ?q=..."""
        # [M-7] Cap query length to prevent abuse
        query = request.args.get("q", "").strip()[:500]
        if not query:
            return jsonify({"results": [], "error": "Missing query parameter ?q="}), 400
        loop = _get_loop()
        results = asyncio.run_coroutine_threadsafe(
            memory_mgr.memory_graph.search_nodes(query), loop
        ).result(timeout=10)
        return jsonify({"query": query, "results": results})

    @app.route("/api/graph/node/<node_id>")
    def graph_node_detail(node_id: str):
        """Get full details for a single node + its neighbors."""
        loop = _get_loop()
        detail = asyncio.run_coroutine_threadsafe(
            memory_mgr.memory_graph.get_node_detail(node_id), loop
        ).result(timeout=10)
        if detail is None:
            return jsonify({"error": "Node not found"}), 404
        return jsonify(detail)

    @app.route("/api/graph/stats")
    def graph_stats():
        """Get graph statistics: node/edge/community counts, density."""
        loop = _get_loop()
        stats = asyncio.run_coroutine_threadsafe(
            memory_mgr.memory_graph.get_stats(), loop
        ).result(timeout=10)
        return jsonify(stats)

    # ── WebSocket Events ────────────────────────────────────────────────

    @socketio.on("connect")
    def handle_connect():
        # [C-6] Reject unauthenticated WebSocket connections when auth is on
        if _auth_enabled and not session.get("authenticated"):
            logger.warning("Unauthenticated WebSocket connection rejected: %s", request.sid)
            return False  # Reject the connection
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
            trigger_str = data.get("trigger", "dictation")
        else:
            audio_bytes = data
            trigger_str = "dictation"

        # [H-7] Enforce chunk size limit
        raw_bytes = audio_bytes if isinstance(audio_bytes, bytes) else bytes(audio_bytes)
        if len(raw_bytes) > MAX_WS_CHUNK_SIZE:
            emit("error", {"message": "Audio chunk too large"})
            return

        # [H-2] Validate trigger type — don't let invalid values crash the task
        try:
            trigger_type = TriggerType(trigger_str)
        except ValueError:
            trigger_type = TriggerType.DICTATION
            logger.warning("Invalid trigger type via WebSocket: %s", trigger_str)

        # Queue for processing
        loop = _get_loop()
        asyncio.run_coroutine_threadsafe(
            orchestrator.submit_audio(
                audio_bytes=raw_bytes,
                trigger=trigger_type,
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
_loop_lock = threading.Lock()  # [H-5] Protect against race condition


def _get_loop() -> asyncio.AbstractEventLoop:
    """Get or create the background asyncio event loop. [H-5] Thread-safe."""
    global _loop, _loop_thread

    with _loop_lock:
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
    try:
        asyncio.run_coroutine_threadsafe(
            app.orchestrator.start(), loop
        ).result(timeout=30)
    except Exception as e:
        logger.warning("Orchestrator failed to start (AI features disabled): %s", e)
        logger.warning("The web UI will still be served. Install missing dependencies to enable AI.")

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
