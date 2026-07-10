import pytest
import requests
import json
import os

BASE_URL = "http://127.0.0.1:5000"

def test_api_status():
    """Verify the orchestrator and server are healthy."""
    response = requests.get(f"{BASE_URL}/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data is not None

def test_transcribe_audio_file():
    """Test STT transcription pipeline (Happy Path)."""
    # Requires a sample test_audio.wav in the same directory
    wav_path = os.path.join(os.path.dirname(__file__), 'test_audio.wav')
    with open(wav_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(f"{BASE_URL}/api/transcribe?format=wav", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "processed_text" in data

def test_graph_memory_stats():
    """Verify Graph-RAG endpoints are reachable."""
    response = requests.get(f"{BASE_URL}/api/graph/stats")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_clipboard_tool():
    """Test local tool execution (Clipboard)."""
    payload = {"content": "Test automated clipboard write"}
    response = requests.post(f"{BASE_URL}/api/tools/clipboard", json=payload)
    assert response.status_code == 200
