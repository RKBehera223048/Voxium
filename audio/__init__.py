# Voxium Audio — VAD, wake word, transcription, TTS, diarization, recording
"""
Audio processing module for the Voxium voice assistant.

Sub-modules:
    - vad: Voice Activity Detection (Silero VAD + RMS gate)
    - wakeword: OpenWakeWord-based wake word detection
    - whisper: Speech-to-text (faster-whisper + sherpa-onnx/Parakeet)
    - tts: Text-to-speech (Piper TTS)
    - diarization: Speaker diarization (pyannote.audio)
    - recorder: Server-side audio buffer management
    - audio_queue: Thread-safe audio chunk queue
"""
