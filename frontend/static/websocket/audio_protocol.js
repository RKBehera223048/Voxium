/**
 * Voxium — AudioWorklet Processor + SocketIO Audio Manager
 * ==========================================================
 * Real-time audio capture from browser microphone via AudioWorklet API,
 * streaming to the Flask-SocketIO backend.
 *
 * Architecture:
 *   Browser Mic → AudioWorklet (capture PCM) → SocketIO binary → Backend
 *   Backend TTS → SocketIO binary → Web Audio API → Speaker
 *
 * This replaces the mock voice.js with actual audio streaming.
 */

// ============================================================================
// AudioWorklet Processor (runs in audio thread)
// ============================================================================

/**
 * Custom AudioWorklet processor that captures raw PCM audio data
 * and sends it to the main thread via MessagePort.
 *
 * Registered as 'voxium-audio-processor' — must be loaded via:
 *   audioContext.audioWorklet.addModule('/static/js/audio_protocol.js')
 *
 * NOTE: This file serves dual purpose — the AudioWorkletProcessor class
 * runs in the Worklet scope, while VoxiumAudioManager runs in the main thread.
 * Only the processor class uses registerProcessor().
 */
if (typeof AudioWorkletProcessor !== 'undefined') {
  class VoxiumAudioProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this._bufferSize = 4096; // ~93ms at 44.1kHz
      this._buffer = new Float32Array(this._bufferSize);
      this._bufferIndex = 0;
      this._isCapturing = true;
    }

    process(inputs, outputs, parameters) {
      if (!this._isCapturing) return true;

      const input = inputs[0];
      if (!input || !input[0]) return true;

      const channelData = input[0]; // Mono channel

      // Accumulate samples into buffer
      for (let i = 0; i < channelData.length; i++) {
        this._buffer[this._bufferIndex++] = channelData[i];

        if (this._bufferIndex >= this._bufferSize) {
          // Send buffer to main thread
          this.port.postMessage({
            type: 'audio',
            buffer: this._buffer.slice(), // Copy
          });
          this._bufferIndex = 0;
        }
      }

      return true; // Keep processor alive
    }
  }

  registerProcessor('voxium-audio-processor', VoxiumAudioProcessor);
}


// ============================================================================
// VoxiumAudioManager (runs in main thread)
// ============================================================================

/**
 * Main-thread audio manager that coordinates:
 *   1. Mic capture via AudioWorklet
 *   2. Sending audio chunks to backend via SocketIO
 *   3. Receiving and playing TTS audio from backend
 *   4. Wake word / VAD status feedback
 */
class VoxiumAudioManager {
  constructor(socketIO) {
    this.socket = socketIO;
    this.audioContext = null;
    this.workletNode = null;
    this.mediaStream = null;
    this.isCapturing = false;
    this.isAlwaysListening = false;

    // TTS playback
    this.ttsQueue = [];
    this.isPlayingTTS = false;

    // Status callbacks
    this.onStatusChange = null;
    this.onTranscription = null;
    this.onWakeWord = null;
    this.onTTSStart = null;
    this.onTTSEnd = null;

    this._setupSocketEvents();
  }

  // ── Mic Capture ─────────────────────────────────────────────────────

  /**
   * Start capturing audio from the microphone.
   * Must be called from a user gesture (click/tap) due to browser policy.
   */
  async startCapture(trigger = 'dictation') {
    if (this.isCapturing) return;

    try {
      // Request mic access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 44100,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create AudioContext (resume if suspended due to autoplay policy)
      this.audioContext = new AudioContext({ sampleRate: 44100 });
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Load AudioWorklet processor
      await this.audioContext.audioWorklet.addModule(
        '/static/websocket/audio_protocol.js'
      );

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'voxium-audio-processor'
      );

      // Handle audio chunks from worklet
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audio') {
          this._sendAudioChunk(event.data.buffer, trigger);
        }
      };

      // Connect mic → worklet
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      source.connect(this.workletNode);
      // Don't connect worklet to destination (we don't want to hear ourselves)

      this.isCapturing = true;
      this.socket.emit('recording_start', { trigger });

      if (this.onStatusChange) {
        this.onStatusChange('capturing', trigger);
      }

      console.log('[Voxium] Audio capture started (trigger:', trigger, ')');

    } catch (err) {
      console.error('[Voxium] Failed to start audio capture:', err);
      if (this.onStatusChange) {
        this.onStatusChange('error', err.message);
      }
    }
  }

  /**
   * Stop capturing audio.
   */
  async stopCapture() {
    if (!this.isCapturing) return;

    // Disconnect worklet
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    // Stop media stream
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    // Close audio context
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    this.isCapturing = false;
    this.socket.emit('recording_stop', {});

    if (this.onStatusChange) {
      this.onStatusChange('idle');
    }

    console.log('[Voxium] Audio capture stopped');
  }

  /**
   * Toggle always-listening mode.
   * In this mode, audio is continuously streamed and the backend uses
   * VAD + wake word to decide when to process.
   */
  async toggleAlwaysListening() {
    if (this.isAlwaysListening) {
      await this.stopCapture();
      this.isAlwaysListening = false;
    } else {
      await this.startCapture('dictation');
      this.isAlwaysListening = true;
    }
    return this.isAlwaysListening;
  }

  // ── Audio Sending ─────────────────────────────────────────────────

  _sendAudioChunk(buffer, trigger) {
    if (!this.socket || !this.socket.connected) return;

    // Send Float32 PCM as binary via SocketIO
    this.socket.emit('audio_chunk', {
      audio: buffer.buffer, // ArrayBuffer
      trigger: trigger,
      sample_rate: 44100,
      timestamp: Date.now(),
    });
  }

  // ── TTS Playback ──────────────────────────────────────────────────

  /**
   * Play TTS audio received from the backend.
   * Queues audio chunks for sequential playback.
   */
  async playTTSAudio(audioData) {
    this.ttsQueue.push(audioData);

    if (!this.isPlayingTTS) {
      await this._processTTSQueue();
    }
  }

  async _processTTSQueue() {
    this.isPlayingTTS = true;

    if (this.onTTSStart) this.onTTSStart();

    while (this.ttsQueue.length > 0) {
      const audioData = this.ttsQueue.shift();

      try {
        // Create a temporary AudioContext for playback
        const playbackCtx = new AudioContext();

        // Decode WAV audio
        const audioBuffer = await playbackCtx.decodeAudioData(audioData);
        const source = playbackCtx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackCtx.destination);

        // Play and wait for completion
        await new Promise((resolve) => {
          source.onended = () => {
            playbackCtx.close();
            resolve();
          };
          source.start(0);
        });

      } catch (err) {
        console.error('[Voxium] TTS playback error:', err);
      }
    }

    this.isPlayingTTS = false;
    if (this.onTTSEnd) this.onTTSEnd();
  }

  /**
   * Stop TTS playback immediately (e.g., user interruption).
   */
  stopTTS() {
    this.ttsQueue = [];
    this.isPlayingTTS = false;
    if (this.onTTSEnd) this.onTTSEnd();
  }

  // ── SocketIO Event Handlers ───────────────────────────────────────

  _setupSocketEvents() {
    // Connection status
    this.socket.on('connect', () => {
      console.log('[Voxium] Connected to server');
      if (this.onStatusChange) this.onStatusChange('connected');
    });

    this.socket.on('disconnect', () => {
      console.log('[Voxium] Disconnected from server');
      if (this.onStatusChange) this.onStatusChange('disconnected');
    });

    // Transcription results
    this.socket.on('transcription_result', (data) => {
      console.log('[Voxium] Transcription:', data);
      if (this.onTranscription) this.onTranscription(data);
    });

    // TTS audio chunks (binary)
    this.socket.on('tts_audio', (data) => {
      if (data && data.audio) {
        this.playTTSAudio(data.audio);
      }
    });

    // Wake word detected
    this.socket.on('wake_word_detected', (data) => {
      console.log('[Voxium] Wake word detected:', data);
      if (this.onWakeWord) this.onWakeWord(data);
    });

    // Processing status
    this.socket.on('processing', (data) => {
      if (this.onStatusChange) this.onStatusChange('processing', data);
    });

    // Pipeline status updates
    this.socket.on('status', (data) => {
      if (this.onStatusChange) this.onStatusChange('status', data);
    });

    // Error handling
    this.socket.on('error', (data) => {
      console.error('[Voxium] Server error:', data);
      if (this.onStatusChange) this.onStatusChange('error', data);
    });
  }

  // ── Cleanup ───────────────────────────────────────────────────────

  async destroy() {
    await this.stopCapture();
    this.stopTTS();
    this.socket.off('transcription_result');
    this.socket.off('tts_audio');
    this.socket.off('wake_word_detected');
    this.socket.off('processing');
    this.socket.off('status');
  }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
  window.VoxiumAudioManager = VoxiumAudioManager;
}
