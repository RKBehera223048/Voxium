/* Meeting Recording Screen Controller - Waveforms, Timers, Transcripts & Node Maps */

window.Voxium = window.Voxium || {};

Voxium.meeting = {
  isRecording: false,
  secondsElapsed: 0,
  timerInterval: null,
  waveformAnimationId: null,
  canvas: null,
  ctx: null,
  transcriptTimer: null,
  mockTranscriptIndex: 0,
  
  mockTranscriptsList: [
    { speaker: 'Speaker A', text: 'Let\'s run the benchmarks on Llama-3 8B. We need to verify response latencies client-side.', time: '0:12' },
    { speaker: 'Speaker B', text: 'The prefill speed is around 32 tokens/second. The generation speed is 18 tokens/second.', time: '0:28' },
    { speaker: 'Speaker C', text: 'Excellent, is this using Q4_K_M quantization?', time: '0:45' },
    { speaker: 'Speaker B', text: 'Yes, running locally. Memory footprint is around 4.8 GB which leaves plenty of space for ChromaDB.', time: '1:02' },
    { speaker: 'Speaker A', text: 'Perfect. We should add vector sync checkpoints as action items.', time: '1:20' }
  ],

  init: function() {
    this.canvas = document.getElementById('waveform-canvas');
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');
      this.resizeCanvas();
      window.addEventListener('resize', () => this.resizeCanvas());
      this.drawIdleWaveform();
    }

    this.bindEvents();
    this.renderSemanticGraph();
  },

  bindEvents: function() {
    const startBtn = document.getElementById('meeting-start-btn');
    const stopBtn = document.getElementById('meeting-stop-btn');
    const pauseBtn = document.getElementById('meeting-pause-btn');

    if (startBtn && stopBtn && pauseBtn) {
      startBtn.addEventListener('click', () => this.startMeeting());
      stopBtn.addEventListener('click', () => this.stopMeeting());
      pauseBtn.addEventListener('click', () => this.pauseMeeting());
    }

    // Action item toggle listener
    const actionsContainer = document.querySelector('.action-items-list');
    if (actionsContainer) {
      actionsContainer.addEventListener('click', (e) => {
        const row = e.target.closest('.action-item-row');
        if (row) {
          const checkbox = row.querySelector('.checkbox-input');
          // If click was on row, toggle checkbox
          if (e.target !== checkbox) {
            checkbox.checked = !checkbox.checked;
          }
          if (checkbox.checked) {
            row.classList.add('completed');
          } else {
            row.classList.remove('completed');
          }
        }
      });
    }
  },

  resizeCanvas: function() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.parentElement.clientWidth;
    this.canvas.height = this.canvas.parentElement.clientHeight;
  },

  startMeeting: function() {
    if (this.isRecording) return;
    this.isRecording = true;
    this.secondsElapsed = 0;
    this.mockTranscriptIndex = 0;

    // Toggle buttons display
    document.getElementById('meeting-start-btn').style.display = 'none';
    document.getElementById('meeting-pause-btn').style.display = 'inline-flex';
    document.getElementById('meeting-stop-btn').style.display = 'inline-flex';

    // Set offline status badge
    const badge = document.querySelector('.meeting-status-badge .status-pill');
    if (badge) {
      badge.className = 'status-pill status-pill-success';
      badge.innerHTML = '<span class="stat-indicator ready" style="background-color: var(--success);"></span>Recording Live';
    }

    // Start Recording Timer loop
    this.timerInterval = setInterval(() => {
      this.secondsElapsed++;
      this.updateTimerDisplay();
    }, 1000);

    // Start Audio Waveform canvas rendering
    this.drawActiveWaveform();

    // Start transcript appender simulator
    const transcriptBody = document.querySelector('.transcript-body');
    if (transcriptBody) {
      transcriptBody.innerHTML = `
        <div style="text-align: center; color: var(--muted); font-size: 0.8rem; padding: 12px 0;" id="transcript-waiting-indicator">
          Always-listening active. Waiting for transcription segment...
        </div>
      `;
    }
    this.startMockTranscriptSimulation();
    
    Voxium.utils.notify('Meeting recording initialized locally.');
  },

  pauseMeeting: function() {
    if (!this.isRecording) return;
    this.isRecording = false;
    clearInterval(this.timerInterval);
    clearInterval(this.transcriptTimer);
    cancelAnimationFrame(this.waveformAnimationId);
    this.drawIdleWaveform();

    // Toggle button display
    document.getElementById('meeting-pause-btn').style.display = 'none';
    document.getElementById('meeting-start-btn').textContent = 'Resume Session';
    document.getElementById('meeting-start-btn').style.display = 'inline-flex';

    const badge = document.querySelector('.meeting-status-badge .status-pill');
    if (badge) {
      badge.className = 'status-pill status-pill-warning';
      badge.innerHTML = '<span class="stat-indicator busy"></span>Recording Paused';
    }
  },

  stopMeeting: function() {
    this.isRecording = false;
    clearInterval(this.timerInterval);
    clearInterval(this.transcriptTimer);
    cancelAnimationFrame(this.waveformAnimationId);
    this.drawIdleWaveform();

    // Toggle button display back to original
    document.getElementById('meeting-pause-btn').style.display = 'none';
    document.getElementById('meeting-stop-btn').style.display = 'none';
    document.getElementById('meeting-start-btn').textContent = 'Start New Session';
    document.getElementById('meeting-start-btn').style.display = 'inline-flex';

    const badge = document.querySelector('.meeting-status-badge .status-pill');
    if (badge) {
      badge.className = 'status-pill status-pill-warning';
      badge.innerHTML = '<span class="stat-indicator"></span>Offline Standby';
    }

    this.secondsElapsed = 0;
    this.updateTimerDisplay();

    // Notify syncing
    Voxium.utils.notify('Session ended. Syncing transcripts & summaries into ChromaDB...');
  },

  updateTimerDisplay: function() {
    const mins = Math.floor(this.secondsElapsed / 60);
    const secs = this.secondsElapsed % 60;
    const pad = (val) => val.toString().padStart(2, '0');
    const timerText = document.getElementById('recording-timer');
    if (timerText) {
      timerText.textContent = `${pad(mins)}:${pad(secs)}`;
    }
  },

  drawIdleWaveform: function() {
    if (!this.ctx) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);
    
    // Draw fine flat centerline with slight organic jitter
    this.ctx.beginPath();
    this.ctx.strokeStyle = '#D8CDB2';
    this.ctx.lineWidth = 1;
    this.ctx.moveTo(0, h / 2);
    this.ctx.lineTo(w, h / 2);
    this.ctx.stroke();
  },

  drawActiveWaveform: function() {
    if (!this.ctx || !this.isRecording) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    
    const render = () => {
      if (!this.isRecording) return;
      this.ctx.clearRect(0, 0, w, h);
      
      this.ctx.beginPath();
      this.ctx.strokeStyle = '#840F16'; // Primary Red color
      this.ctx.lineWidth = 1.5;
      
      const sliceWidth = w / 150;
      let x = 0;
      
      this.ctx.moveTo(0, h / 2);
      for (let i = 0; i < 150; i++) {
        // Multi-sine wave combination to mimic audio signals
        const time = Date.now() * 0.004;
        const amplitude = 30 * Math.sin(i * 0.05 + time) * Math.cos(i * 0.01 + time * 0.5) * Math.sin(i * 0.005);
        const y = h / 2 + amplitude;
        
        if (i === 0) {
          this.ctx.moveTo(x, y);
        } else {
          this.ctx.lineTo(x, y);
        }
        x += sliceWidth;
      }
      this.ctx.stroke();
      
      // Draw secondary outline wave behind
      this.ctx.beginPath();
      this.ctx.strokeStyle = 'rgba(78, 96, 120, 0.4)'; // Muted blue accent
      this.ctx.lineWidth = 1;
      x = 0;
      this.ctx.moveTo(0, h / 2);
      for (let i = 0; i < 150; i++) {
        const time = Date.now() * 0.003;
        const amplitude = 20 * Math.sin(i * 0.04 + time + 2) * Math.cos(i * 0.02 + time * 0.3);
        const y = h / 2 + amplitude;
        if (i === 0) this.ctx.moveTo(x, y);
        else this.ctx.lineTo(x, y);
        x += sliceWidth;
      }
      this.ctx.stroke();
      
      this.waveformAnimationId = requestAnimationFrame(render);
    };
    
    render();
  },

  // Simulates Whisper speaker diarization text appending
  startMockTranscriptSimulation: function() {
    this.transcriptTimer = setInterval(() => {
      if (this.mockTranscriptIndex >= this.mockTranscriptsList.length) {
        clearInterval(this.transcriptTimer);
        return;
      }

      const item = this.mockTranscriptsList[this.mockTranscriptIndex];
      this.mockTranscriptIndex++;

      const waitingIndicator = document.getElementById('transcript-waiting-indicator');
      if (waitingIndicator) waitingIndicator.remove();

      const transcriptBody = document.querySelector('.transcript-body');
      if (transcriptBody) {
        const bubble = document.createElement('div');
        const initial = item.speaker.split(' ').pop().replace(/[()]/g, '')[0] || 'S';
        const speakerClass = item.speaker.includes('Speaker A') ? 'speaker-A' : (item.speaker.includes('Speaker B') ? 'speaker-B' : 'speaker-C');

        bubble.className = `speaker-bubble ${speakerClass}`;
        bubble.innerHTML = `
          <div class="speaker-avatar">${initial}</div>
          <div class="speaker-msg-box">
            <div class="speaker-name">
              <span>${item.speaker}</span>
              <span class="speaker-time">${item.time}</span>
            </div>
            <p class="speaker-text">${item.text}</p>
          </div>
        `;
        transcriptBody.appendChild(bubble);
        
        // Auto-scroll transcript container
        transcriptBody.scrollTop = transcriptBody.scrollHeight;

        // Add simulated Live AI summary action trigger
        this.addMockSummaryItem(item.text);
      }
    }, 4500);
  },

  // Append new summary bullet points dynamically based on spoken words
  addMockSummaryItem: function(text) {
    const list = document.querySelector('.ai-summary-list');
    if (!list) return;

    let summaryText = '';
    if (text.includes('benchmarks')) {
      summaryText = 'Benchmarking local model processing latency on Llama-3 8B.';
    } else if (text.includes('quantization')) {
      summaryText = 'Utilizing Q4_K_M weights to decrease local RAM footprint to 4.8 GB.';
    } else if (text.includes('checkpoints')) {
      summaryText = 'Action items populated for scheduling local vector database syncing.';
      
      // Auto-append checking box item in list
      const checkList = document.querySelector('.action-items-list');
      if (checkList) {
        const row = document.createElement('div');
        row.className = 'action-item-row';
        row.innerHTML = `
          <input type="checkbox" class="checkbox-input">
          <div class="action-item-text">Run script to sync local Whisper output to ChromaDB</div>
        `;
        checkList.appendChild(row);
      }
    }

    if (summaryText) {
      const li = document.createElement('li');
      li.textContent = summaryText;
      list.appendChild(li);
    }
  },

  // Renders a high-end SVG semantic memory map
  renderSemanticGraph: function() {
    const svg = document.getElementById('memory-graph-svg');
    if (!svg) return;

    const width = svg.clientWidth || 300;
    const height = svg.clientHeight || 200;

    // Define 6 semantic memory node coordinates
    const nodes = [
      { id: 1, label: 'Whisper CLI', x: width * 0.2, y: height * 0.3, r: 6 },
      { id: 2, label: 'SQLite db', x: width * 0.45, y: height * 0.15, r: 6 },
      { id: 3, label: 'ChromaDB', x: width * 0.75, y: height * 0.35, r: 6 },
      { id: 4, label: 'Llama LLM', x: width * 0.5, y: height * 0.6, r: 8 },
      { id: 5, label: 'Speech Assistant', x: width * 0.25, y: height * 0.8, r: 6 },
      { id: 6, label: 'Vector Indexes', x: width * 0.8, y: height * 0.75, r: 6 }
    ];

    // Define vector connection lines
    const links = [
      { source: 1, target: 4 },
      { source: 1, target: 2 },
      { source: 2, target: 3 },
      { source: 3, target: 4 },
      { source: 4, target: 5 },
      { source: 4, target: 6 },
      { source: 3, target: 6 }
    ];

    let html = '';

    // Draw paths links
    links.forEach(link => {
      const s = nodes.find(n => n.id === link.source);
      const t = nodes.find(n => n.id === link.target);
      if (s && t) {
        html += `<line x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" class="graph-link" />`;
      }
    });

    // Draw circles and text labels
    nodes.forEach(node => {
      const color = node.id === 4 ? 'var(--primary)' : 'var(--accent)';
      html += `
        <g>
          <!-- Radiating sun-motif dot circle (15% visual weight) -->
          <circle cx="${node.x}" cy="${node.y}" r="${node.r + 3}" fill="none" stroke="${color}" stroke-width="0.8" stroke-dasharray="1.5 1.5" />
          <circle cx="${node.x}" cy="${node.y}" r="${node.r}" fill="${color}" class="graph-node" />
          <text x="${node.x}" y="${node.y - 12}" text-anchor="middle" class="graph-label" style="font-weight: 600;">${node.label}</text>
        </g>
      `;
    });

    svg.innerHTML = html;
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.meeting.init();
});
