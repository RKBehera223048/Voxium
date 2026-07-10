/* Voice Actions & Assistant Drawer Simulation */

window.Voxium = window.Voxium || {};

Voxium.voice = {
  isListening: false,
  isAssistantOpen: false,

  init: function() {
    this.bindEvents();
    this.restoreState();
  },

  bindEvents: function() {
    // Floating voice button pulse trigger
    const voiceFloatBtn = document.querySelector('.btn-voice-float');
    if (voiceFloatBtn) {
      voiceFloatBtn.addEventListener('click', () => this.toggleVoiceAssistant());
    }

    // Toggle always listening inside sidebar or settings
    const listenToggles = document.querySelectorAll('.listening-toggle');
    listenToggles.forEach(toggle => {
      toggle.addEventListener('change', (e) => {
        this.setAlwaysListening(e.target.checked);
      });
    });
  },

  restoreState: function() {
    // Set always listening active by default for mock display
    this.setAlwaysListening(true);
  },

  setAlwaysListening: function(active) {
    this.isListening = active;
    
    // Update toggle controls state
    const listenToggles = document.querySelectorAll('.listening-toggle');
    listenToggles.forEach(toggle => {
      toggle.checked = active;
    });

    // Update sidebar listening badge pulse ring
    const badges = document.querySelectorAll('.listening-badge');
    badges.forEach(badge => {
      if (active) {
        badge.classList.add('active');
      } else {
        badge.classList.remove('active');
      }
    });

    // Update status badge on landing page
    const statusText = document.getElementById('always-listening-status-text');
    if (statusText) {
      statusText.textContent = active ? 'Always-Listening Active' : 'Always-Listening Paused';
    }
  },

  // Toggle Voice Assistant Panel Overlay
  toggleVoiceAssistant: function() {
    if (this.isAssistantOpen) {
      this.closeAssistant();
    } else {
      this.openAssistant();
    }
  },

  openAssistant: function() {
    this.isAssistantOpen = true;
    
    // Create assistant overlay panel if not existing
    let drawer = document.getElementById('voice-assistant-drawer');
    if (!drawer) {
      drawer = document.createElement('div');
      drawer.id = 'voice-assistant-drawer';
      this.styleDrawer(drawer);
      document.body.appendChild(drawer);
    }

    // Set floating mic button state
    const floatBtn = document.querySelector('.btn-voice-float');
    if (floatBtn) {
      floatBtn.classList.add('recording');
      floatBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="24" height="24">
          <rect x="6" y="6" width="12" height="12" fill="currentColor" rx="2"/>
        </svg>
      `;
    }

    // Render contents inside the assistant
    drawer.innerHTML = `
      <div class="voice-drawer-header">
        <h4 class="headline-md" style="font-size: 1.15rem; color: var(--primary);">Voxium Assistant</h4>
        <button class="voice-drawer-close" style="background:none; border:none; color:var(--muted); cursor:pointer;">
          <svg viewBox="0 0 24 24" width="16" height="16"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/></svg>
        </button>
      </div>
      <div class="voice-drawer-body">
        <div class="voice-pulse-circle">
          <div class="pulse-ring ring-1"></div>
          <div class="pulse-ring ring-2"></div>
          <div class="pulse-mic">🎙️</div>
        </div>
        <p class="voice-transcription">Listening for your memory request...</p>
        <div class="voice-actions-box" style="display:none;">
          <p class="voice-response" style="font-family: var(--font-display); font-style: italic; color: var(--muted); margin-bottom:12px;"></p>
          <div class="voice-action-shortcuts">
            <button class="btn btn-sm btn-primary btn-action-execute">Apply Sync</button>
            <button class="btn btn-sm btn-secondary btn-action-dismiss">Dismiss</button>
          </div>
        </div>
      </div>
    `;

    // Bind close action
    drawer.querySelector('.voice-drawer-close').addEventListener('click', () => this.closeAssistant());

    // Trigger simulated transcriptions
    setTimeout(() => {
      drawer.querySelector('.voice-transcription').innerHTML = '“Who did I discuss vector indexes with yesterday?”';
    }, 1500);

    setTimeout(() => {
      drawer.querySelector('.voice-transcription').innerHTML = 'Analyzing semantic records...';
      const actionBox = drawer.querySelector('.voice-actions-box');
      const responseP = drawer.querySelector('.voice-response');
      responseP.textContent = 'Found conversation "Voxium Architecture Sync" with Speaker B (Dev) discussing SQLite compilation requirements.';
      actionBox.style.display = 'block';
    }, 3200);

    // Animate open
    setTimeout(() => {
      drawer.style.transform = 'translateY(0)';
      drawer.style.opacity = '1';
    }, 50);
  },

  closeAssistant: function() {
    this.isAssistantOpen = false;
    const drawer = document.getElementById('voice-assistant-drawer');
    if (drawer) {
      drawer.style.transform = 'translateY(30px)';
      drawer.style.opacity = '0';
      setTimeout(() => drawer.remove(), 250);
    }

    // Set floating mic button state back
    const floatBtn = document.querySelector('.btn-voice-float');
    if (floatBtn) {
      floatBtn.classList.remove('recording');
      floatBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20">
          <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.48 6-3.3 6-6.72h-1.7z" fill="currentColor"/>
        </svg>
      `;
    }
  },

  styleDrawer: function(el) {
    el.style.position = 'fixed';
    el.style.bottom = '96px';
    el.style.right = '24px';
    el.style.width = '340px';
    el.style.backgroundColor = '#ffffff';
    el.style.border = '1px solid var(--border)';
    el.style.borderRadius = '12px';
    el.style.boxShadow = 'var(--shadow-lg)';
    el.style.zIndex = '9998';
    el.style.padding = '16px';
    el.style.transform = 'translateY(30px)';
    el.style.opacity = '0';
    el.style.transition = 'all 0.25s cubic-bezier(0.16, 1, 0.3, 1)';
    el.style.display = 'flex';
    el.style.flexDirection = 'column';
    el.style.gap = '12px';

    // Inject styles for pulsed rings inside the voice assistant drawer
    if (!document.getElementById('voice-drawer-custom-styles')) {
      const styles = document.createElement('style');
      styles.id = 'voice-drawer-custom-styles';
      styles.innerHTML = `
        .voice-drawer-header { display: flex; justify-content: space-between; align-items: center; }
        .voice-pulse-circle { position: relative; width: 60px; height: 60px; border-radius: 50%; background-color: rgba(132, 15, 22, 0.08); display: flex; align-items: center; justify-content: center; margin: 16px auto; }
        .pulse-mic { font-size: 1.5rem; z-index: 2; }
        .pulse-ring { position: absolute; border: 1px solid var(--primary); border-radius: 50%; width: 100%; height: 100%; top: 0; left: 0; animation: drawer-mic-pulse 2s infinite ease-out; }
        .ring-2 { animation-delay: 1s; }
        @keyframes drawer-mic-pulse {
          0% { transform: scale(1); opacity: 0.8; }
          100% { transform: scale(1.8); opacity: 0; }
        }
        .voice-transcription { text-align: center; font-size: 0.85rem; color: var(--muted); font-weight: 500; min-height: 40px; margin-bottom: 8px; }
      `;
      document.head.appendChild(styles);
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.voice.init();
});
