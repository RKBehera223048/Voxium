/* Voxium Core App Initializer */

window.Voxium = window.Voxium || {};

Voxium.app = {
  stats: {
    whisper: { active: true, model: 'Whisper Base (English)', status: 'Ready' },
    llm: { active: true, model: 'Llama-3-8B-Q4', status: 'Ready' },
    chroma: { active: true, documentsCount: 142, lastSync: '2026-06-25T09:12:00Z' }
  },

  init: function() {
    this.updateGlobalBadgeStats();
    this.bindGlobalKeys();
    this.simulateDBSync();
    this.initNotesPage();
    this.initSettingsPage();
  },

  // Update top bar stat badges across pages dynamically
  updateGlobalBadgeStats: function() {
    const whisperDot = document.getElementById('whisper-status-dot');
    const whisperLabel = document.getElementById('whisper-status-label');
    if (whisperDot && whisperLabel) {
      whisperDot.className = 'stat-indicator ready';
      whisperLabel.textContent = `Whisper: ${this.stats.whisper.status}`;
    }

    const llmDot = document.getElementById('llm-status-dot');
    const llmLabel = document.getElementById('llm-status-label');
    if (llmDot && llmLabel) {
      llmDot.className = 'stat-indicator ready';
      llmLabel.textContent = `Local LLM: ${this.stats.llm.status}`;
    }

    const dbTotalNotes = document.getElementById('db-total-notes-card');
    if (dbTotalNotes) {
      dbTotalNotes.textContent = this.stats.chroma.documentsCount;
    }
  },

  bindGlobalKeys: function() {
    // Escape key closes modals and panels
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        // Close search overlay
        if (Voxium.navbar && typeof Voxium.navbar.closeSearch === 'function') {
          Voxium.navbar.closeSearch();
        }
        // Close voice assistant
        if (Voxium.voice && typeof Voxium.voice.closeAssistant === 'function') {
          Voxium.voice.closeAssistant();
        }
      }
    });
  },

  simulateDBSync: function() {
    // Increment DB document index count occasionally to show local sync flow activity
    setInterval(() => {
      if (Math.random() > 0.85) {
        this.stats.chroma.documentsCount++;
        this.stats.chroma.lastSync = new Date().toISOString();
        this.updateGlobalBadgeStats();
        
        // Show silent notification occasionally
        if (Math.random() > 0.8) {
          Voxium.utils.notify('Memory Graph updated: indexed 1 new vector node.');
        }
      }
    }, 15000);
  },

  initNotesPage: function() {
    const container = document.getElementById('notes-container');
    if (!container) return;

    const searchInput = document.getElementById('notes-search-input');
    const filterPills = document.getElementById('notes-filter-pills');
    const gridBtn = document.getElementById('view-grid-btn');
    const listBtn = document.getElementById('view-list-btn');

    let currentFilter = 'all';
    let searchQuery = '';

    const render = () => {
      let notes = Voxium.db.notes;

      // Filter by tag pill
      if (currentFilter !== 'all') {
        notes = notes.filter(n => n.tags.includes(currentFilter) || (currentFilter === 'journal' && n.tags.includes('journal')));
      }

      // Filter by text search
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        notes = notes.filter(n => 
          n.title.toLowerCase().includes(query) || 
          n.content.toLowerCase().includes(query) ||
          n.tags.some(t => t.toLowerCase().includes(query))
        );
      }

      const emptyState = document.getElementById('notes-empty-state');
      if (notes.length === 0) {
        container.style.display = 'none';
        if (emptyState) emptyState.style.display = 'flex';
        return;
      }

      container.style.display = container.classList.contains('notes-grid') ? 'grid' : 'flex';
      if (emptyState) emptyState.style.display = 'none';

      container.innerHTML = notes.map(note => {
        const textSnippet = note.content.replace(/<[^>]*>/g, '').substring(0, 140);
        const tagsHTML = note.tags.map(t => `<span class="tag-label">${t}</span>`).join('');
        
        return `
          <div class="card card-hover note-item-card" onclick="window.location.href='editor.html?id=${note.id}'">
            <div class="card-header">
              <span class="card-title">${Voxium.utils.escapeHTML(note.title)}</span>
              <span class="card-subtitle">${Voxium.utils.formatDate(note.modified)}</span>
            </div>
            <div class="card-body">
              <p>${textSnippet}...</p>
              <div class="note-tags-row">
                ${tagsHTML}
              </div>
            </div>
          </div>
        `;
      }).join('');
    };

    // Toggle views
    if (gridBtn && listBtn) {
      gridBtn.addEventListener('click', () => {
        gridBtn.classList.add('active');
        listBtn.classList.remove('active');
        container.className = 'notes-grid';
        render();
      });
      listBtn.addEventListener('click', () => {
        listBtn.classList.add('active');
        gridBtn.classList.remove('active');
        container.className = 'notes-list';
        render();
      });
    }

    // Filter pills click
    if (filterPills) {
      filterPills.addEventListener('click', (e) => {
        const pill = e.target.closest('.filter-pill');
        if (!pill) return;

        filterPills.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');

        currentFilter = pill.getAttribute('data-filter');
        render();
      });
    }

    // Search typing
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        render();
      });
    }

    render();
  },

  initSettingsPage: function() {
    const menu = document.querySelector('.settings-menu');
    if (!menu) return;

    menu.addEventListener('click', (e) => {
      const item = e.target.closest('.settings-menu-item');
      if (!item) return;

      // Reset active tabs
      menu.querySelectorAll('.settings-menu-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      // Toggle panes
      const paneId = item.getAttribute('data-pane-id');
      document.querySelectorAll('.settings-pane').forEach(pane => {
        if (pane.id === paneId) {
          pane.classList.add('active');
        } else {
          pane.classList.remove('active');
        }
      });
    });
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.app.init();
});
