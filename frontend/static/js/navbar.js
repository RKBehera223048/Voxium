/* Navbar Search Command palette Overlay and Notification Center */

window.Voxium = window.Voxium || {};

Voxium.navbar = {
  searchActive: false,

  init: function() {
    this.bindEvents();
  },

  bindEvents: function() {
    const searchTrigger = document.querySelector('.nav-search-trigger');
    const searchModal = document.getElementById('search-modal');
    
    if (searchTrigger && searchModal) {
      searchTrigger.addEventListener('click', () => this.openSearch());
      
      // Close on modal overlay click
      searchModal.addEventListener('click', (e) => {
        if (e.target === searchModal) {
          this.closeSearch();
        }
      });
    }

    // Keyboard shortcut Cmd/Ctrl + K to open search modal
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        this.openSearch();
      }
    });

    // Handle search typing inputs
    const searchInput = document.getElementById('search-modal-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.executeSearch(e.target.value);
      });
    }
  },

  openSearch: function() {
    const searchModal = document.getElementById('search-modal');
    const searchInput = document.getElementById('search-modal-input');
    if (searchModal) {
      searchModal.classList.add('active');
      this.searchActive = true;
      if (searchInput) {
        setTimeout(() => searchInput.focus(), 50);
      }
    }
  },

  closeSearch: function() {
    const searchModal = document.getElementById('search-modal');
    const searchInput = document.getElementById('search-modal-input');
    if (searchModal) {
      searchModal.classList.remove('active');
      this.searchActive = false;
      if (searchInput) {
        searchInput.value = '';
      }
      this.clearResults();
    }
  },

  // Perform search across notes and return list of results
  executeSearch: function(query) {
    const resultsContainer = document.getElementById('search-results');
    if (!resultsContainer) return;

    if (!query.trim()) {
      this.clearResults();
      return;
    }

    const cleanQuery = query.toLowerCase();
    const matches = Voxium.db.notes.filter(note => 
      note.title.toLowerCase().includes(cleanQuery) || 
      note.content.toLowerCase().includes(cleanQuery)
    );

    if (matches.length === 0) {
      resultsContainer.innerHTML = `
        <div style="padding: 24px; text-align: center; color: var(--muted); font-size: 0.9rem;">
          No semantic memory nodes match "${Voxium.utils.escapeHTML(query)}"
        </div>
      `;
      return;
    }

    resultsContainer.innerHTML = matches.map(note => `
      <div class="search-result-item" data-note-id="${note.id}">
        <div class="result-title">${Voxium.utils.escapeHTML(note.title)}</div>
        <div class="result-snippet">${note.content.replace(/<[^>]*>/g, '').substring(0, 100)}...</div>
      </div>
    `).join('');

    // Bind click event to open search matches in the editor
    resultsContainer.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => {
        const noteId = item.getAttribute('data-note-id');
        this.closeSearch();
        window.location.href = `editor.html?id=${noteId}`;
      });
    });
  },

  clearResults: function() {
    const resultsContainer = document.getElementById('search-results');
    if (resultsContainer) {
      resultsContainer.innerHTML = `
        <div style="padding: 24px; text-align: center; color: var(--muted); font-size: 0.85rem;">
          Type to search semantic memories and transcript segments...
        </div>
      `;
    }
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.navbar.init();
});
