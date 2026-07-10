/* Document Editor Workspace Controller - Folders, Rich Text Actions, and AI Panels */

window.Voxium = window.Voxium || {};

Voxium.editor = {
  activeNoteId: null,
  activeTab: 'summary', // summary, ask, search

  init: function() {
    this.bindEvents();
    this.renderFolderTree();
    this.loadNoteFromUrl();
  },

  bindEvents: function() {
    // Toolbar edit formatting simulation
    const toolbar = document.querySelector('.editor-toolbar');
    if (toolbar) {
      toolbar.addEventListener('click', (e) => {
        const btn = e.target.closest('.toolbar-btn');
        if (!btn) return;
        
        const action = btn.getAttribute('data-action');
        if (action === 'bold' || action === 'italic') {
          document.execCommand(action, false, null);
          btn.classList.toggle('active');
        } else if (action === 'h1' || action === 'h2') {
          const tag = btn.classList.contains('active') ? 'p' : action;
          document.execCommand('formatBlock', false, `<${tag}>`);
          btn.classList.toggle('active');
        } else if (action === 'ul' || action === 'ol') {
          const cmd = action === 'ul' ? 'insertUnorderedList' : 'insertOrderedList';
          document.execCommand(cmd, false, null);
        } else if (action === 'upload') {
          Voxium.utils.notify('Upload initialized: parsing local PDF document...');
        }
      });
    }

    // Input sync listeners to mock data registry updates
    const titleInput = document.getElementById('editor-title');
    const contentBody = document.getElementById('editor-body');

    if (titleInput && contentBody) {
      titleInput.addEventListener('input', () => {
        if (!this.activeNoteId) return;
        const note = Voxium.db.notes.find(n => n.id === this.activeNoteId);
        if (note) {
          note.title = titleInput.value;
          // Update title label inside folder tree
          const treeLabel = document.querySelector(`[data-tree-note-id="${this.activeNoteId}"] .note-node-title`);
          if (treeLabel) treeLabel.textContent = titleInput.value;
        }
      });

      contentBody.addEventListener('input', () => {
        if (!this.activeNoteId) return;
        const note = Voxium.db.notes.find(n => n.id === this.activeNoteId);
        if (note) {
          note.content = contentBody.innerHTML;
        }
      });
    }

    // AI Side Panel Navigation tabs
    const aiTabs = document.querySelector('.ai-sidebar-tabs');
    if (aiTabs) {
      aiTabs.addEventListener('click', (e) => {
        const btn = e.target.closest('.ai-tab-btn');
        if (!btn) return;
        
        const tab = btn.getAttribute('data-tab');
        this.switchAITab(tab);
      });
    }

    // AI Chat Input field submit
    const chatInput = document.getElementById('ai-chat-input');
    const chatSendBtn = document.getElementById('ai-chat-send');
    if (chatInput && chatSendBtn) {
      const sendMsg = () => {
        const text = chatInput.value.trim();
        if (!text) return;
        this.appendChatMessage(text, 'user');
        chatInput.value = '';
        
        // Simulated AI response
        setTimeout(() => {
          this.generateSimulatedAIResponse(text);
        }, 1200);
      };
      
      chatSendBtn.addEventListener('click', sendMsg);
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          sendMsg();
        }
      });
    }
  },

  renderFolderTree: function() {
    const treeContainer = document.getElementById('folder-tree-list');
    if (!treeContainer) return;

    treeContainer.innerHTML = Voxium.db.folders.map(folder => {
      const notes = Voxium.db.notes.filter(n => n.folderId === folder.id);
      
      return `
        <li class="folder-node">
          <div class="folder-row" data-folder-id="${folder.id}">
            <svg viewBox="0 0 24 24" width="16" height="16" class="folder-icon" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
            </svg>
            <span>${Voxium.utils.escapeHTML(folder.name)}</span>
          </div>
          <ul class="notes-list-under-folder">
            ${notes.map(note => `
              <li class="note-node-row" data-tree-note-id="${note.id}">
                <svg viewBox="0 0 24 24" width="12" height="12" style="margin-right:6px; color:var(--muted);" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                <span class="note-node-title">${Voxium.utils.escapeHTML(note.title)}</span>
              </li>
            `).join('')}
          </ul>
        </li>
      `;
    }).join('');

    // Bind navigation click events
    treeContainer.querySelectorAll('.note-node-row').forEach(row => {
      row.addEventListener('click', () => {
        const id = row.getAttribute('data-tree-note-id');
        this.selectNote(id);
      });
    });
  },

  selectNote: function(noteId) {
    this.activeNoteId = noteId;
    const note = Voxium.db.notes.find(n => n.id === noteId);
    if (!note) return;

    // Highlight row active status in sidebar tree
    document.querySelectorAll('.note-node-row').forEach(row => {
      if (row.getAttribute('data-tree-note-id') === noteId) {
        row.classList.add('active');
      } else {
        row.classList.remove('active');
      }
    });

    // Load content details into input fields
    const titleInput = document.getElementById('editor-title');
    const contentBody = document.getElementById('editor-body');

    if (titleInput && contentBody) {
      titleInput.value = note.title;
      contentBody.innerHTML = note.content || '<p>Start typing...</p>';
    }

    // Refresh dynamic summaries based on loaded files
    this.updateEditorAISidebar(note);
  },

  loadNoteFromUrl: function() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (id) {
      this.selectNote(id);
    } else if (Voxium.db.notes.length > 0) {
      // Load first note by default
      this.selectNote(Voxium.db.notes[0].id);
    }
  },

  switchAITab: function(tabName) {
    this.activeTab = tabName;
    document.querySelectorAll('.ai-tab-btn').forEach(btn => {
      if (btn.getAttribute('data-tab') === tabName) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    const summaryPane = document.getElementById('ai-pane-summary');
    const askPane = document.getElementById('ai-pane-ask');
    const searchPane = document.getElementById('ai-pane-search');

    if (summaryPane && askPane && searchPane) {
      summaryPane.style.display = tabName === 'summary' ? 'block' : 'none';
      askPane.style.display = tabName === 'ask' ? 'block' : 'none';
      searchPane.style.display = tabName === 'search' ? 'block' : 'none';
    }

    if (tabName === 'search') {
      this.renderSemanticReferences();
    }
  },

  updateEditorAISidebar: function(note) {
    // Generate simulated AI summarization list
    const summaryList = document.querySelector('#ai-pane-summary .ai-summary-list');
    if (summaryList) {
      if (note.id === 'n1') {
        summaryList.innerHTML = `
          <li>Proposed offline SQLite migration pathways for personal vector storage schemas.</li>
          <li>Identified local execution latency benchmarks on Whisper model weights.</li>
          <li>Recommended async processes pipelines to handle knowledge memory connections.</li>
        `;
      } else if (note.id === 'n2') {
        summaryList.innerHTML = `
          <li>Analysed folk-futurism aesthetic requirements matching parchment colors.</li>
          <li>Outlined 1px hairline border requirements to eliminate heavy graphic loads.</li>
        `;
      } else {
        summaryList.innerHTML = `
          <li>Local-first note containing details on performance index metrics.</li>
          <li>Linked semantic tags: ${note.tags.map(t => `#${t}`).join(', ')}</li>
        `;
      }
    }
  },

  appendChatMessage: function(text, sender) {
    const list = document.getElementById('ai-chat-list');
    if (!list) return;

    const bubble = document.createElement('div');
    bubble.className = `ai-chat-bubble ${sender}`;
    bubble.textContent = text;
    list.appendChild(bubble);

    // Scroll chat list down
    list.scrollTop = list.scrollHeight;
  },

  generateSimulatedAIResponse: function(userText) {
    const query = userText.toLowerCase();
    let responseText = 'I am looking through your offline knowledge database...';

    if (query.includes('quantization')) {
      responseText = 'Q4_K_M quantization weights provide the optimal balance for local CPU execution, utilizing 4.8 GB of RAM and achieving 18 tokens/sec generation speed.';
    } else if (query.includes('sqlite')) {
      responseText = 'To compile SQLite binaries locally, we need to map node-gyp targets during embedded system builds.';
    } else {
      responseText = 'Based on your local files, you have indexed connections between ChromaDB, SQLite vector caches, and Whisper CLI audio transcription segments.';
    }

    this.appendChatMessage(responseText, 'assistant');
  },

  renderSemanticReferences: function() {
    const list = document.getElementById('semantic-search-results');
    if (!list) return;

    // Load semantic matches from simulated DB
    list.innerHTML = Voxium.db.notes.map(note => `
      <div class="semantic-ref-item" onclick="Voxium.editor.selectNote('${note.id}')">
        <div class="semantic-ref-title">${Voxium.utils.escapeHTML(note.title)}</div>
        <div class="semantic-ref-score">Cosine Similarity: ${(0.82 + Math.random()*0.15).toFixed(4)}</div>
        <div class="semantic-ref-desc">${note.content.replace(/<[^>]*>/g, '').substring(0, 80)}...</div>
      </div>
    `).join('');
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.editor.init();
});
