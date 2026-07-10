/* Voxium Utilities & Database Simulation */

// Global App Namespace
window.Voxium = window.Voxium || {};

// Database Mock Simulation (Local-first state)
Voxium.db = {
  // Folder tree nodes
  folders: [
    { id: 'f1', name: 'Brain Dump', icon: 'folder' },
    { id: 'f2', name: 'Work Meetings', icon: 'briefcase' },
    { id: 'f3', name: 'AI Research', icon: 'cpu' },
    { id: 'f4', name: 'Personal Journal', icon: 'book-open' }
  ],
  
  // Notes linked to folders or independent
  notes: [
    {
      id: 'n1',
      folderId: 'f2',
      title: 'Voxium Architecture Sync',
      content: '<p>We discussed migrating the database layer to support offline vector indexes. ChromaDB is working well but we need local indexing for mobile devices.</p><p>Key points:</p><ul><li>Whisper local execution takes 800ms for a 5s chunk.</li><li>We should explore ONNX runtime for client-side processing.</li><li>Graph memory structures must be updated asynchronously.</li></ul>',
      tags: ['architecture', 'vector-db', 'offline'],
      modified: '2026-06-25T09:12:00Z',
      transcriptId: 't1'
    },
    {
      id: 'n2',
      folderId: 'f1',
      title: 'Thoughts on Folk-Futurism Design',
      content: '<p>The concept of bridging Mithila tribal paint styles with high-end tech surfaces is fascinating. The warm cream backgrounds (#FFF9ED) offer immense reading comfort. Instead of heavy borders, we must keep borders thin (#D8CDB2) and place small, elegant Indian accents in icons and empty states.</p>',
      tags: ['design-system', 'ideation'],
      modified: '2026-06-25T08:30:00Z'
    },
    {
      id: 'n3',
      folderId: 'f3',
      title: 'Local LLM Performance Benchmark',
      content: '<p>Llama-3 8B Instruct running via Llama.cpp with Q4_K_M quantization shows steady performance. On a standard modern laptop:</p><ul><li>Prefill speed: 32 tokens/sec.</li><li>Generation speed: 18 tokens/sec.</li><li>Context window utilized: 4096 tokens.</li></ul>',
      tags: ['llm', 'benchmarks', 'local-first'],
      modified: '2026-06-24T18:45:00Z'
    },
    {
      id: 'n4',
      folderId: 'f4',
      title: 'Morning Reflections & Goals',
      content: '<p>Woke up early. The air is fresh today. Goals for today: complete the Voxium frontend framework, write the responsive stylesheets, and verify the waveforms drawing routines.</p>',
      tags: ['journal', 'daily'],
      modified: '2026-06-25T06:00:00Z'
    }
  ],
  
  // Transcripts simulator
  transcripts: {
    't1': [
      { speaker: 'Speaker A (Lead)', text: 'Alright, let\'s look at our local vector database storage path.', time: '10:00' },
      { speaker: 'Speaker B (Dev)', text: 'ChromaDB is running as an embedded process. It\'s fast, but it needs sqlite binaries compiled locally.', time: '10:15' },
      { speaker: 'Speaker A (Lead)', text: 'Understood. We need to make sure always-listening mode feeds audio chunks directly to Whisper.', time: '10:45' }
    ]
  },

  // Action Items checklist
  actions: [
    { id: 'a1', text: 'Optimize Whisper quantization weights (use q5_k)', completed: false },
    { id: 'a2', text: 'Define border radii values for organic buttons outline', completed: true },
    { id: 'a3', text: 'Setup indexing pipeline for semantic memory graph nodes', completed: false }
  ]
};

// UI Helpers
Voxium.utils = {
  // Format Date ISO into user friendly string
  formatDate: function(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    
    if (date.toDateString() === now.toDateString()) {
      return 'Today, ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    const options = { month: 'short', day: 'numeric' };
    if (date.getFullYear() !== now.getFullYear()) {
      options.year = 'numeric';
    }
    return date.toLocaleDateString([], options);
  },

  // Generate simple short UUIDs
  generateId: function() {
    return 'n_' + Math.random().toString(36).substr(2, 9);
  },

  // Safely escape HTML to prevent XSS issues in user generated areas
  escapeHTML: function(str) {
    return str.replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  },

  // Dynamic notification alert
  notify: function(message, type = 'success') {
    const banner = document.createElement('div');
    banner.style.position = 'fixed';
    banner.style.bottom = '24px';
    banner.style.left = '50%';
    banner.style.transform = 'translateX(-50%) translateY(20px)';
    banner.style.padding = '12px 24px';
    banner.style.borderRadius = '6px';
    banner.style.fontSize = '0.9rem';
    banner.style.fontWeight = '600';
    banner.style.color = '#ffffff';
    banner.style.zIndex = '9999';
    banner.style.boxShadow = '0 4px 15px rgba(31, 28, 11, 0.15)';
    banner.style.transition = 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)';
    banner.style.opacity = '0';
    
    if (type === 'success') banner.style.backgroundColor = '#3E8F56';
    else if (type === 'warning') banner.style.backgroundColor = '#C99B00';
    else banner.style.backgroundColor = '#BA1A1A';
    
    banner.textContent = message;
    document.body.appendChild(banner);
    
    // Animate in
    setTimeout(() => {
      banner.style.transform = 'translateX(-50%) translateY(0)';
      banner.style.opacity = '1';
    }, 50);
    
    // Animate out
    setTimeout(() => {
      banner.style.transform = 'translateX(-50%) translateY(-20px)';
      banner.style.opacity = '0';
      setTimeout(() => banner.remove(), 300);
    }, 3000);
  }
};
