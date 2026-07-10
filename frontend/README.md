# Voxium - AI Personal Memory System Frontend

Voxium is a local-first, AI-powered Personal Memory System. It functions as "A Second Brain," helping users capture transcripts, analyze documents, graph notes, and interact with a local voice assistant.

This repository contains the premium, production-ready frontend for Voxium, built using semantic HTML5, CSS3, and Vanilla JavaScript. It adheres to an editorial, minimal design language with warm cream tones and Literata/Work Sans typography.

## Project Structure

```
frontend/
├── index.html           # Landing Dashboard
├── login.html           # Login & Onboarding
├── meeting.html         # Live Meeting Mode
├── editor.html          # Document Editor
├── notes.html           # Notes & Database Manager
├── settings.html        # Configuration Panels
├── README.md            # Project Information
│
├── assets/
│   ├── css/
│   │   ├── variables.css   # Theme Design Tokens
│   │   ├── global.css      # Base Element Reset & Typography
│   │   ├── layout.css      # App Grids & Spacing
│   │   ├── sidebar.css     # Navigation Drawer
│   │   ├── navbar.css      # Search Bar & Top Stats
│   │   ├── buttons.css     # Ripple & Hover Interactions
│   │   ├── cards.css       # Core Cards & Lists
│   │   ├── forms.css       # Text Fields & Inputs
│   │   ├── meeting.css     # Waveform Canvas & Diarization
│   │   ├── editor.css      # Custom Text Editing Panel
│   │   ├── notes.css       # Note Library Grid
│   │   ├── settings.css    # Stats & Synced Paths
│   │   └── responsive.css  # Responsive Adjustments
│   ├── icons/              # Stylized Indian Accent Glyphs
│   ├── fonts/              # Custom Local Webfonts (optional)
│   └── images/             # Visual Assets
│
└── js/
    ├── app.js           # Core State Bootstrap
    ├── sidebar.js       # Navigation Events
    ├── navbar.js        # Search Cmd+K Modal
    ├── meeting.js       # Audio waveform, Timer & Semantic Graph
    ├── editor.js        # Rich-text handlers & AI Ask Panels
    ├── voice.js         # Always listening mic controllers
    ├── animations.js    # Ripples, skeleton loaders & page fades
    └── utils.js         # Common state & mock data formatters
```

## Running Locally

Since this is built with standard HTML, CSS, and JS, you can run it directly:
1. Open any HTML page in your browser.
2. For the best experience (and to allow full module imports or local asset path loading if needed), serve the files using a simple HTTP server:
   ```bash
   # Using python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   ```
3. Open `http://localhost:8000` in your web browser.

## Design Ethos
- **Minimalist & Warm:** Utilizes soft parchment-colored backgrounds (#FFF9ED / #FAF7ED) to reduce eye strain.
- **Editorial Typography:** Bold Serif Literata headlines combined with high-contrast, clean sans-serif Work Sans details.
- **Micro-interactions:** Fine-tuned CSS transitions, ripple effects on interactions, animated HTML5 canvas waveforms, and animated loading skeletons.
