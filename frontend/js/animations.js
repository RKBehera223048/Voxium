/* UI Interactions, Transitions & Skeleton Loaders */

window.Voxium = window.Voxium || {};

Voxium.animations = {
  init: function() {
    this.bindRipples();
    this.applyPageTransitions();
  },

  // Premium ripple click effect on clickable elements
  bindRipples: function() {
    document.addEventListener('click', function(e) {
      const button = e.target.closest('.btn, .nav-item-link, .card-hover, .toggle-view-btn');
      if (!button) return;

      // Avoid adding ripples to voice floating button, as it uses pulse ring
      if (button.classList.contains('btn-voice-float')) return;

      const rect = button.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const ripple = document.createElement('span');
      ripple.classList.add('ripple-effect');
      ripple.style.left = `${x}px`;
      ripple.style.top = `${y}px`;
      
      const maxDim = Math.max(rect.width, rect.height);
      ripple.style.width = ripple.style.height = `${maxDim}px`;

      button.appendChild(ripple);

      ripple.addEventListener('animationend', function() {
        ripple.remove();
      });
    });
  },

  // Standard fade-in animations on load
  applyPageTransitions: function() {
    const mainContents = document.querySelector('.content-wrapper, .editor-layout, .settings-layout');
    if (mainContents) {
      mainContents.classList.add('fade-in');
    }
  },

  // Toggle Loading Skeleton Elements
  showSkeletons: function(parentSelector) {
    const parent = document.querySelector(parentSelector);
    if (!parent) return;

    // Create and insert skeletons
    parent.classList.add('loading-state');
    const skeletonHTML = `
      <div class="skeleton-wrapper" style="display:flex; flex-direction:column; gap:16px; width:100%;">
        <div class="skeleton" style="height:24px; width:45%;"></div>
        <div class="skeleton" style="height:100px; width:100%;"></div>
        <div class="skeleton" style="height:16px; width:80%;"></div>
      </div>
    `;
    parent.setAttribute('data-original-content', parent.innerHTML);
    parent.innerHTML = skeletonHTML;
  },

  hideSkeletons: function(parentSelector) {
    const parent = document.querySelector(parentSelector);
    if (!parent || !parent.classList.contains('loading-state')) return;

    parent.classList.remove('loading-state');
    const original = parent.getAttribute('data-original-content');
    if (original) {
      parent.innerHTML = original;
    }
  }
};

// Start when document is ready
document.addEventListener('DOMContentLoaded', () => {
  Voxium.animations.init();
});
