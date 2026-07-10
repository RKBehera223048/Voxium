/* Sidebar Collapse and Active Navigation Controller */

window.Voxium = window.Voxium || {};

Voxium.sidebar = {
  collapsed: false,

  init: function() {
    this.bindEvents();
    this.highlightActiveNavItem();
  },

  bindEvents: function() {
    const toggleBtn = document.getElementById('sidebar-toggle-btn');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener('click', () => {
        this.collapsed = !this.collapsed;
        
        if (this.collapsed) {
          sidebar.classList.add('collapsed');
          toggleBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18">
              <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" fill="currentColor"/>
            </svg>
          `;
        } else {
          sidebar.classList.remove('collapsed');
          toggleBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="18" height="18">
              <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" fill="currentColor"/>
            </svg>
          `;
        }

        // Trigger canvas resize in case waveform is rendering
        setTimeout(() => {
          window.dispatchEvent(new Event('resize'));
        }, 300);
      });
    }
  },

  // Highlight navigation item matching current URL file path
  highlightActiveNavItem: function() {
    const path = window.location.pathname;
    const page = path.split("/").pop();
    
    const navLinks = document.querySelectorAll('.nav-item-link');
    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (page === href || (page === '' && href === 'dashboard.html')) {
        link.classList.add('active');
      } else {
        link.classList.remove('active');
      }
    });
  }
};

document.addEventListener('DOMContentLoaded', () => {
  Voxium.sidebar.init();
});
