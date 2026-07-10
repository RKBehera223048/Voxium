/* Voxium Landing Page Interactions & Animations */

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveals();
  initFAQAccordion();
  initSVGFloatLoop();
  initSmoothScroll();
});

// 1. INTERSECTION OBSERVER SCROLL REVEALS
function initScrollReveals() {
  const elements = document.querySelectorAll('.reveal-on-scroll');
  if (elements.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        // Unobserve once shown
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  });

  elements.forEach(el => observer.observe(el));
}

// 2. FAQ ACCORDION COLLAPSE CONTROL
function initFAQAccordion() {
  const accordion = document.querySelector('.faq-wrapper');
  if (!accordion) return;

  accordion.addEventListener('click', (e) => {
    const trigger = e.target.closest('.faq-trigger');
    if (!trigger) return;

    const item = trigger.closest('.faq-item');
    const isActive = item.classList.contains('active');

    // Close all other accordion items first
    accordion.querySelectorAll('.faq-item').forEach(i => {
      i.classList.remove('active');
    });

    // Toggle active on clicked item
    if (!isActive) {
      item.classList.add('active');
    }
  });
}

// 3. PROCEDURAL SVG FLOATING NODE SIMULATION
function initSVGFloatLoop() {
  const nodes = document.querySelectorAll('.pulse-node, .pulse-node-delayed');
  const links = document.querySelectorAll('.link-line');
  const wavePaths = document.querySelectorAll('.wave-path');
  
  if (nodes.length === 0 && wavePaths.length === 0) return;

  let angle = 0;

  function animate() {
    angle += 0.02;
    
    // Wave paths offset animation (Procedural vibration)
    wavePaths.forEach((path, index) => {
      const freq = 0.05 + index * 0.02;
      const amp = 4 + index * 2;
      const offset = angle * 2 + index * Math.PI;
      
      let d = `M 10,${150 + index * 10}`;
      for (let x = 10; x <= 430; x += 10) {
        const y = (150 + index * 15) + Math.sin(x * freq + offset) * amp;
        d += ` L ${x},${y}`;
      }
      path.setAttribute('d', d);
    });

    // Node items float circles translation
    nodes.forEach((node, idx) => {
      const floatY = Math.sin(angle + idx) * 4;
      const floatX = Math.cos(angle * 0.5 + idx) * 3;
      node.style.transform = `translate(${floatX}px, ${floatY}px)`;
      node.style.transformOrigin = 'center';
    });

    // Adjust lines linking nodes
    links.forEach(link => {
      const sourceId = link.getAttribute('data-source');
      const targetId = link.getAttribute('data-target');
      const sourceNode = document.getElementById(sourceId);
      const targetNode = document.getElementById(targetId);

      if (sourceNode && targetNode) {
        const sx = parseFloat(sourceNode.getAttribute('cx')) || 0;
        const sy = parseFloat(sourceNode.getAttribute('cy')) || 0;
        const tx = parseFloat(targetNode.getAttribute('cx')) || 0;
        const ty = parseFloat(targetNode.getAttribute('cy')) || 0;

        // Apply same offset translation roughly to coordinates
        link.setAttribute('x1', sx);
        link.setAttribute('y1', sy);
        link.setAttribute('x2', tx);
        link.setAttribute('y2', ty);
      }
    });

    requestAnimationFrame(animate);
  }

  animate();
}

// 4. SMOOTH NAVIGATION SCROLL LINKS
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      if (targetId === '#') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }
      
      const target = document.querySelector(targetId);
      if (target) {
        const offset = 72; // Nav height offset
        const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - offset;
        
        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });
      }
    });
  });
}
