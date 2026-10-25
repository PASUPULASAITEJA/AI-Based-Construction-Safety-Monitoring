/**
 * ============================================================================
 * SITEGUARD AI — Enterprise Client Runtime & Application Framework
 * File: static/js/app.js
 * 
 * Features:
 *  - Global SG namespace (SiteGuard Core)
 *  - High-performance Toast Notification System with Progress Bar
 *  - Accessible Modal Dialog Manager (Esc, backdrop, focus-trap)
 *  - Smart Polling with HTML5 Page Visibility API
 *  - RequestAnimationFrame Number Counter Animation
 *  - Canvas Surveillance Overlays & HUD Bounding Boxes
 *  - Keyboard Navigation Shortcuts (g+d, g+m, g+v, g+a, ?)
 *  - REST API JSON client & Drag-and-Drop Dropzone Bindings
 * ============================================================================
 */

(function (window, document) {
  'use strict';

  // ============================================================================
  // GLOBAL NAMESPACE: SG
  // ============================================================================
  const SG = window.SG = window.SG || {};

  // ----------------------------------------------------------------------------
  // 1. TOAST NOTIFICATION SYSTEM
  // ----------------------------------------------------------------------------
  const TOAST_ICONS = {
    success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
    error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    critical: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    warning: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    accent: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
  };

  function getToastContainer() {
    let container = document.getElementById('sg-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'sg-toast-container';
      container.className = 'sg-toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  /**
   * Display an animated toast notification
   * @param {string} type - 'success' | 'error' | 'critical' | 'warning' | 'info' | 'accent'
   * @param {string} title - Heading text
   * @param {string} message - Body text
   * @param {number} [duration=4500] - Duration in ms before auto-dismiss
   */
  SG.toast = function (type, title, message, duration) {
    type = (type || 'info').toLowerCase();
    if (type === 'danger') type = 'error';
    duration = typeof duration === 'number' ? duration : 4500;

    const container = getToastContainer();
    const toast = document.createElement('div');
    toast.className = `sg-toast sg-toast--${type}`;

    const iconHtml = TOAST_ICONS[type] || TOAST_ICONS.info;

    toast.innerHTML = `
      <div class="sg-toast-icon">${iconHtml}</div>
      <div class="sg-toast-content">
        ${title ? `<div class="sg-toast-title">${escapeHtml(title)}</div>` : ''}
        ${message ? `<div class="sg-toast-message">${escapeHtml(message)}</div>` : ''}
      </div>
      <button class="sg-toast-close" title="Dismiss">&times;</button>
      ${duration > 0 ? `<div class="sg-toast-progress" style="animation-duration: ${duration}ms;"></div>` : ''}
    `;

    const closeBtn = toast.querySelector('.sg-toast-close');
    let timer = null;

    function dismiss() {
      if (timer) clearTimeout(timer);
      toast.classList.add('sg-toast--closing');
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 250);
    }

    closeBtn.addEventListener('click', dismiss);

    if (duration > 0) {
      timer = setTimeout(dismiss, duration);
      // Pause timer on hover
      toast.addEventListener('mouseenter', () => {
        if (timer) clearTimeout(timer);
        const bar = toast.querySelector('.sg-toast-progress');
        if (bar) bar.style.animationPlayState = 'paused';
      });
      toast.addEventListener('mouseleave', () => {
        const bar = toast.querySelector('.sg-toast-progress');
        if (bar) bar.style.animationPlayState = 'running';
        timer = setTimeout(dismiss, 1500);
      });
    }

    container.appendChild(toast);
    return { dismiss };
  };

  SG.toast.success = (title, msg, dur) => SG.toast('success', title, msg, dur);
  SG.toast.error = (title, msg, dur) => SG.toast('error', title, msg, dur);
  SG.toast.warning = (title, msg, dur) => SG.toast('warning', title, msg, dur);
  SG.toast.info = (title, msg, dur) => SG.toast('info', title, msg, dur);


  // ----------------------------------------------------------------------------
  // 2. MODAL DIALOG MANAGER
  // ----------------------------------------------------------------------------
  SG.modal = {
    open(modalIdOrEl) {
      const el = typeof modalIdOrEl === 'string'
        ? document.getElementById(modalIdOrEl)
        : modalIdOrEl;
      if (!el) return;

      el.classList.add('active', 'open');
      el.style.display = 'flex';
      document.body.style.overflow = 'hidden';

      // Auto-focus first input if exists
      const input = el.querySelector('input:not([type="hidden"]), select, textarea, button:not(.modal-close)');
      if (input) {
        setTimeout(() => input.focus(), 80);
      }
    },

    close(modalIdOrEl) {
      const el = typeof modalIdOrEl === 'string'
        ? document.getElementById(modalIdOrEl)
        : modalIdOrEl;
      if (!el) return;

      el.classList.remove('active', 'open');
      el.style.display = 'none';

      // Check if any other modal is still active
      if (!document.querySelector('.modal-backdrop.active, .modal-backdrop.open, .sg-modal-backdrop.active, .sg-modal-backdrop.open')) {
        document.body.style.overflow = '';
      }
    },

    closeAll() {
      const modals = document.querySelectorAll('.modal-backdrop, .sg-modal-backdrop');
      modals.forEach(m => {
        m.classList.remove('active', 'open');
        m.style.display = 'none';
      });
      document.body.style.overflow = '';
    }
  };


  // ----------------------------------------------------------------------------
  // 3. SMART POLLING ENGINE (With HTML5 Page Visibility API)
  // ----------------------------------------------------------------------------
  /**
   * Resilient, visibility-aware poller
   * Automatically suspends polling when tab is hidden, resumes on tab activation.
   */
  SG.poll = function (url, intervalMs, callback, options) {
    options = options || {};
    const hiddenIntervalMs = options.hiddenIntervalMs || (intervalMs * 4);
    let timer = null;
    let isStopped = false;
    let consecutiveErrors = 0;

    async function execute() {
      if (isStopped) return;

      try {
        const res = await fetch(url, {
          headers: { 'Accept': 'application/json' },
          cache: 'no-store'
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        consecutiveErrors = 0;
        if (typeof callback === 'function') {
          callback(data);
        }
      } catch (err) {
        consecutiveErrors++;
        if (typeof options.onError === 'function') {
          options.onError(err, consecutiveErrors);
        }
      } finally {
        if (!isStopped) {
          // Dynamic interval: slowed when hidden or experiencing backoff errors
          let delay = document.hidden ? hiddenIntervalMs : intervalMs;
          if (consecutiveErrors > 3) {
            delay = Math.min(delay * 2, 30000);
          }
          timer = setTimeout(execute, delay);
        }
      }
    }

    // React immediately when user returns to tab
    function onVisibilityChange() {
      if (isStopped) return;
      if (!document.hidden) {
        if (timer) clearTimeout(timer);
        execute();
      }
    }
    document.addEventListener('visibilitychange', onVisibilityChange);

    // Initial trigger
    execute();

    return {
      stop() {
        isStopped = true;
        if (timer) clearTimeout(timer);
        document.removeEventListener('visibilitychange', onVisibilityChange);
      },
      resume() {
        if (isStopped) {
          isStopped = false;
          execute();
        }
      },
      trigger() {
        if (timer) clearTimeout(timer);
        execute();
      }
    };
  };


  // ----------------------------------------------------------------------------
  // 4. ANIMATED NUMBER COUNTER (RequestAnimationFrame)
  // ----------------------------------------------------------------------------
  /**
   * Animate a numeric value on an element
   * @param {HTMLElement|string} elOrSelector
   * @param {number} startVal
   * @param {number} endVal
   * @param {number} [duration=700]
   * @param {string} [suffix=""]
   */
  SG.animateCount = function (elOrSelector, startVal, endVal, duration, suffix) {
    const el = typeof elOrSelector === 'string'
      ? document.querySelector(elOrSelector)
      : elOrSelector;
    if (!el) return;

    duration = typeof duration === 'number' ? duration : 700;
    suffix = suffix || '';

    const start = Number(startVal) || 0;
    const end = Number(endVal) || 0;
    const isDecimal = String(endVal).includes('.');
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic curve
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * ease;

      el.textContent = isDecimal
        ? current.toFixed(1) + suffix
        : Math.round(current).toLocaleString() + suffix;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.textContent = isDecimal ? end.toFixed(1) + suffix : end.toLocaleString() + suffix;
      }
    }

    requestAnimationFrame(update);
  };


  // ----------------------------------------------------------------------------
  // 5. CANVAS SURVEILLANCE OVERLAYS & HUD GRAPHICS
  // ----------------------------------------------------------------------------
  SG.canvas = {
    /**
     * Draw HUD Bounding Box with surveillance corner brackets
     */
    drawHudBox(ctx, x, y, w, h, options) {
      options = options || {};
      const color = options.color || '#F97316';
      const bracketLen = Math.min(w * 0.25, h * 0.25, 18);
      const lineWidth = options.lineWidth || 2;
      const label = options.label || '';
      const conf = options.conf;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'square';

      // Top-Left bracket
      ctx.beginPath();
      ctx.moveTo(x, y + bracketLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + bracketLen, y);
      ctx.stroke();

      // Top-Right bracket
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + bracketLen);
      ctx.stroke();

      // Bottom-Left bracket
      ctx.beginPath();
      ctx.moveTo(x, y + h - bracketLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + bracketLen, y + h);
      ctx.stroke();

      // Bottom-Right bracket
      ctx.beginPath();
      ctx.moveTo(x + w - bracketLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - bracketLen);
      ctx.stroke();

      // Optional bounding border (subtle)
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.35;
      ctx.strokeRect(x, y, w, h);
      ctx.globalAlpha = 1.0;

      // Draw Label Badge
      if (label) {
        const text = conf ? `${label} ${(conf * 100).toFixed(0)}%` : label;
        ctx.font = '700 11px JetBrains Mono, monospace';
        const textWidth = ctx.measureText(text).width;
        const tagHeight = 18;
        const tagY = y > tagHeight + 4 ? y - tagHeight - 2 : y + h + 2;

        ctx.fillStyle = color;
        ctx.fillRect(x, tagY, textWidth + 10, tagHeight);

        ctx.fillStyle = '#07090F';
        ctx.fillText(text, x + 5, tagY + 13);
      }

      ctx.restore();
    },

    /**
     * Draw spatial hazard zone polygon
     */
    drawPolygon(ctx, points, color, fillAlpha, name) {
      if (!points || points.length < 3) return;
      ctx.save();
      ctx.strokeStyle = color || '#EF4444';
      ctx.fillStyle = color || '#EF4444';
      ctx.lineWidth = 2;

      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.closePath();

      ctx.globalAlpha = fillAlpha || 0.18;
      ctx.fill();

      ctx.globalAlpha = 0.9;
      ctx.stroke();

      if (name) {
        ctx.globalAlpha = 1.0;
        ctx.font = '700 11px Inter, sans-serif';
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(name, points[0][0] + 6, points[0][1] + 16);
      }

      ctx.restore();
    }
  };


  // ----------------------------------------------------------------------------
  // 6. GLOBAL SHORTCUTS & MODAL HELP
  // ----------------------------------------------------------------------------
  const SHORTCUTS = [
    { chord: 'g d', desc: 'Go to Operations Dashboard', url: '/' },
    { chord: 'g m', desc: 'Go to Live Monitor Workstation', url: '/monitor' },
    { chord: 'g v', desc: 'Go to Violations Review', url: '/violations' },
    { chord: 'g a', desc: 'Go to Alerts Management', url: '/alerts' },
    { chord: 'g s', desc: 'Go to Sites & Zones', url: '/sites' },
    { chord: 'g c', desc: 'Go to Cameras Network', url: '/cameras' },
    { chord: 'g w', desc: 'Go to Worker Tracking', url: '/workers' },
    { chord: 'g r', desc: 'Go to Safety Reports', url: '/reports' },
    { chord: 'g h', desc: 'Go to System Health', url: '/system-health' },
    { chord: 'g i', desc: 'Go to AI Insights', url: '/insights' },
    { chord: '?', desc: 'Show Keyboard Shortcuts Help', action: 'help' }
  ];

  let lastKey = '';
  let lastKeyTime = 0;

  function initShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ignore typing inside form inputs
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) {
        return;
      }

      // Escape always closes active modal
      if (e.key === 'Escape') {
        SG.modal.closeAll();
        return;
      }

      // '?' opens shortcut help
      if (e.key === '?' && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        showShortcutsModal();
        return;
      }

      const now = Date.now();
      const key = e.key.toLowerCase();

      // Check two-key chords starting with 'g'
      if (lastKey === 'g' && (now - lastKeyTime < 1000)) {
        const chord = `g ${key}`;
        const match = SHORTCUTS.find(s => s.chord === chord);
        if (match && match.url) {
          e.preventDefault();
          window.location.href = match.url;
          lastKey = '';
          return;
        }
      }

      lastKey = key;
      lastKeyTime = now;
    });
  }

  function showShortcutsModal() {
    let modal = document.getElementById('sg-shortcuts-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'sg-shortcuts-modal';
      modal.className = 'modal-backdrop';
      modal.onclick = (e) => { if (e.target === modal) SG.modal.close(modal); };

      const rowsHtml = SHORTCUTS.map(s => `
        <div class="flex flex-between flex-center p-8 border-b" style="border-color: var(--border);">
          <span class="fs-13 text-secondary">${escapeHtml(s.desc)}</span>
          <span class="sg-kbd">${escapeHtml(s.chord)}</span>
        </div>
      `).join('');

      modal.innerHTML = `
        <div class="modal-dialog" onclick="event.stopPropagation()">
          <div class="modal-header">
            <h3 class="fs-16 fw-700 text-primary m-0 flex flex-center gap-8">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><line x1="6" y1="8" x2="6" y2="8"/><line x1="10" y1="8" x2="10" y2="8"/><line x1="14" y1="8" x2="14" y2="8"/><line x1="18" y1="8" x2="18" y2="8"/><line x1="6" y1="12" x2="6" y2="12"/><line x1="10" y1="12" x2="10" y2="12"/><line x1="14" y1="12" x2="14" y2="12"/><line x1="18" y1="12" x2="18" y2="12"/><line x1="7" y1="16" x2="17" y2="16"/></svg>
              Keyboard Navigation Shortcuts
            </h3>
            <button class="modal-close" onclick="SG.modal.close('sg-shortcuts-modal')">&times;</button>
          </div>
          <div class="modal-body">
            <div class="flex-col gap-4">
              ${rowsHtml}
            </div>
          </div>
          <div class="modal-footer">
            <button class="btn btn-secondary btn-sm" onclick="SG.modal.close('sg-shortcuts-modal')">Close (Esc)</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
    }
    SG.modal.open(modal);
  }


  // ============================================================================
  // GLOBAL HELPERS & BACKWARD COMPATIBILITY
  // ============================================================================

  /**
   * Universal JSON API fetch wrapper
   */
  window.apiJSON = async function (url, method, payload) {
    method = (method || 'GET').toUpperCase();
    const headers = { 'Accept': 'application/json' };

    // Auto-detect CSRF token
    const csrfInput = document.querySelector('input[name="csrf_token"]');
    const csrfMeta = document.querySelector('meta[name="csrf-token"]');
    const csrfToken = csrfInput ? csrfInput.value : (csrfMeta ? csrfMeta.content : null);
    if (csrfToken) {
      headers['X-CSRFToken'] = csrfToken;
    }

    const init = { method, headers };
    if (payload && method !== 'GET' && method !== 'HEAD') {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(payload);
    }

    const res = await fetch(url, init);
    let data = null;
    try { data = await res.json(); } catch (e) {}

    if (!res.ok) {
      const err = new Error(data && data.message ? data.message : `HTTP ${res.status}`);
      err.response = data;
      err.status = res.status;
      throw err;
    }
    return data;
  };

  /**
   * Robust Drag & Drop dropzone binder
   */
  window.enableFileDrop = function (dropzoneId, inputId, onSelectCallback) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    if (!dropzone || !input) return;

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover', 'border-orange');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover', 'border-orange');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        input.files = files;
        if (typeof onSelectCallback === 'function') {
          onSelectCallback({ target: input });
        }
      }
    });
  };

  /**
   * Escape HTML entities
   */
  window.escapeHtml = function (str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  };

  /**
   * Global snapshot modal handlers
   */
  window.viewSnapshot = function (imageUrl, incidentCode, violationType) {
    const modal = document.getElementById('snapshot-modal');
    const modalImg = document.getElementById('snapshot-modal-img') || document.getElementById('modal-snapshot-img');
    const modalTitle = document.getElementById('snapshot-modal-title') || document.getElementById('modal-incident-title');
    const dlBtn = document.getElementById('snapshot-download-btn');

    if (modalImg) modalImg.src = imageUrl;
    if (modalTitle) modalTitle.textContent = `Incident Evidence: ${incidentCode || 'Snapshot'} (${violationType || 'Breach'})`;
    if (dlBtn) dlBtn.href = imageUrl;

    if (modal) {
      SG.modal.open(modal);
    }
  };

  window.closeSnapshotModal = function () {
    const modal = document.getElementById('snapshot-modal');
    if (modal) SG.modal.close(modal);
  };


  // ============================================================================
  // AUTO-INITIALIZATION ON DOMContentLoaded
  // ============================================================================
  document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Keyboard Shortcuts
    initShortcuts();

    // 2. Auto-wire backdrop clicks for modals
    document.querySelectorAll('.modal-backdrop, .sg-modal-backdrop').forEach(backdrop => {
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) {
          SG.modal.close(backdrop);
        }
      });
    });

    // 3. Auto-wire modal close buttons
    document.querySelectorAll('.modal-close, .modal-close-btn, [data-dismiss="modal"]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modal = btn.closest('.modal-backdrop, .sg-modal-backdrop');
        if (modal) SG.modal.close(modal);
      });
    });

    // 4. Lucide icons pass
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  });

})(window, document);
