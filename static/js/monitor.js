/**
 * SiteGuard AI - Live Monitoring Controller, Source Switcher, & Real-Time Worker Roster
 */

let statusPollInterval = null;
let currentSource = 'simulator';
let browserStream = null;
let browserCaptureInterval = null;

// Helper element getters with multi-ID fallbacks
function getStreamImg() {
  return document.getElementById('monitor-feed') || document.getElementById('live-video-stream');
}

function getBrowserVideo() {
  return document.getElementById('browser-webcam') || document.getElementById('browser-webcam-element');
}

function getStreamBadge() {
  return document.getElementById('live-stream-badge') || document.querySelector('.sg-cam-badge');
}

function getRosterContainer() {
  return document.getElementById('live-workers-list') || document.getElementById('worker-roster-list');
}

function getWorkerCountBadge() {
  return document.getElementById('workers-count-badge') || document.getElementById('worker-count-badge');
}

function getSourceSelect() {
  return document.getElementById('monitor-cam-select') || document.getElementById('camera-source-select');
}

document.addEventListener('DOMContentLoaded', () => {
  setupStreamErrorHandler();
  startPollingStatus();
  syncInitialSourceState();
});

function syncInitialSourceState() {
  // Sync UI to default mode (simulator)
  const sel = getSourceSelect();
  if (sel && sel.value) {
    currentSource = sel.value;
  }
  updateSourceUI(currentSource);
}

function setupStreamErrorHandler() {
  const streamImg = getStreamImg();
  if (streamImg) {
    streamImg.onerror = () => {
      console.warn('[Monitor] Video stream error/disconnect, attempting reconnect...');
      setTimeout(() => {
        if (currentSource !== 'browser') {
          const img = getStreamImg();
          if (img) img.src = '/video_feed?t=' + Date.now();
        }
      }, 1200);
    };
  }
}

function startPollingStatus() {
  if (statusPollInterval) clearInterval(statusPollInterval);
  statusPollInterval = setInterval(fetchLiveStatus, 700);
  fetchLiveStatus(); // Immediate first fetch
}

async function fetchLiveStatus() {
  try {
    const res = await fetch('/api/detections/frame');
    if (res.ok) {
      const data = await res.json();
      updateWorkerRoster(data.workers || []);
      updateMonitorHeader(data);
    }
  } catch (err) {
    // Network or background glitch ignored gracefully
  }
}

function updateMonitorHeader(data) {
  const countBadge = getWorkerCountBadge();
  const monitorWorkers = document.getElementById('monitor-workers');
  const count = data.worker_count !== undefined ? data.worker_count : (data.workers ? data.workers.length : 0);

  if (countBadge) {
    countBadge.innerText = `${count} Active`;
  }
  if (monitorWorkers) {
    monitorWorkers.innerText = count;
  }

  const fpsDisplay = document.getElementById('fps-display');
  const monitorFps = document.getElementById('monitor-fps');
  if (data.fps !== undefined) {
    const fpsVal = Number(data.fps).toFixed(1);
    if (fpsDisplay) fpsDisplay.innerText = fpsVal;
    if (monitorFps) monitorFps.innerText = fpsVal;
  }

  const monitorViolations = document.getElementById('monitor-violations');
  if (monitorViolations && data.violation_count !== undefined) {
    monitorViolations.innerText = data.violation_count;
  }
}

const PPE_CONFIG = [
  { key: 'helmet', altKey: 'hardhat', label: 'Helmet', icon: '🪖' },
  { key: 'goggles', altKey: 'glasses', label: 'Goggles', icon: '🥽' },
  { key: 'earmuff', altKey: 'earmuffs', label: 'Earmuff', icon: '🎧' },
  { key: 'vest', altKey: 'safety_vest', label: 'Vest', icon: '🦺' },
  { key: 'harness', altKey: 'safety_harness', label: 'Harness', icon: '🪢' },
  { key: 'safety_pants', altKey: 'pants', label: 'Pants', icon: '👖' },
  { key: 'gloves', altKey: 'glove', label: 'Gloves', icon: '🧤' },
  { key: 'boots', altKey: 'shoes', label: 'Boots', icon: '🥾' }
];

function renderPPEBadge(val) {
  if (val === true || val === 'compliant' || val === 1) {
    return `<span style="color: #4ade80; font-weight: 700;">✓</span>`;
  } else if (val === 'uncertain' || val === '?') {
    return `<span style="color: #94a3b8; font-weight: 700;" title="Uncertain / Verifying">?</span>`;
  } else {
    return `<span style="color: #f87171; font-weight: 700;">✕</span>`;
  }
}

function updateWorkerRoster(workers) {
  const container = getRosterContainer();
  if (!container) return;

  if (!workers || workers.length === 0) {
    container.innerHTML = `
      <div class="sg-empty" style="padding: 30px 10px; text-align: center; color: var(--text-3, #94a3b8);">
        <div style="font-size: 26px; margin-bottom: 6px;">👷</div>
        <p style="font-size: 11px; margin: 0;">Awaiting worker observations in camera stream</p>
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  workers.forEach(w => {
    const card = document.createElement('div');
    const isViolation = (w.violations && w.violations.length > 0) || (w.compliance_score !== undefined && w.compliance_score < 0.8);
    const scoreVal = w.compliance_score !== undefined ? w.compliance_score : 1.0;
    const scorePct = Math.round(scoreVal <= 1.0 ? scoreVal * 100 : scoreVal);

    card.className = `sg-card mb-2`;
    card.style.cssText = `background: var(--surface-2, #111827); border-radius: 8px; padding: 10px; margin-bottom: 8px; border: 1px solid ${scorePct < 70 ? 'var(--red, #ef4444)' : 'var(--border, #1f2937)'};`;

    let scoreBadgeClass = scorePct >= 80 ? 'sg-badge-resolved' : scorePct >= 50 ? 'sg-badge-warning' : 'sg-badge-critical';
    let scoreColor = scorePct >= 80 ? '#22c55e' : scorePct >= 50 ? '#f59e0b' : '#ef4444';

    const ppeDict = w.ppe || w;
    const missingList = (w.missing_ppe || []).map(s => String(s).toLowerCase());
    const detectedList = (w.detected_ppe || []).map(s => String(s).toLowerCase());

    let ppeGridHtml = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; background: rgba(0,0,0,0.3); padding: 6px; border-radius: 6px; margin: 8px 0;">';
    PPE_CONFIG.forEach(item => {
      let val = ppeDict[item.key];
      if (val === undefined && item.altKey && ppeDict[item.altKey] !== undefined) {
        val = ppeDict[item.altKey];
      }
      if (val === undefined) {
        if (missingList.some(m => m.includes(item.key) || (item.altKey && m.includes(item.altKey)))) {
          val = false;
        } else if (detectedList.some(d => d.includes(item.key) || (item.altKey && d.includes(item.altKey)))) {
          val = true;
        }
      }

      ppeGridHtml += `
        <div style="display: flex; flex-direction: column; align-items: center; font-size: 10px; padding: 2px 0;">
          <span style="font-size: 12px;">${item.icon}</span>
          <span style="color: #94a3b8; font-size: 8px; text-transform: uppercase;">${item.label}</span>
          <div>${renderPPEBadge(val)}</div>
        </div>
      `;
    });
    ppeGridHtml += '</div>';

    const workerLabel = w.label || (w.worker_id !== undefined ? `Worker #${w.worker_id}` : 'Worker');

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 12px; color: #f8fafc;">${workerLabel}</span>
        <span class="sg-badge ${scoreBadgeClass}" style="font-size: 10px; padding: 1px 6px;">
          ${scorePct}% Compliance
        </span>
      </div>
      ${ppeGridHtml}
      <div style="font-size: 10px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
        <span>Zone: <strong style="color: #cbd5e1;">${w.zone_name || 'Active Site'}</strong></span>
        ${w.missing_ppe && w.missing_ppe.length > 0 ? `<span style="color: #f87171; font-weight: 600;">⚠️ ${w.missing_ppe.slice(0, 2).join(', ')}</span>` : ''}
      </div>
    `;

    container.appendChild(card);
  });
}

function updateSourceUI(normSource) {
  // Update top button tabs
  const tabHw = document.getElementById('tab-hardware');
  const tabBr = document.getElementById('tab-browser');
  const tabSim = document.getElementById('tab-simulator');

  if (tabHw) tabHw.className = (normSource === 'webcam') ? 'sg-btn sg-btn-sm sg-btn-primary' : 'sg-btn sg-btn-sm sg-btn-secondary';
  if (tabBr) tabBr.className = (normSource === 'browser') ? 'sg-btn sg-btn-sm sg-btn-primary' : 'sg-btn sg-btn-sm sg-btn-secondary';
  if (tabSim) tabSim.className = (normSource === 'simulator') ? 'sg-btn sg-btn-sm sg-btn-primary' : 'sg-btn sg-btn-sm sg-btn-secondary';

  // Update dropdown select
  const sel = getSourceSelect();
  if (sel && sel.value !== normSource) {
    sel.value = normSource;
  }

  // Update status badge
  const badge = getStreamBadge();
  if (badge) {
    if (normSource === 'simulator') {
      badge.innerText = 'SIMULATOR ACTIVE';
      badge.className = 'sg-cam-badge';
    } else if (normSource === 'browser') {
      badge.innerText = 'BROWSER WEBRTC';
      badge.className = 'sg-cam-badge rec';
    } else {
      badge.innerText = 'WEBCAM ONLINE';
      badge.className = 'sg-cam-badge';
    }
  }
}

async function switchCameraSource(source) {
  let normSource = source;
  if (source === 'hardware') normSource = 'webcam';
  currentSource = normSource;

  updateSourceUI(normSource);

  if (normSource === 'browser') {
    startBrowserWebcam();
  } else {
    stopBrowserWebcam();

    const streamImg = getStreamImg();
    const videoEl = getBrowserVideo();
    if (videoEl) videoEl.style.display = 'none';
    if (streamImg) {
      streamImg.style.display = 'block';
      streamImg.src = '/video_feed?t=' + Date.now();
    }

    try {
      await fetch('/camera/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: normSource })
      });
      if (streamImg) {
        streamImg.src = '/video_feed?t=' + Date.now();
      }
    } catch (err) {
      console.error('[Monitor] Mode switch error:', err);
    }
  }
}

// Global aliases to ensure all HTML onclick bindings work flawlessly
window.switchCameraSource = switchCameraSource;
window.switchSource = switchCameraSource;
window.changeCameraSource = switchCameraSource;

async function startBrowserWebcam() {
  const videoEl = getBrowserVideo();
  const streamImg = getStreamImg();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (window.sgToast) {
      sgToast('error', 'Browser Webcam', 'WebRTC camera is not supported or permission denied.');
    } else {
      alert('Browser WebRTC camera is not supported on this browser.');
    }
    switchCameraSource('simulator');
    return;
  }

  // Ensure capture canvas exists
  let captureCanvas = document.getElementById('browser-capture-canvas');
  if (!captureCanvas) {
    captureCanvas = document.createElement('canvas');
    captureCanvas.id = 'browser-capture-canvas';
    captureCanvas.style.display = 'none';
    document.body.appendChild(captureCanvas);
  }

  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });

    if (videoEl) {
      videoEl.srcObject = browserStream;
      videoEl.style.display = 'none'; // Keep hidden, display AI-annotated frames on streamImg
      await videoEl.play();
    }
    if (streamImg) {
      streamImg.style.display = 'block';
    }

    const ctx = captureCanvas.getContext('2d');
    captureCanvas.width = 640;
    captureCanvas.height = 480;

    let isProcessing = false;

    if (browserCaptureInterval) clearInterval(browserCaptureInterval);
    browserCaptureInterval = setInterval(async () => {
      if (isProcessing || !browserStream || !videoEl) return;
      isProcessing = true;

      try {
        ctx.drawImage(videoEl, 0, 0, 640, 480);
        const b64 = captureCanvas.toDataURL('image/jpeg', 0.75);

        const res = await fetch('/api/process_browser_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: b64 })
        });

        if (res.ok) {
          const data = await res.json();
          if (data.annotated_image && streamImg) {
            streamImg.src = data.annotated_image;
          }
          if (data.summary) {
            updateWorkerRoster(data.summary.workers || []);
            updateMonitorHeader(data.summary);
          }
        }
      } catch (e) {
        console.warn('[Monitor] Browser frame processing:', e);
      } finally {
        isProcessing = false;
      }
    }, 120); // ~8 FPS WebRTC stream to server for ultra-responsive low-latency processing

    if (window.sgToast) {
      sgToast('success', 'Webcam Connected', 'Browser camera is now streaming with live PPE inference.');
    }
  } catch (err) {
    console.error('Could not access browser camera:', err);
    if (window.sgToast) {
      sgToast('error', 'Webcam Error', 'Could not open browser camera: ' + err.message);
    } else {
      alert('Could not access browser camera: ' + err.message);
    }
    switchCameraSource('simulator');
  }
}

function stopBrowserWebcam() {
  if (browserCaptureInterval) {
    clearInterval(browserCaptureInterval);
    browserCaptureInterval = null;
  }
  if (browserStream) {
    browserStream.getTracks().forEach(t => t.stop());
    browserStream = null;
  }
  const videoEl = getBrowserVideo();
  if (videoEl) {
    videoEl.srcObject = null;
    videoEl.style.display = 'none';
  }
}

async function startCamera() {
  if (currentSource === 'browser') {
    startBrowserWebcam();
    return;
  }

  try {
    const res = await fetch('/camera/start', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      const streamImg = getStreamImg();
      if (streamImg) streamImg.src = '/video_feed?t=' + Date.now();
      const badge = getStreamBadge();
      if (badge) {
        badge.innerText = 'Camera Online';
      }
      if (window.sgToast) sgToast('success', 'Camera Started', 'Video stream resumed.');
    }
  } catch (err) {
    console.error('Camera start error:', err);
  }
}
window.startCamera = startCamera;

async function stopCamera() {
  if (currentSource === 'browser') {
    stopBrowserWebcam();
    const badge = getStreamBadge();
    if (badge) badge.innerText = 'Camera Paused';
    return;
  }

  try {
    const res = await fetch('/camera/stop', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      const badge = getStreamBadge();
      if (badge) badge.innerText = 'Camera Paused';
      if (window.sgToast) sgToast('info', 'Camera Paused', 'Video feed paused.');
    }
  } catch (err) {
    console.error('Stop camera error:', err);
  }
}
window.stopCamera = stopCamera;
