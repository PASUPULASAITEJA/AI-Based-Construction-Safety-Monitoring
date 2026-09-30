/**
 * SiteGuard AI — Live Monitoring Controller
 */

let statusPollInterval = null;
let currentSource = 'webcam';
let browserStream = null;
let browserCaptureInterval = null;

document.addEventListener('DOMContentLoaded', () => {
  startPollingStatus();
  setupStreamErrorHandler();
});

function setupStreamErrorHandler() {
  const streamImg = document.getElementById('live-video-stream');
  if (streamImg) {
    streamImg.onerror = () => {
      setTimeout(() => {
        if (currentSource !== 'browser') {
          streamImg.src = '/video_feed?t=' + Date.now();
        }
      }, 2000);
    };
  }
}

function startPollingStatus() {
  if (statusPollInterval) clearInterval(statusPollInterval);
  statusPollInterval = setInterval(fetchLiveStatus, 800);
}

async function fetchLiveStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      updateWorkerRoster(data.workers || []);
      updateHUD(data);
    }
  } catch (err) { /* ignore */ }
}

function updateHUD(data) {
  const countBadge = document.getElementById('worker-count-badge');
  if (countBadge) countBadge.textContent = data.worker_count || 0;

  const hudFps = document.getElementById('hud-fps');
  const hudWorkers = document.getElementById('hud-workers');
  const hudViolations = document.getElementById('hud-violations');
  const hudChannel = document.getElementById('hud-channel');

  if (hudFps && data.fps !== undefined) hudFps.textContent = data.fps.toFixed(1);
  if (hudWorkers) hudWorkers.textContent = data.worker_count || 0;
  if (hudViolations) hudViolations.textContent = data.violation_count || 0;
  if (hudChannel) hudChannel.textContent = data.is_live ? 'Live' : 'Idle';
}

function refreshWorkerList() { fetchLiveStatus(); }

function updateWorkerRoster(workers) {
  const container = document.getElementById('worker-roster-list');
  if (!container) return;

  if (!workers || workers.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="padding:30px 10px;">
        <div class="empty-state-title">No workers detected</div>
        <div class="empty-state-text">Waiting for camera feed.</div>
      </div>`;
    return;
  }

  container.innerHTML = '';
  workers.forEach(w => {
    const card = document.createElement('div');
    const isViolation = w.violations && w.violations.length > 0;
    card.className = 'worker-card' + (isViolation ? ' violation' : '');

    let badgeClass = 'badge-green';
    if (w.status === 'CRITICAL VIOLATION') badgeClass = 'badge-red';
    else if (isViolation) badgeClass = 'badge-yellow';
    else if (w.status === 'PENDING VERIFICATION') badgeClass = 'badge-yellow';

    card.innerHTML = `
      <div class="worker-card-header">
        <span class="worker-id-title">${w.label}</span>
        <span class="badge ${badgeClass}">${w.status}</span>
      </div>
      <div class="worker-ppe-status">
        <div style="font-size: 11px; font-weight: 600; margin-bottom: 6px; color: var(--text-secondary);">CURRENT PPE STATUS</div>
        ${window.PPE_CAPABILITIES ? window.PPE_CAPABILITIES.map(cat => {
            const hasCat = w[`has_${cat.category}`];
            if (cat.is_supported) {
                return `
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                    <span style="color: var(--text-primary);">${cat.name}</span>
                    <span style="color: ${hasCat ? 'var(--color-compliant)' : 'var(--color-critical)'}; font-weight: 600;">
                        ${hasCat ? '✓ Detected' : '✗ Missing'}
                    </span>
                </div>`;
            } else {
                return `
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                    <span style="color: var(--text-muted);">${cat.name}</span>
                    <span style="color: var(--text-muted);">— Unsupported</span>
                </div>`;
            }
        }).join('') : `
        <div class="worker-ppe-row">
            <span class="ppe-pill ${w.helmet ? 'ok' : 'no'}">${w.helmet ? 'Helmet ✓' : 'No Helmet'}</span>
            <span class="ppe-pill ${w.vest ? 'ok' : 'no'}">${w.vest ? 'Vest ✓' : 'No Vest'}</span>
        </div>
        `}
      </div>
      <div style="margin-top:8px;font-size:11px;color:var(--text-muted);display:flex;justify-content:space-between;border-top:1px solid var(--border-color);padding-top:6px;">
        <span>Zone: <strong>${w.zone_name || 'General'}</strong></span>
        ${isViolation ? '<span style="color:var(--red);font-weight:600;max-width:150px;text-align:right;">' + w.violations.join(', ') + '</span>' : ''}
      </div>`;
    container.appendChild(card);
  });
}

async function switchCameraSource(source) {
  currentSource = source;
  stopBrowserWebcam();

  if (source === 'browser') {
    startBrowserWebcam();
  } else {
    const streamImg = document.getElementById('live-video-stream');
    if (streamImg) {
      streamImg.style.display = '';
      streamImg.src = '/video_feed?' + Date.now();
    }
    document.getElementById('live-stream-badge').textContent = 'Online';
    document.getElementById('live-stream-badge').className = 'badge badge-green';
  }
}

async function startBrowserWebcam() {
  const videoEl = document.getElementById('browser-webcam-element');
  const streamImg = document.getElementById('live-video-stream');
  const captureCanvas = document.getElementById('browser-capture-canvas');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('WebRTC not supported in this browser.');
    return;
  }

  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
    });
    videoEl.srcObject = browserStream;
    await videoEl.play();

    document.getElementById('live-stream-badge').textContent = 'WebRTC Active';
    document.getElementById('live-stream-badge').className = 'badge badge-green';

    const ctx = captureCanvas.getContext('2d');
    captureCanvas.width = 640;
    captureCanvas.height = 480;
    let isProcessing = false;

    if (browserCaptureInterval) clearInterval(browserCaptureInterval);
    browserCaptureInterval = setInterval(async () => {
      if (isProcessing || !browserStream) return;
      isProcessing = true;
      try {
        ctx.drawImage(videoEl, 0, 0, 640, 480);
        const b64 = captureCanvas.toDataURL('image/jpeg', 0.7);
        const res = await fetch('/api/process_browser_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: b64 })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.annotated_image) streamImg.src = data.annotated_image;
          if (data.summary) {
            updateWorkerRoster(data.summary.workers || []);
            updateHUD(data.summary);
          }
        }
      } catch (e) { console.error('Frame error:', e); }
      finally { isProcessing = false; }
    }, 100);
  } catch (err) {
    alert('Camera access denied: ' + err.message);
    document.getElementById('camera-source-select').value = 'webcam';
    switchCameraSource('webcam');
  }
}

function stopBrowserWebcam() {
  if (browserCaptureInterval) { clearInterval(browserCaptureInterval); browserCaptureInterval = null; }
  if (browserStream) { browserStream.getTracks().forEach(t => t.stop()); browserStream = null; }
}

async function startCamera() {
  if (currentSource === 'browser') { startBrowserWebcam(); return; }
  try {
    const res = await fetch('/camera/start', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      const streamImg = document.getElementById('live-video-stream');
      if (streamImg) streamImg.src = '/video_feed?' + Date.now();
      document.getElementById('live-stream-badge').textContent = 'Online';
      document.getElementById('live-stream-badge').className = 'badge badge-green';
    }
  } catch (err) { alert('Camera error: ' + err.message); }
}

async function stopCamera() {
  if (currentSource === 'browser') {
    stopBrowserWebcam();
    document.getElementById('live-stream-badge').textContent = 'Stopped';
    document.getElementById('live-stream-badge').className = 'badge badge-yellow';
    return;
  }
  try {
    const res = await fetch('/camera/stop', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      document.getElementById('live-stream-badge').textContent = 'Stopped';
      document.getElementById('live-stream-badge').className = 'badge badge-yellow';
    }
  } catch (err) { console.error('Stop error:', err); }
}
