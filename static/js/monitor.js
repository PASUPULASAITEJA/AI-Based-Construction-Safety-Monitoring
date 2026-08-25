/**
 * Live Monitoring Controller, Source Switcher, & Personnel Status Poller
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
      console.warn('Video stream disconnected, auto-reconnecting in 1s...');
      setTimeout(() => {
        if (currentSource !== 'browser') {
          streamImg.src = '/video_feed?t=' + Date.now();
        }
      }, 1000);
    };
  }
}

function startPollingStatus() {
  if (statusPollInterval) clearInterval(statusPollInterval);
  statusPollInterval = setInterval(fetchLiveStatus, 600);
}

async function fetchLiveStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      updateWorkerRoster(data.workers || []);
      updateMonitorHeader(data);
    }
  } catch (err) {
    // Graceful error ignore
  }
}

function updateMonitorHeader(data) {
  const countBadge = document.getElementById('worker-count-badge');
  if (countBadge) {
    countBadge.innerText = `${data.worker_count || 0} Active`;
  }
}

function updateWorkerRoster(workers) {
  const container = document.getElementById('worker-roster-list');
  if (!container) return;

  if (!workers || workers.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; color: var(--text-muted); padding: 40px 10px;">
        No workers currently detected in camera view.
      </div>
    `;
    return;
  }

  container.innerHTML = '';
  workers.forEach(w => {
    const card = document.createElement('div');
    const isViolation = w.violations && w.violations.length > 0;
    const isCritical = w.status === 'CRITICAL VIOLATION';

    card.className = `worker-card ${isViolation ? 'violation' : ''}`;

    let statusBadgeClass = 'badge-compliant';
    if (isCritical) statusBadgeClass = 'badge-critical';
    else if (isViolation) statusBadgeClass = 'badge-medium';
    else if (w.status === 'PENDING VERIFICATION') statusBadgeClass = 'badge-medium';

    card.innerHTML = `
      <div class="worker-card-header">
        <span class="worker-id-title">${w.label}</span>
        <span class="badge ${statusBadgeClass}">${w.status}</span>
      </div>
      <div class="worker-ppe-row">
        <span class="ppe-pill ${w.helmet ? 'ok' : 'no'}">
          ${w.helmet ? '🪖 Helmet ✓' : '❌ No Helmet'}
        </span>
        <span class="ppe-pill ${w.vest ? 'ok' : 'no'}">
          ${w.vest ? '🦺 Vest ✓' : '❌ No Vest'}
        </span>
      </div>
      <div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary); display: flex; justify-content: space-between;">
        <span>Zone: <strong>${w.zone_name || 'SAFE'}</strong></span>
        ${w.violations && w.violations.length > 0 ? `<span style="color: var(--color-danger); font-weight: 600;">⚠️ ${w.violations.join(', ')}</span>` : ''}
      </div>
    `;

    container.appendChild(card);
  });
}

async function switchCameraSource(source) {
  currentSource = source;
  stopBrowserWebcam();

  if (source === 'browser') {
    startBrowserWebcam();
  } else {
    // Send mode change to server
    try {
      await fetch('/camera/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: source })
      });
      const streamImg = document.getElementById('live-video-stream');
      if (streamImg) streamImg.src = '/video_feed?' + new Date().getTime();
      document.getElementById('live-stream-badge').innerText = source === 'simulator' ? 'Simulator Active' : 'Camera Online';
      document.getElementById('live-stream-badge').className = 'badge badge-compliant';
    } catch (err) {
      console.error('Mode switch error:', err);
    }
  }
}

async function startBrowserWebcam() {
  const videoEl = document.getElementById('browser-webcam-element');
  const streamImg = document.getElementById('live-video-stream');
  const captureCanvas = document.getElementById('browser-capture-canvas');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Browser WebRTC camera is not supported on this browser.');
    return;
  }

  try {
    browserStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    videoEl.srcObject = browserStream;
    await videoEl.play();

    document.getElementById('live-stream-badge').innerText = 'Browser WebRTC Active';
    document.getElementById('live-stream-badge').className = 'badge badge-compliant';

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
          if (data.annotated_image) {
            streamImg.src = data.annotated_image;
          }
          if (data.summary) {
            updateWorkerRoster(data.summary.workers || []);
            updateMonitorHeader(data.summary);
          }
        }
      } catch (e) {
        console.error('Browser frame error:', e);
      } finally {
        isProcessing = false;
      }
    }, 100); // 10 FPS streaming
  } catch (err) {
    alert('Could not access browser camera: ' + err.message);
    document.getElementById('camera-source-select').value = 'webcam';
    switchCameraSource('webcam');
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
      const streamImg = document.getElementById('live-video-stream');
      if (streamImg) streamImg.src = '/video_feed?' + new Date().getTime();
      document.getElementById('live-stream-badge').innerText = 'Camera Online';
      document.getElementById('live-stream-badge').className = 'badge badge-compliant';
    }
  } catch (err) {
    alert('Camera start error: ' + err.message);
  }
}

async function stopCamera() {
  if (currentSource === 'browser') {
    stopBrowserWebcam();
    document.getElementById('live-stream-badge').innerText = 'Camera Paused';
    document.getElementById('live-stream-badge').className = 'badge badge-medium';
    return;
  }

  try {
    const res = await fetch('/camera/stop', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      document.getElementById('live-stream-badge').innerText = 'Camera Paused';
      document.getElementById('live-stream-badge').className = 'badge badge-medium';
    }
  } catch (err) {
    console.error('Stop camera error:', err);
  }
}
