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
    const isViolation = (w.violations && w.violations.length > 0) || (w.compliance_score !== undefined && w.compliance_score < 0.8);
    const isCritical = w.status === 'CRITICAL VIOLATION';
    const scoreVal = w.compliance_score !== undefined ? w.compliance_score : 1.0;
    const scorePct = Math.round(scoreVal <= 1.0 ? scoreVal * 100 : scoreVal);

    card.className = `worker-card ${isViolation ? 'violation' : ''}`;
    card.style.cssText = "background: #1e293b; border-radius: 8px; padding: 12px; margin-bottom: 12px; border: 1px solid " + (scorePct < 70 ? "#ef4444" : "#334155") + ";";

    let scoreBadgeColor = scorePct >= 80 ? '#22c55e' : scorePct >= 50 ? '#f59e0b' : '#ef4444';

    let ppeGridHtml = '<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: #0f172a; padding: 8px; border-radius: 6px; margin: 8px 0;">';
    PPE_CONFIG.forEach(item => {
      let val = w[item.key];
      if (val === undefined && item.altKey && w[item.altKey] !== undefined) {
        val = w[item.altKey];
      }
      ppeGridHtml += `
        <div style="display: flex; flex-direction: column; align-items: center; font-size: 11px; padding: 3px 0;">
          <span style="font-size: 13px;">${item.icon}</span>
          <span style="color: #94a3b8; font-size: 9px;">${item.label}</span>
          <div>${renderPPEBadge(val)}</div>
        </div>
      `;
    });
    ppeGridHtml += '</div>';

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 14px; color: #f8fafc;">${w.label || w.worker_id}</span>
        <span style="font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: ${scoreBadgeColor}22; color: ${scoreBadgeColor}; border: 1px solid ${scoreBadgeColor}66;">
          ${scorePct}% Compliance
        </span>
      </div>
      ${ppeGridHtml}
      <div style="font-size: 11px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
        <span>Zone: <strong style="color: #cbd5e1;">${w.zone_name || 'General Site'}</strong></span>
        ${w.missing_ppe && w.missing_ppe.length > 0 ? `<span style="color: #f87171; font-weight: 600;">⚠️ ${w.missing_ppe.slice(0, 2).join(', ')}</span>` : ''}
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
window.switchCameraSource = switchCameraSource;
window.changeCameraSource = switchCameraSource;

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
