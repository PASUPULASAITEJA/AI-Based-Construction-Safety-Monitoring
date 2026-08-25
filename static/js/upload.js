/**
 * Upload Studio Controller for Image and Video Safety Analysis
 */

let selectedImageFile = null;
let selectedVideoFile = null;

function handleImageSelection(e) {
  const file = e.target.files[0];
  if (file) {
    selectedImageFile = file;
    const label = document.getElementById('image-selected-label');
    label.innerText = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    label.style.display = 'block';
    document.getElementById('btn-process-image').disabled = false;
  }
}

function handleVideoSelection(e) {
  const file = e.target.files[0];
  if (file) {
    selectedVideoFile = file;
    const label = document.getElementById('video-selected-label');
    label.innerText = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    label.style.display = 'block';
    document.getElementById('btn-process-video').disabled = false;
  }
}

async function submitImageAnalysis() {
  if (!selectedImageFile) return;

  const formData = new FormData();
  formData.append('image', selectedImageFile);

  showProgress('Running YOLOv8 & PPE Association on Image...', 30);

  try {
    const res = await fetch('/upload/image', {
      method: 'POST',
      body: formData
    });

    showProgress('Generating Visual Overlays...', 80);

    const data = await res.json();
    if (data.success) {
      showProgress('Completed!', 100);
      setTimeout(() => {
        hideProgress();
        renderImageResults(data);
      }, 400);
    } else {
      hideProgress();
      alert(data.message || 'Image processing error.');
    }
  } catch (err) {
    hideProgress();
    alert('Error processing image: ' + err.message);
  }
}

async function submitVideoAnalysis() {
  if (!selectedVideoFile) return;

  const formData = new FormData();
  formData.append('video', selectedVideoFile);

  showProgress('Uploading & Initializing Sequential Video Processor...', 20);

  try {
    const res = await fetch('/upload/video', {
      method: 'POST',
      body: formData
    });

    showProgress('Analyzing Video Frames (YOLO + ByteTrack + Rules)...', 70);

    const data = await res.json();
    if (data.success) {
      showProgress('Video Processing Complete!', 100);
      setTimeout(() => {
        hideProgress();
        renderVideoResults(data);
      }, 500);
    } else {
      hideProgress();
      alert(data.message || 'Video processing error.');
    }
  } catch (err) {
    hideProgress();
    alert('Error processing video: ' + err.message);
  }
}

function renderImageResults(data) {
  const container = document.getElementById('results-display-container');
  const imgEl = document.getElementById('result-annotated-img');
  const vidEl = document.getElementById('result-annotated-video');
  const downloadBtn = document.getElementById('btn-download-annotated');

  container.style.display = 'block';
  vidEl.style.display = 'none';
  imgEl.style.display = 'block';
  imgEl.src = data.annotated_url + '?' + new Date().getTime();

  if (downloadBtn) {
    downloadBtn.href = data.annotated_url;
    downloadBtn.download = 'safety_inspection_annotated.jpg';
    downloadBtn.style.display = 'inline-flex';
  }

  document.getElementById('result-workers-count').innerText = data.stats.worker_count || 0;
  document.getElementById('result-violations-count').innerText = data.stats.violation_count || 0;

  renderWorkersTable(data.stats.workers || []);
  container.scrollIntoView({ behavior: 'smooth' });
}

function renderVideoResults(data) {
  const container = document.getElementById('results-display-container');
  const imgEl = document.getElementById('result-annotated-img');
  const vidEl = document.getElementById('result-annotated-video');
  const downloadBtn = document.getElementById('btn-download-annotated');

  container.style.display = 'block';
  imgEl.style.display = 'none';
  vidEl.style.display = 'block';
  vidEl.src = data.output_video_url + '?' + new Date().getTime();

  if (downloadBtn) {
    downloadBtn.href = data.output_video_url;
    downloadBtn.download = 'safety_inspection_video.mp4';
    downloadBtn.style.display = 'inline-flex';
  }

  document.getElementById('result-workers-count').innerText = data.stats.total_workers || 0;
  document.getElementById('result-violations-count').innerText = data.stats.total_violations || 0;

  renderWorkersTable(data.stats.sample_workers || []);
  container.scrollIntoView({ behavior: 'smooth' });
}

function renderWorkersTable(workers) {
  const tbody = document.getElementById('result-workers-table-body');
  tbody.innerHTML = '';

  if (!workers || workers.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No workers detected in this media.</td></tr>';
    return;
  }

  workers.forEach(w => {
    const isViolation = w.violations && w.violations.length > 0;
    const isCritical = w.status === 'CRITICAL VIOLATION';
    const tr = document.createElement('tr');

    let statusBadge = `<span class="badge badge-compliant">✓ COMPLIANT</span>`;
    if (isCritical) {
      statusBadge = `<span class="badge badge-critical">🚨 CRITICAL VIOLATION</span>`;
    } else if (isViolation) {
      statusBadge = `<span class="badge badge-critical">⚠️ VIOLATION</span>`;
    }

    const violationDetails = isViolation 
      ? `<div style="font-size: 11px; color: var(--color-danger); margin-top: 4px; font-weight: 500;">
           ${w.violations.map(v => typeof v === 'object' ? (v.description || v.type) : v).join(' • ')}
         </div>`
      : '';

    tr.innerHTML = `
      <td>
        <strong>${w.label || 'Worker #' + w.worker_id}</strong>
        ${w.zone_name && w.zone_name !== 'SAFE' ? `<div style="font-size: 11px; color: var(--color-warning);">Zone: ${w.zone_name}</div>` : ''}
      </td>
      <td><span class="ppe-pill ${w.helmet ? 'ok' : 'no'}">${w.helmet ? '🪖 YES' : '❌ NO'}</span></td>
      <td><span class="ppe-pill ${w.vest ? 'ok' : 'no'}">${w.vest ? '🦺 YES' : '❌ NO'}</span></td>
      <td>
        ${statusBadge}
        ${violationDetails}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function showProgress(title, pct) {
  const pContainer = document.getElementById('upload-progress-container');
  pContainer.style.display = 'block';
  document.getElementById('upload-status-title').innerText = title;
  document.getElementById('upload-status-pct').innerText = `${pct}%`;
  document.getElementById('upload-progress-bar').style.width = `${pct}%`;
}

function hideProgress() {
  document.getElementById('upload-progress-container').style.display = 'none';
}

function resetUploadView() {
  document.getElementById('results-display-container').style.display = 'none';
  document.getElementById('image-selected-label').style.display = 'none';
  document.getElementById('video-selected-label').style.display = 'none';
  document.getElementById('btn-process-image').disabled = true;
  document.getElementById('btn-process-video').disabled = true;
  selectedImageFile = null;
  selectedVideoFile = null;
}
