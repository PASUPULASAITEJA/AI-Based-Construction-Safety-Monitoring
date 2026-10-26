/**
 * Interactive HTML5 Canvas Polygon Zone Drawer
 */

let isDrawing = false;
let currentPoints = [];
let savedZones = [];
let canvas = null;
let ctx = null;

document.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('zone-drawing-canvas');
  if (!canvas) return;
  ctx = canvas.getContext('2d');

  function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    redrawCanvas();
  }

  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 200);

  canvas.addEventListener('click', handleCanvasClick);
  canvas.addEventListener('mousemove', handleCanvasMouseMove);
  loadSavedZones();
});

function toggleDrawMode() {
  isDrawing = !isDrawing;
  const btn = document.getElementById('btn-draw-mode');
  const saveBtn = document.getElementById('btn-save-zone');

  if (isDrawing) {
    currentPoints = [];
    btn.classList.remove('btn-secondary');
    btn.classList.add('btn-primary');
    btn.innerText = '🛑 Finish Polygon';
    saveBtn.disabled = true;
  } else {
    btn.classList.remove('btn-primary');
    btn.classList.add('btn-secondary');
    btn.innerText = '✏️ New Polygon';
    if (currentPoints.length >= 3) {
      saveBtn.disabled = false;
    }
  }
  redrawCanvas();
}

function handleCanvasClick(e) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round(e.clientX - rect.left);
  const y = Math.round(e.clientY - rect.top);

  currentPoints.push([x, y]);
  if (currentPoints.length >= 3) {
    document.getElementById('btn-save-zone').disabled = false;
  }
  redrawCanvas();
}

function handleCanvasMouseMove(e) {
  if (!isDrawing || currentPoints.length === 0) return;
  const rect = canvas.getBoundingClientRect();
  const mouseX = Math.round(e.clientX - rect.left);
  const mouseY = Math.round(e.clientY - rect.top);

  redrawCanvas();

  // Draw guide line from last point to mouse
  const lastPt = currentPoints[currentPoints.length - 1];
  ctx.strokeStyle = 'rgba(0, 210, 255, 0.7)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(lastPt[0], lastPt[1]);
  ctx.lineTo(mouseX, mouseY);
  ctx.stroke();
  ctx.setLineDash([]);
}

function clearActiveDrawing() {
  isDrawing = false;
  currentPoints = [];
  document.getElementById('btn-draw-mode').innerText = '✏️ New Polygon';
  document.getElementById('btn-draw-mode').classList.remove('btn-primary');
  document.getElementById('btn-draw-mode').classList.add('btn-secondary');
  document.getElementById('btn-save-zone').disabled = true;
  redrawCanvas();
}

function redrawCanvas() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / 640;
  const scaleY = canvas.height / 480;

  // Draw saved zones
  savedZones.forEach(z => {
    if (!z.coordinates || z.coordinates.length < 3) return;

    ctx.beginPath();
    const startX = z.coordinates[0][0] * scaleX;
    const startY = z.coordinates[0][1] * scaleY;
    ctx.moveTo(startX, startY);
    for (let i = 1; i < z.coordinates.length; i++) {
      ctx.lineTo(z.coordinates[i][0] * scaleX, z.coordinates[i][1] * scaleY);
    }
    ctx.closePath();

    if (z.zone_type === 'HAZARD') {
      ctx.fillStyle = 'rgba(255, 42, 81, 0.25)';
      ctx.strokeStyle = '#ff2a51';
    } else if (z.zone_type === 'RESTRICTED') {
      ctx.fillStyle = 'rgba(227, 98, 9, 0.25)';
      ctx.strokeStyle = '#e36209';
    } else {
      ctx.fillStyle = 'rgba(46, 160, 67, 0.25)';
      ctx.strokeStyle = '#2ea043';
    }

    ctx.fill();
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw label
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText(`[${z.zone_type}] ${z.name}`, startX + 6, Math.max(16, startY - 6));
  });

  // Draw active drawing in-progress
  if (currentPoints.length > 0) {
    ctx.beginPath();
    ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
    for (let i = 1; i < currentPoints.length; i++) {
      ctx.lineTo(currentPoints[i][0], currentPoints[i][1]);
    }

    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw vertex dots
    currentPoints.forEach((pt, idx) => {
      ctx.fillStyle = idx === 0 ? '#2ea043' : '#00d2ff';
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

async function loadSavedZones() {
  try {
    const res = await fetch('/api/zones');
    if (res.ok) {
      savedZones = await res.json();
      renderActiveZoneBadges();
      redrawCanvas();
    }
  } catch (err) {
    console.error('Error loading zones:', err);
  }
}

function renderActiveZoneBadges() {
  const container = document.getElementById('active-zones-container');
  if (!container) return;
  container.innerHTML = '';

  if (savedZones.length === 0) {
    container.innerHTML = '<span style="font-size: 12px; color: var(--text-muted);">No active restricted zones created. Click "New Polygon" above.</span>';
    return;
  }

  savedZones.forEach(z => {
    const badge = document.createElement('div');
    badge.className = `badge badge-${z.zone_type === 'HAZARD' ? 'critical' : (z.zone_type === 'RESTRICTED' ? 'high' : 'compliant')}`;
    badge.style.display = 'inline-flex';
    badge.style.alignItems = 'center';
    badge.style.gap = '6px';
    badge.innerHTML = `
      <span>[${z.zone_type}] ${z.name}</span>
      <button onclick="deleteZone(${z.id})" style="background:none; border:none; color:inherit; cursor:pointer; font-size:12px;">&times;</button>
    `;
    container.appendChild(badge);
  });
}

async function saveCurrentZone() {
  if (currentPoints.length < 3) return;

  const zType = document.getElementById('zone-type-select').value;
  const zNameInput = document.getElementById('zone-name-input');
  const zName = zNameInput.value.trim() || `${zType} Zone ${savedZones.length + 1}`;

  const scaleX = 640 / canvas.width;
  const scaleY = 480 / canvas.height;
  const normalizedCoords = currentPoints.map(pt => [
    Math.round(pt[0] * scaleX),
    Math.round(pt[1] * scaleY)
  ]);

  const payload = {
    name: zName,
    zone_type: zType,
    coordinates: normalizedCoords
  };

  try {
    const res = await fetch('/zones', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      zNameInput.value = '';
      clearActiveDrawing();
      await loadSavedZones();
    } else {
      alert('Failed to save zone.');
    }
  } catch (err) {
    console.error('Save zone error:', err);
  }
}

async function deleteZone(zoneId) {
  if (!confirm('Are you sure you want to delete this zone?')) return;
  try {
    const res = await fetch(`/zones/${zoneId}`, { method: 'DELETE' });
    if (res.ok) {
      await loadSavedZones();
    }
  } catch (err) {
    console.error('Delete zone error:', err);
  }
}
