/**
 * Common utilities and snapshot modal logic
 */

function viewSnapshot(imageUrl, incidentCode, violationType) {
  const modal = document.getElementById('snapshot-modal');
  const modalImg = document.getElementById('modal-snapshot-img');
  const modalTitle = document.getElementById('modal-incident-title');
  const modalMeta = document.getElementById('modal-incident-meta');

  if (modal && modalImg) {
    modalImg.src = imageUrl;
    modalTitle.innerText = `Incident: ${incidentCode}`;
    modalMeta.innerText = `Type: ${violationType} | Recorded from live analysis`;
    modal.classList.add('active');
  }
}

function closeSnapshotModal(e) {
  const modal = document.getElementById('snapshot-modal');
  if (modal) {
    modal.classList.remove('active');
  }
}

// Close on Escape key
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    closeSnapshotModal();
  }
});
