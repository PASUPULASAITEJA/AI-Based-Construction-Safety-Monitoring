/**
 * Dashboard live status poller
 *
 * Only updates live-camera elements. DB-backed KPIs (today's violations,
 * compliance rate) are rendered server-side and must not be overwritten
 * with the transient /api/status snapshot.
 */

document.addEventListener('DOMContentLoaded', () => {
  setInterval(pollDashboardStats, 1000);
});

async function pollDashboardStats() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();

      const workersCard = document.getElementById('card-workers-count');
      const liveStatus = document.getElementById('card-live-status');
      const criticalCard = document.getElementById('card-critical-count');

      const total = data.worker_count || 0;
      const viols = data.violation_count || 0;

      if (workersCard) workersCard.textContent = total;
      if (liveStatus) {
        // Live-only indicator; no compliance claim when nobody is in view
        liveStatus.textContent = total > 0 ? ` \u00b7 ${viols} live violation${viols === 1 ? '' : 's'}` : '';
      }
      if (criticalCard) criticalCard.textContent = data.critical_count || 0;
    }
  } catch (err) {
    // Graceful silent ignore on network hiccups
  }
}
