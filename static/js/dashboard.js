/**
 * Dashboard live status poller
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
      const complianceCard = document.getElementById('card-compliance-rate');
      const violationsCard = document.getElementById('card-violations-count');
      const criticalCard = document.getElementById('card-critical-count');

      if (workersCard) workersCard.innerText = data.worker_count || 0;
      if (complianceCard) {
        const total = data.worker_count || 0;
        const viols = data.violation_count || 0;
        const rate = total > 0 ? Math.round(((total - viols) / total) * 100) : 100;
        complianceCard.innerText = `${rate}%`;
      }
      if (violationsCard) violationsCard.innerText = data.violation_count || 0;
      if (criticalCard) criticalCard.innerText = data.critical_count || 0;
    }
  } catch (err) {
    // Graceful silent ignore on network hiccups
  }
}
