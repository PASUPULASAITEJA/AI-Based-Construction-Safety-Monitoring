import React from 'react';

/**
 * 8 Standard PPE Category Definitions
 */
export const PPE_ITEMS = [
  { key: 'helmet', label: 'Helmet / Hard Hat', icon: '🪖' },
  { key: 'goggles', label: 'Safety Goggles', icon: '🥽' },
  { key: 'earmuff', label: 'Ear Protection', icon: '🎧' },
  { key: 'vest', label: 'Safety Vest', icon: '🦺' },
  { key: 'harness', label: 'Safety Harness', icon: '🪢' },
  { key: 'safety_pants', label: 'Safety Pants', icon: '👖' },
  { key: 'gloves', label: 'Work Gloves', icon: '🧤' },
  { key: 'boots', label: 'Safety Boots', icon: '🥾' }
];

export interface WorkerPPEData {
  worker_id: string;
  label?: string;
  zone_name?: string;
  confidence?: number;
  helmet: boolean | 'uncertain';
  goggles: boolean | 'uncertain';
  earmuff: boolean | 'uncertain';
  vest: boolean | 'uncertain';
  harness: boolean | 'uncertain';
  safety_pants: boolean | 'uncertain';
  gloves: boolean | 'uncertain';
  boots: boolean | 'uncertain';
  compliance_score: number; // 0.0 to 1.0 or 0 to 100
  missing_ppe?: string[];
  last_seen?: string;
}

/**
 * Per-Worker Real-Time Compliance Card
 */
export const WorkerPPECard: React.FC<{ worker: WorkerPPEData }> = ({ worker }) => {
  const scorePercent = Math.round(
    worker.compliance_score <= 1.0
      ? worker.compliance_score * 100
      : worker.compliance_score
  );

  const getScoreTheme = (score: number) => {
    if (score >= 80) return { bg: '#052e16', text: '#4ade80', border: '#15803d' };
    if (score >= 50) return { bg: '#451a03', text: '#fb923c', border: '#b45309' };
    return { bg: '#450a0a', text: '#f87171', border: '#b91c1c' };
  };

  const theme = getScoreTheme(scorePercent);

  const renderStatusBadge = (status: boolean | 'uncertain') => {
    if (status === true) {
      return (
        <span
          title="Compliant (Detected)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            background: '#16a34a22',
            color: '#22c55e',
            fontWeight: 'bold',
            fontSize: '12px',
            border: '1px solid #16a34a55'
          }}
        >
          ✓
        </span>
      );
    } else if (status === 'uncertain') {
      return (
        <span
          title="Uncertain (Low Confidence / Occluded)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            background: '#47556922',
            color: '#94a3b8',
            fontWeight: 'bold',
            fontSize: '12px',
            border: '1px solid #47556955'
          }}
        >
          ?
        </span>
      );
    } else {
      return (
        <span
          title="Missing (Violation)"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            background: '#dc262622',
            color: '#ef4444',
            fontWeight: 'bold',
            fontSize: '12px',
            border: '1px solid #dc262655'
          }}
        >
          ✕
        </span>
      );
    }
  };

  return (
    <div
      style={{
        background: '#1e293b',
        border: `1px solid ${scorePercent < 80 ? theme.border : '#334155'}`,
        borderRadius: '10px',
        padding: '14px',
        marginBottom: '12px',
        color: '#f8fafc',
        boxShadow: scorePercent < 50 ? '0 0 12px rgba(239,68,68,0.2)' : 'none',
        transition: 'all 0.2s ease-in-out'
      }}
    >
      {/* Header: ID, Zone & Score */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '15px', fontWeight: '700' }}>
              {worker.label || worker.worker_id}
            </span>
            {worker.zone_name && (
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: '600',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  background: '#334155',
                  color: '#93c5fd'
                }}
              >
                {worker.zone_name}
              </span>
            )}
          </div>
          <span style={{ fontSize: '11px', color: '#64748b' }}>Anchor: {worker.worker_id}</span>
        </div>

        <div
          style={{
            background: theme.bg,
            color: theme.text,
            border: `1px solid ${theme.border}`,
            padding: '4px 10px',
            borderRadius: '20px',
            fontWeight: '700',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}
        >
          <span>{scorePercent}%</span>
        </div>
      </div>

      {/* 8 PPE Grid Checklist */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '8px',
          background: '#0f172a',
          padding: '10px',
          borderRadius: '8px',
          border: '1px solid #1e293b'
        }}
      >
        {PPE_ITEMS.map(item => {
          const status = worker[item.key as keyof WorkerPPEData] as boolean | 'uncertain';
          return (
            <div
              key={item.key}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '4px',
                padding: '6px 4px',
                borderRadius: '6px',
                background: '#1e293b66'
              }}
            >
              <span style={{ fontSize: '16px' }}>{item.icon}</span>
              <span style={{ fontSize: '10px', color: '#94a3b8', textAlign: 'center', lineHeight: '1.1' }}>
                {item.label.split('/')[0].trim()}
              </span>
              <div>{renderStatusBadge(status)}</div>
            </div>
          );
        })}
      </div>

      {/* Missing Alerts Footer if any */}
      {worker.missing_ppe && worker.missing_ppe.length > 0 && (
        <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
          <span style={{ fontSize: '11px', color: '#ef4444', fontWeight: '600' }}>Missing:</span>
          {worker.missing_ppe.map(missing => (
            <span
              key={missing}
              style={{
                fontSize: '10px',
                padding: '2px 6px',
                borderRadius: '4px',
                background: '#7f1d1d44',
                color: '#fca5a5',
                border: '1px solid #ef444444'
              }}
            >
              {missing.replace('_', ' ')}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Site-Wide PPE Category Compliance Breakdown
 */
export const SiteComplianceBreakdown: React.FC<{ workers: WorkerPPEData[] }> = ({ workers }) => {
  const total = workers.length;

  const calculateRates = () => {
    if (total === 0) return {};
    const counts: Record<string, number> = {};
    PPE_ITEMS.forEach(p => {
      counts[p.key] = workers.filter(w => w[p.key as keyof WorkerPPEData] === true).length;
    });
    return counts;
  };

  const counts = calculateRates();

  return (
    <div
      style={{
        background: '#1e293b',
        border: '1px solid #334155',
        borderRadius: '10px',
        padding: '16px',
        color: '#f8fafc'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <h4 style={{ margin: 0, fontSize: '15px', fontWeight: '700' }}>Site-Wide PPE Adherence</h4>
        <span style={{ fontSize: '12px', color: '#94a3b8' }}>{total} Personnel Active</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {PPE_ITEMS.map(item => {
          const count = counts[item.key] || 0;
          const pct = total > 0 ? Math.round((count / total) * 100) : 0;
          const barColor = pct >= 85 ? '#22c55e' : pct >= 60 ? '#f59e0b' : '#ef4444';

          return (
            <div key={item.key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span>{item.icon}</span>
                  <span style={{ color: '#cbd5e1' }}>{item.label}</span>
                </span>
                <span style={{ fontWeight: '600', color: barColor }}>
                  {pct}% ({count}/{total})
                </span>
              </div>
              <div
                style={{
                  height: '6px',
                  background: '#0f172a',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: '100%',
                    background: barColor,
                    borderRadius: '3px',
                    transition: 'width 0.4s ease-out'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
