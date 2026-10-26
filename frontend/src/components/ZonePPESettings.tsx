import React, { useState } from "react";

export interface ZoneConfig {
  zone_id: string;
  zone_name: string;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  description?: string;
  mandatory_ppe: string[];
}

const ALL_PPE = [
  { id: "helmet", label: "Hard Hat / Helmet", icon: "🪖" },
  { id: "goggles", label: "Eye Protection / Goggles", icon: "🥽" },
  { id: "earmuff", label: "Hearing Protection / Earmuff", icon: "🎧" },
  { id: "vest", label: "High-Vis Safety Vest", icon: "🦺" },
  { id: "safety_pants", label: "Safety Pants / Chaps", icon: "👖" },
  { id: "gloves", label: "Heavy Duty Work Gloves", icon: "🧤" },
  { id: "boots", label: "Steel-Toe Safety Boots", icon: "🥾" },
  { id: "harness", label: "Fall Arrest Safety Harness", icon: "🪢" }
];

export const ZonePPESettings: React.FC = () => {
  const [zones, setZones] = useState<ZoneConfig[]>([
    {
      zone_id: "general_site",
      zone_name: "General Construction Area",
      risk_level: "MEDIUM",
      description: "Standard ground site safety boundary",
      mandatory_ppe: ["helmet", "vest"]
    },
    {
      zone_id: "chemical_zone",
      zone_name: "Hazardous Chemical Handling",
      risk_level: "CRITICAL",
      description: "Chemical storage and pressurized solvent wash area",
      mandatory_ppe: ["helmet", "vest", "goggles", "gloves"]
    },
    {
      zone_id: "heights_zone",
      zone_name: "Working at Heights / Scaffolding",
      risk_level: "CRITICAL",
      description: "Areas exceeding 2 meters elevation requiring fall arrest",
      mandatory_ppe: ["helmet", "vest", "harness"]
    },
    {
      zone_id: "machinery_zone",
      zone_name: "Heavy Machinery & Fabrication",
      risk_level: "HIGH",
      description: "Operating heavy cranes, earth movers, and high-decibel tools",
      mandatory_ppe: ["helmet", "vest", "earmuff", "boots"]
    }
  ]);

  const [saving, setSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const togglePPE = (zoneId: string, ppeId: string) => {
    setZones(prev =>
      prev.map(z => {
        if (z.zone_id !== zoneId) return z;
        const exists = z.mandatory_ppe.includes(ppeId);
        return {
          ...z,
          mandatory_ppe: exists
            ? z.mandatory_ppe.filter(id => id !== ppeId)
            : [...z.mandatory_ppe, ppeId]
        };
      })
    );
    setSavedSuccess(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch("/api/zones/rules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: "1.0.0", zones })
      });
      if (response.ok) {
        setSavedSuccess(true);
        setTimeout(() => setSavedSuccess(false), 3000);
      }
      setSaving(false);
    } catch (err) {
      setSaving(false);
      alert("Failed to save rules to backend API.");
    }
  };

  return (
    <div style={{ background: "#0f172a", color: "#f8fafc", padding: "24px", borderRadius: "12px", fontFamily: "system-ui, sans-serif" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <div>
          <h2 style={{ margin: "0 0 6px 0", fontSize: "22px", fontWeight: "700" }}>Site Zone PPE Requirement Rules</h2>
          <p style={{ margin: 0, color: "#94a3b8", fontSize: "14px" }}>Configure mandatory PPE categories enforced per zone</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {savedSuccess && <span style={{ color: "#4ade80", fontSize: "13px", fontWeight: "600" }}>✓ Saved Successfully</span>}
          <button
            onClick={handleSave}
            disabled={saving}
            style={{
              background: "#3b82f6",
              color: "#fff",
              border: "none",
              padding: "10px 20px",
              borderRadius: "8px",
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            {saving ? "Saving..." : "Save Configuration"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gap: "16px" }}>
        {zones.map(zone => (
          <div
            key={zone.zone_id}
            style={{
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "10px",
              padding: "18px"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
              <div>
                <span style={{ fontSize: "16px", fontWeight: "600" }}>{zone.zone_name}</span>
                <span style={{ marginLeft: "10px", fontSize: "12px", color: "#64748b" }}>ID: {zone.zone_id}</span>
                {zone.description && <p style={{ margin: "4px 0 0 0", fontSize: "12px", color: "#94a3b8" }}>{zone.description}</p>}
              </div>
              <span
                style={{
                  fontSize: "11px",
                  fontWeight: "700",
                  padding: "4px 10px",
                  borderRadius: "20px",
                  background: zone.risk_level === "CRITICAL" ? "#ef444422" : zone.risk_level === "HIGH" ? "#f9731622" : "#3b82f622",
                  color: zone.risk_level === "CRITICAL" ? "#ef4444" : zone.risk_level === "HIGH" ? "#f97316" : "#60a5fa"
                }}
              >
                {zone.risk_level} RISK
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px" }}>
              {ALL_PPE.map(ppe => {
                const checked = zone.mandatory_ppe.includes(ppe.id);
                return (
                  <button
                    key={ppe.id}
                    type="button"
                    onClick={() => togglePPE(zone.zone_id, ppe.id)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      padding: "8px 12px",
                      borderRadius: "6px",
                      border: checked ? "1px solid #22c55e" : "1px solid #334155",
                      background: checked ? "#14532d33" : "#0f172a",
                      color: checked ? "#4ade80" : "#94a3b8",
                      cursor: "pointer",
                      fontSize: "13px"
                    }}
                  >
                    <span>{ppe.icon}</span>
                    <span style={{ flex: 1, textAlign: "left" }}>{ppe.label}</span>
                    <span>{checked ? "✓" : "○"}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ZonePPESettings;
