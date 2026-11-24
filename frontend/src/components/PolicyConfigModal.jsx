import React, { useState, useEffect } from "react";
import { X, Sliders, CheckCircle2, Shield } from "lucide-react";

export default function PolicyConfigModal({ isOpen, onClose, zones, onUpdatePolicy }) {
  const [selectedZoneId, setSelectedZoneId] = useState(zones[0]?.zone_id || "active_work_zone");
  const currentZone = zones.find((z) => z.zone_id === selectedZoneId) || zones[0];

  const [requiredGear, setRequiredGear] = useState(
    Array.isArray(currentZone?.required_ppe) ? currentZone.required_ppe : []
  );
  const [nFrames, setNFrames] = useState(currentZone?.n_frame_confirm || 5);
  const [cooldown, setCooldown] = useState(currentZone?.alert_cooldown_seconds || 30);
  const [savedSuccess, setSavedSuccess] = useState(false);

  useEffect(() => {
    if (zones && zones.length > 0) {
      const z = zones.find((item) => item.zone_id === selectedZoneId) || zones[0];
      if (z) {
        setSelectedZoneId(z.zone_id);
        setRequiredGear(Array.isArray(z.required_ppe) ? z.required_ppe : []);
        setNFrames(z.n_frame_confirm || 5);
        setCooldown(z.alert_cooldown_seconds || 30);
      }
    }
  }, [isOpen, zones]);

  if (!isOpen) return null;

  const handleZoneChange = (zoneId) => {
    setSelectedZoneId(zoneId);
    const z = zones.find((item) => item.zone_id === zoneId);
    if (z) {
      setRequiredGear(Array.isArray(z.required_ppe) ? z.required_ppe : []);
      setNFrames(z.n_frame_confirm || 5);
      setCooldown(z.alert_cooldown_seconds || 30);
    }
  };

  const handleToggleGear = (gear) => {
    if (requiredGear.includes(gear)) {
      setRequiredGear(requiredGear.filter((g) => g !== gear));
    } else {
      setRequiredGear([...requiredGear, gear]);
    }
  };

  const handleSave = async () => {
    await onUpdatePolicy(selectedZoneId, {
      required_ppe: requiredGear,
      n_frame_confirm: parseInt(nFrames),
      alert_cooldown_seconds: parseInt(cooldown),
    });
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 1200);
  };

  const allAvailableGear = [
    { id: "helmet", label: "Hard Hat / Safety Helmet", desc: "Head Protection" },
    { id: "vest", label: "High-Visibility Safety Vest", desc: "Torso Visibility" },
    { id: "boots", label: "Steel-Toe Footwear", desc: "Impact & Puncture Safety" },
    { id: "gloves", label: "Protective Work Gloves", desc: "Hand & Cut Resistance" },
    { id: "goggles", label: "Eye Protection / Goggles", desc: "Splash & Particle Hazard" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-xs p-4">
      <div className="bg-white rounded-lg shadow-2xl border border-slate-300 max-w-lg w-full overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center space-x-2">
            <Sliders className="w-5 h-5 text-blue-600" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Zone Safety Compliance Rules
              </h3>
              <p className="text-xs text-slate-500">
                Configure required PPE items and temporal confirmation filters
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4 text-xs">
          {/* Zone Selector */}
          <div>
            <label className="font-semibold text-slate-700 block mb-1">
              Select Operating Zone:
            </label>
            <select
              value={selectedZoneId}
              onChange={(e) => handleZoneChange(e.target.value)}
              className="w-full bg-white border border-slate-300 rounded p-2 text-slate-800 font-medium focus:ring-1 focus:ring-blue-500 focus:outline-none"
            >
              {zones.map((z) => (
                <option key={z.zone_id} value={z.zone_id}>
                  {z.name} (Risk Level: {z.risk_level})
                </option>
              ))}
            </select>
          </div>

          {/* Required PPE Checkboxes */}
          <div>
            <label className="font-semibold text-slate-700 block mb-1.5">
              Mandatory Required Equipment (OSHA / Plant Standard):
            </label>
            <div className="space-y-1.5">
              {allAvailableGear.map((g) => {
                const checked = requiredGear.includes(g.id);
                return (
                  <label
                    key={g.id}
                    className={`flex items-center justify-between p-2 rounded border cursor-pointer transition-colors ${
                      checked
                        ? "bg-blue-50/70 border-blue-200 text-slate-900"
                        : "bg-white border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center space-x-2.5">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => handleToggleGear(g.id)}
                        className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 h-4 w-4"
                      />
                      <div>
                        <div className="font-medium text-slate-800">{g.label}</div>
                        <div className="text-[10px] text-slate-500">{g.desc}</div>
                      </div>
                    </div>
                    {checked && (
                      <span className="text-[10px] font-bold text-blue-700 uppercase">
                        Enforced
                      </span>
                    )}
                  </label>
                );
              })}
            </div>
          </div>

          {/* Sliders: N-Frame confirmation and alert cooldown */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="p-3 bg-slate-50 rounded border border-slate-200 space-y-1">
              <div className="flex items-center justify-between font-medium">
                <span className="text-slate-700">Confirmation (N):</span>
                <span className="font-mono font-bold text-blue-700">{nFrames} Frames</span>
              </div>
              <input
                type="range"
                min="1"
                max="60"
                value={nFrames}
                onChange={(e) => setNFrames(e.target.value)}
                className="w-full accent-blue-600 cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">
                Worker must miss gear for N consecutive frames before siren triggers.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded border border-slate-200 space-y-1">
              <div className="flex items-center justify-between font-medium">
                <span className="text-slate-700">Cooldown:</span>
                <span className="font-mono font-bold text-blue-700">{cooldown}s</span>
              </div>
              <input
                type="range"
                min="5"
                max="60"
                step="5"
                value={cooldown}
                onChange={(e) => setCooldown(e.target.value)}
                className="w-full accent-blue-600 cursor-pointer"
              />
              <p className="text-[10px] text-slate-500">
                Minimum delay before logging another violation for same worker.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="text-xs text-emerald-600 font-medium">
            {savedSuccess && "Policy updated and broadcast to edge nodes!"}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-1.5 rounded text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white shadow-sm"
            >
              Save Policy Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

