import React from "react";
import { ShieldCheck, Users, AlertOctagon, Cpu, TrendingUp } from "lucide-react";

export default function AnalyticsOverview({ telemetry }) {
  const complianceRate = telemetry?.compliance_rate_percent ?? 100;
  const activeWorkers = telemetry?.average_active_workers ?? 0;
  const totalViolations = telemetry?.total_violations ?? 0;
  const avgFps = telemetry?.average_fps ?? 0;
  const ppeBreakdown = telemetry?.violations_by_ppe || {
    helmet: 0,
    vest: 0,
    boots: 0,
    gloves: 0,
  };

  const ppeMax = Math.max(...Object.values(ppeBreakdown), 1);

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-5">
      <div className="flex items-center justify-between border-b border-slate-200 pb-3">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-4 h-4 text-blue-600" />
          <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Operational Safety KPIs &amp; Telemetry
          </h2>
        </div>
      </div>

      {/* Top 4 Real KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
        <div className="p-3.5 bg-slate-50 rounded border border-slate-200">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Compliance Rate</span>
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1 font-mono">
            {complianceRate}%
          </div>
        </div>

        <div className="p-3.5 bg-slate-50 rounded border border-slate-200">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Workers in Frame</span>
            <Users className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1 font-mono">
            {activeWorkers}
          </div>
        </div>

        <div className="p-3.5 bg-slate-50 rounded border border-slate-200">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Logged Violations</span>
            <AlertOctagon className="w-4 h-4 text-red-600" />
          </div>
          <div className="text-2xl font-bold text-red-600 mt-1 font-mono">
            {totalViolations}
          </div>
        </div>

        <div className="p-3.5 bg-slate-50 rounded border border-slate-200">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Pipeline Speed</span>
            <Cpu className="w-4 h-4 text-slate-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 mt-1 font-mono">
            {avgFps > 0 ? avgFps.toFixed(1) : "0.0"} <span className="text-xs text-slate-500">FPS</span>
          </div>
        </div>
      </div>

      {/* PPE Violation Breakdown */}
      <div className="pt-2">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wide mb-3">
          Missing PPE Violations Breakdown
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { id: "helmet", label: "⛑️ Helmet / Hard Hat", count: ppeBreakdown.helmet || 0 },
            { id: "vest", label: "🦺 High-Vis Vest", count: ppeBreakdown.vest || 0 },
            { id: "boots", label: "🥾 Safety Boots", count: ppeBreakdown.boots || 0 },
            { id: "gloves", label: "🧤 Protective Gloves", count: ppeBreakdown.gloves || 0 },
          ].map((item) => {
            const widthPct = Math.min(100, Math.round((item.count / ppeMax) * 100));

            return (
              <div key={item.id} className="p-3 bg-slate-50 rounded border border-slate-200 space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-700">{item.label}</span>
                  <span className="font-mono font-bold text-slate-900">{item.count}</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      item.count > 0 ? "bg-red-500" : "bg-emerald-500"
                    }`}
                    style={{ width: `${item.count > 0 ? Math.max(widthPct, 15) : 0}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
