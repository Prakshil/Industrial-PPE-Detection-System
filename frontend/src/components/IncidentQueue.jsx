import React from "react";
import {
  AlertTriangle,
  FileSearch,
  Clock,
  Trash2,
  ShieldCheck,
  UserX,
} from "lucide-react";

export default function IncidentQueue({
  violations = [],
  onDeleteViolation,
  onClearAllViolations,
  onSelectViolationForEvidence,
  onTriageViolation,
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm flex flex-col h-full overflow-hidden">
      {/* Top Header Bar */}
      <div className="px-4 py-3 border-b border-slate-200 flex items-center justify-between flex-wrap gap-2.5 bg-slate-50/80">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
          <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Safety Violation Ledger
          </h2>
          <span className="text-xs font-mono font-bold bg-rose-100 text-rose-800 px-2 py-0.5 rounded-full border border-rose-200">
            {violations.length} {violations.length === 1 ? "Incident" : "Incidents"}
          </span>
        </div>

        {violations.length > 0 && onClearAllViolations && (
          <button
            onClick={onClearAllViolations}
            title="Clear all recorded incidents from ledger"
            className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-semibold text-slate-600 hover:text-rose-700 hover:bg-rose-50 border border-slate-300 hover:border-rose-300 transition-colors cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Ledger</span>
          </button>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto max-h-[620px] p-3 space-y-2.5">
        {violations.length === 0 ? (
          <div className="py-16 text-center text-slate-400 space-y-3">
            <div className="w-12 h-12 mx-auto rounded-full bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-700">
                Ledger Clean — No Active Violations
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5">
                Confirmed PPE violations will be recorded here in real-time.
              </div>
            </div>
          </div>
        ) : (
          violations.map((item) => {
            const timeLabel = new Date(item.timestamp * 1000).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            const missingList = Array.isArray(item.missing_ppe)
              ? item.missing_ppe
              : [item.missing_ppe];

            return (
              <div
                key={item.id}
                className="bg-white border border-rose-200 hover:border-rose-400 rounded-lg p-3 space-y-2.5 shadow-2xs hover:shadow-sm transition-all"
              >
                {/* Card Header: Worker Identity & Timestamp */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <UserX className="w-4 h-4 text-rose-600 shrink-0" />
                    <span className="text-xs font-bold text-slate-900">
                      Worker #{item.track_id}
                    </span>
                    <span className="text-slate-400 text-xs">•</span>
                    <span className="text-[11px] font-mono text-slate-500">
                      Frame {item.frame_number}
                    </span>
                  </div>

                  <div className="flex items-center space-x-1 text-[11px] font-mono text-slate-500">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span>{timeLabel}</span>
                  </div>
                </div>

                {/* All Violated Items Cleanly Formatted */}
                <div className="bg-rose-50 border border-rose-200 rounded p-2.5">
                  <div className="text-[11px] font-bold text-rose-800 uppercase tracking-wide mb-1 flex items-center space-x-1">
                    <span>Missing Required PPE:</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {missingList.map((m, idx) => (
                      <span
                        key={idx}
                        className="bg-rose-600 text-white font-bold px-2 py-0.5 rounded text-[11px] tracking-wide uppercase shadow-2xs"
                      >
                        ✗ {m}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Bottom Action Bar */}
                <div className="flex items-center justify-between pt-1 border-t border-slate-100">
                  <span className="text-[10px] text-slate-400 font-mono">
                    ID #{item.id}
                  </span>

                  <div className="flex items-center space-x-1.5">
                    <button
                      onClick={() => onSelectViolationForEvidence(item)}
                      title="Inspect Snapshot Image"
                      className="flex items-center space-x-1 px-3 py-1 rounded bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-2xs transition-colors cursor-pointer"
                    >
                      <FileSearch className="w-3.5 h-3.5" />
                      <span>Inspect</span>
                    </button>

                    {item.review_status === "UNREVIEWED" && onTriageViolation && (
                      <>
                        <button onClick={() => onTriageViolation(item.id, "CONFIRMED")} className="px-2 py-1 rounded text-xs font-semibold bg-emerald-600 text-white">Confirm</button>
                        <button onClick={() => onTriageViolation(item.id, "FALSE_POSITIVE")} className="px-2 py-1 rounded text-xs font-semibold border border-slate-300">False positive</button>
                      </>
                    )}

                    {onDeleteViolation && (
                      <button
                        onClick={() => onDeleteViolation(item.id)}
                        title="Dismiss / Remove this incident"
                        className="flex items-center space-x-1 px-2.5 py-1 rounded bg-white hover:bg-slate-100 text-slate-600 hover:text-rose-700 border border-slate-300 font-semibold text-xs transition-colors cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                        <span>Dismiss</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
