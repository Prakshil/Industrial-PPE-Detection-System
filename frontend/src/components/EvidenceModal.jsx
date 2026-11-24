import React, { useState } from "react";
import { X, ShieldAlert, Check, XCircle, FileText, Trash2 } from "lucide-react";
import { getEvidenceImageUrl } from "../services/api";

export default function EvidenceModal({ violation, onClose, onDeleteViolation, onTriageViolation }) {
  if (!violation) return null;

  const [imgError, setImgError] = useState(false);

  const missingList = Array.isArray(violation.missing_ppe)
    ? violation.missing_ppe
    : [violation.missing_ppe];
  const presentList = Array.isArray(violation.present_ppe) ? violation.present_ppe : [];
  const confMap = violation.confidence_map || {};

  const timeLabel = new Date(violation.timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const handleDelete = () => {
    if (onDeleteViolation) {
      onDeleteViolation(violation.id);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-xs p-4">
      <div className="bg-white rounded-lg shadow-2xl border border-slate-300 max-w-2xl w-full overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-rose-600" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Incident Evidence Snapshot — Worker #{violation.track_id}
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                Incident #{violation.id} • Recorded at {timeLabel} (Frame {violation.frame_number})
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200 cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>

          {violation.review_status === "UNREVIEWED" && onTriageViolation && (
            <div className="flex space-x-2">
              <button onClick={() => onTriageViolation(violation.id, "CONFIRMED")} className="px-3 py-1.5 rounded text-xs font-semibold bg-emerald-600 text-white">Confirm</button>
              <button onClick={() => onTriageViolation(violation.id, "FALSE_POSITIVE")} className="px-3 py-1.5 rounded text-xs font-semibold border border-slate-300">False positive</button>
            </div>
          )}
        </div>

        {/* Modal Body */}
        <div className="p-5 overflow-y-auto space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Left: Photographic crop */}
            <div className="bg-slate-950 rounded border border-slate-300 flex flex-col items-center justify-center p-2 relative aspect-3/4 overflow-hidden">
              {!imgError && violation.evidence_path ? (
                <img
                  src={getEvidenceImageUrl(violation.evidence_path)}
                  alt={`Worker #${violation.track_id} Evidence Crop`}
                  onError={() => setImgError(true)}
                  className="max-h-full max-w-full object-contain rounded"
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-slate-400 p-4 text-center space-y-2">
                  <FileText className="w-10 h-10 text-slate-500" />
                  <div className="text-xs font-semibold text-slate-300">
                    Live Photographic Snapshot
                  </div>
                  <div className="text-[10px] font-mono text-slate-500 truncate max-w-[200px]">
                    {violation.evidence_path || "Snapshot saved on edge storage"}
                  </div>
                </div>
              )}
              <div className="absolute bottom-2 left-2 bg-slate-900/90 text-white text-[10px] font-mono px-2 py-0.5 rounded border border-slate-700">
                Worker #{violation.track_id}
              </div>
            </div>

            {/* Right: PPE Compliance Item Status */}
            <div className="space-y-3 flex flex-col justify-between">
              <div className="space-y-2">
                <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wide border-b border-slate-200 pb-1">
                  Required PPE Audit Breakdown
                </h4>

                {[
                  { name: "helmet", label: "Hard Hat / Helmet" },
                  { name: "vest", label: "High-Visibility Safety Vest" },
                  { name: "boots", label: "Steel-Toe Safety Boots" },
                  { name: "gloves", label: "Protective Work Gloves" },
                ].map((item) => {
                  const isMissing = missingList.includes(item.name);
                  const isPresent = presentList.includes(item.name);
                  const conf = confMap[item.name];

                  return (
                    <div
                      key={item.name}
                      className={`p-2.5 rounded border text-xs flex items-center justify-between ${
                        isMissing
                          ? "bg-rose-50 border-rose-200 text-rose-900 font-bold"
                          : isPresent
                          ? "bg-emerald-50 border-emerald-200 text-emerald-900 font-semibold"
                          : "bg-slate-50 border-slate-200 text-slate-500"
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        {isMissing ? (
                          <XCircle className="w-4 h-4 text-rose-600 shrink-0" />
                        ) : isPresent ? (
                          <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                        ) : (
                          <span className="w-4 h-4 text-slate-400 font-mono text-center">-</span>
                        )}
                        <div>
                          <div className="uppercase tracking-tight text-[11px]">
                            {item.label}
                          </div>
                        </div>
                      </div>

                      <div className="text-right font-mono text-[11px]">
                        {isMissing ? (
                          <span className="text-rose-700 font-bold uppercase">MISSING</span>
                        ) : isPresent ? (
                          <span className="text-emerald-700 font-bold">
                            {conf ? `${(conf * 100).toFixed(0)}%` : "WORN"}
                          </span>
                        ) : (
                          "N/A"
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Status summary box */}
              <div className="bg-rose-50 border border-rose-200 rounded p-2.5">
                <div className="text-[11px] font-bold text-rose-900 uppercase">
                  Incident Result
                </div>
                <div className="text-xs text-rose-700 mt-0.5">
                  Worker #{violation.track_id} confirmed missing {missingList.length} required safety {missingList.length === 1 ? "item" : "items"} ({missingList.join(", ")}).
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer Controls */}
        <div className="px-5 py-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <button
            onClick={handleDelete}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-semibold text-slate-600 hover:text-rose-700 hover:bg-rose-50 border border-slate-300 transition-colors cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Delete Incident</span>
          </button>

          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded text-xs font-bold bg-slate-900 hover:bg-slate-800 text-white shadow-xs transition-colors cursor-pointer"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
