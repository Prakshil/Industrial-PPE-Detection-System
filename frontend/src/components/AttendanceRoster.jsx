import React, { useState } from "react";
import {
  Users,
  ShieldCheck,
  AlertTriangle,
  Clock,
  Trash2,
  FileSpreadsheet,
  FileText,
  CheckCircle2,
  XCircle,
  Eye,
  Activity,
  Filter,
} from "lucide-react";
import { getExportAttendanceUrl, getEvidenceImageUrl } from "../services/api";

export default function AttendanceRoster({
  attendance = [],
  summary = {},
  onClearAttendance,
  onRefresh,
}) {
  const [filterStatus, setFilterStatus] = useState("all");
  const [previewImage, setPreviewImage] = useState(null);

  const totalWorkers = summary?.total_workers ?? attendance.length;
  const passedWorkers = summary?.passed_workers ?? attendance.filter((a) => a.compliance_status === "PASSED").length;
  const flaggedWorkers = summary?.flagged_workers ?? attendance.filter((a) => a.compliance_status === "FLAGGED").length;
  const passRate = summary?.pass_rate_percent ?? (totalWorkers > 0 ? ((passedWorkers / totalWorkers) * 100).toFixed(1) : 100);

  const filteredList = attendance.filter((item) => {
    if (filterStatus === "all") return true;
    return item.compliance_status === filterStatus;
  });

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm flex flex-col h-full overflow-hidden">
      {/* Top Header & Actions */}
      <div className="px-4 py-3 border-b border-slate-200 bg-slate-50/90 flex items-center justify-between flex-wrap gap-2.5">
        <div className="flex items-center space-x-2">
          <Users className="w-4 h-4 text-blue-600 shrink-0" />
          <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
            Shift Attendance &amp; PPE Registry
          </h2>
          <span className="text-xs font-mono font-bold bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full border border-blue-200">
            {totalWorkers} {totalWorkers === 1 ? "Worker" : "Workers"}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          {attendance.length > 0 && (
            <>
              <a
                href={getExportAttendanceUrl("excel", "all", "all")}
                download
                title="Download Shift Attendance Roster in Microsoft Excel format"
                className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-semibold text-emerald-700 bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 transition-colors"
              >
                <FileSpreadsheet className="w-3.5 h-3.5" />
                <span>Excel</span>
              </a>

              <a
                href={getExportAttendanceUrl("csv", "all", "all")}
                download
                title="Download Shift Attendance in CSV format"
                className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-semibold text-slate-700 bg-white hover:bg-slate-100 border border-slate-300 transition-colors"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>CSV</span>
              </a>

              {onClearAttendance && (
                <button
                  onClick={onClearAttendance}
                  title="Clear all recorded shift attendance"
                  className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-semibold text-slate-600 hover:text-rose-700 hover:bg-rose-50 border border-slate-300 hover:border-rose-300 transition-colors cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Reset Roster</span>
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* KPI Mini-Strip */}
      <div className="bg-slate-100/70 border-b border-slate-200 px-4 py-2 grid grid-cols-3 gap-2 text-center text-xs">
        <div className="bg-white rounded border border-slate-200 py-1.5 px-2">
          <div className="text-[10px] text-slate-500 font-medium uppercase">Shift Headcount</div>
          <div className="text-base font-bold text-slate-800 font-mono">{totalWorkers}</div>
        </div>
        <div className="bg-emerald-50/60 rounded border border-emerald-200 py-1.5 px-2">
          <div className="text-[10px] text-emerald-700 font-medium uppercase">PPE Passed (Clean)</div>
          <div className="text-base font-bold text-emerald-700 font-mono">
            {passedWorkers} <span className="text-[10px] font-normal">({passRate}%)</span>
          </div>
        </div>
        <div className="bg-rose-50/60 rounded border border-rose-200 py-1.5 px-2">
          <div className="text-[10px] text-rose-700 font-medium uppercase">Flagged Violations</div>
          <div className="text-base font-bold text-rose-700 font-mono">{flaggedWorkers}</div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="px-3 pt-2 pb-1 border-b border-slate-200 bg-white flex items-center justify-between">
        <div className="flex items-center space-x-1">
          <button
            onClick={() => setFilterStatus("all")}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors cursor-pointer ${
              filterStatus === "all"
                ? "bg-slate-800 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
          >
            All ({totalWorkers})
          </button>
          <button
            onClick={() => setFilterStatus("PASSED")}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors cursor-pointer ${
              filterStatus === "PASSED"
                ? "bg-emerald-600 text-white"
                : "text-emerald-700 hover:bg-emerald-50"
            }`}
          >
            ✓ Passed ({passedWorkers})
          </button>
          <button
            onClick={() => setFilterStatus("FLAGGED")}
            className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-colors cursor-pointer ${
              filterStatus === "FLAGGED"
                ? "bg-rose-600 text-white"
                : "text-rose-700 hover:bg-rose-50"
            }`}
          >
            ✗ Flagged ({flaggedWorkers})
          </button>
        </div>
      </div>

      {/* Worker List */}
      <div className="flex-1 overflow-y-auto max-h-[580px] p-3 space-y-2.5">
        {filteredList.length === 0 ? (
          <div className="py-16 text-center text-slate-400 space-y-3">
            <div className="w-12 h-12 mx-auto rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
              <Users className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-700">
                {filterStatus === "all"
                  ? "No Workers Recorded in Roster Yet"
                  : `No ${filterStatus} Workers Found`}
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5 max-w-xs mx-auto">
                Turn on the live laptop camera or upload a video clip — tracked personnel will be logged here automatically.
              </div>
            </div>
          </div>
        ) : (
          filteredList.map((worker) => {
            const isPassed = worker.compliance_status === "PASSED";
            const inTime = new Date(worker.first_seen * 1000).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            const outTime = new Date(worker.last_seen * 1000).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            });
            const dwellSec = worker.dwell_seconds || 0;
            const dwellFmt =
              dwellSec >= 60
                ? `${Math.floor(dwellSec / 60)}m ${Math.floor(dwellSec % 60)}s`
                : `${dwellSec.toFixed(1)}s`;

            const score = worker.verification_score ?? 100;
            const scoreColor =
              score >= 90
                ? "bg-emerald-500 text-emerald-700"
                : score >= 70
                ? "bg-amber-500 text-amber-700"
                : "bg-rose-500 text-rose-700";

            const snapUrl = worker.snapshot_path ? getEvidenceImageUrl(worker.snapshot_path) : null;

            return (
              <div
                key={worker.id || `${worker.session_id}_${worker.track_id}`}
                className={`bg-white rounded-lg border p-3 space-y-2 transition-all shadow-2xs hover:shadow-xs ${
                  isPassed
                    ? "border-emerald-200 hover:border-emerald-300"
                    : "border-rose-200 hover:border-rose-300"
                }`}
              >
                {/* Header: ID, Status, and Feed */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono font-bold text-xs bg-slate-100 text-slate-800 px-2 py-0.5 rounded border border-slate-300">
                      Worker #{worker.track_id}
                    </span>

                    <span
                      className={`flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-bold border ${
                        isPassed
                          ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                          : "bg-rose-50 text-rose-800 border-rose-300"
                      }`}
                    >
                      {isPassed ? (
                        <>
                          <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                          <span>PASSED (100% PPE)</span>
                        </>
                      ) : (
                        <>
                          <XCircle className="w-3 h-3 text-rose-600" />
                          <span>FLAGGED ({worker.violation_count} Violations)</span>
                        </>
                      )}
                    </span>
                  </div>

                  <span className="text-[10px] text-slate-500 font-mono truncate max-w-[130px]" title={worker.source}>
                    {worker.source}
                  </span>
                </div>

                {/* Details Grid & Snapshot */}
                <div className="flex items-start justify-between gap-3 pt-1">
                  <div className="space-y-1.5 flex-1 text-[11px]">
                    {/* In / Out time & Dwell */}
                    <div className="grid grid-cols-2 gap-1 text-slate-600">
                      <div>
                        <span className="text-slate-400 block text-[9.5px] uppercase">First Seen</span>
                        <span className="font-mono font-semibold text-slate-800">{inTime}</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[9.5px] uppercase">Last Seen</span>
                        <span className="font-mono font-semibold text-slate-800">{outTime}</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-3 text-slate-500 font-mono text-[10.5px]">
                      <span className="flex items-center space-x-1">
                        <Clock className="w-3 h-3 text-slate-400" />
                        <span>Dwell: <strong className="text-slate-800">{dwellFmt}</strong></span>
                      </span>
                      <span>•</span>
                      <span>{worker.total_frames || 1} frames</span>
                    </div>

                    {/* PPE Verification Score Bar */}
                    <div className="space-y-1 pt-0.5">
                      <div className="flex items-center justify-between text-[10.5px]">
                        <span className="text-slate-600 font-medium">PPE Verification Score:</span>
                        <span className={`font-mono font-bold ${scoreColor.split(" ")[1]}`}>
                          {score.toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${scoreColor.split(" ")[0]}`}
                          style={{ width: `${Math.min(100, Math.max(score, 5))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Thumbnail Snapshot */}
                  {snapUrl && (
                    <div
                      onClick={() => setPreviewImage(snapUrl)}
                      className="cursor-pointer group relative shrink-0 w-16 h-16 rounded border border-slate-200 overflow-hidden bg-slate-100 shadow-2xs hover:border-blue-400"
                      title="Click to view full worker snapshot"
                    >
                      <img
                        src={snapUrl}
                        alt={`Worker #${worker.track_id}`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          e.target.style.display = "none";
                        }}
                      />
                      <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white">
                        <Eye className="w-4 h-4" />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Snapshot Preview Modal */}
      {previewImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/70 backdrop-blur-xs p-4"
          onClick={() => setPreviewImage(null)}
        >
          <div
            className="bg-white rounded-xl overflow-hidden shadow-2xl max-w-sm w-full border border-slate-300"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <span className="text-xs font-bold text-slate-800">Worker Attendance Snapshot</span>
              <button
                onClick={() => setPreviewImage(null)}
                className="text-slate-400 hover:text-slate-700 text-xs font-bold"
              >
                ✕ Close
              </button>
            </div>
            <div className="p-3 bg-slate-950 flex items-center justify-center">
              <img src={previewImage} alt="Worker" className="max-h-80 w-auto rounded object-contain" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
