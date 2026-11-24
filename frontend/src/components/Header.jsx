import React, { useState, useEffect } from "react";
import { Download, Radio, FileSpreadsheet, FileText, Sliders, Upload } from "lucide-react";
import { getExportAuditUrl } from "../services/api";

export default function Header({ isCameraRunning = false, fps = 0, onOpenPolicy, onOpenUpload }) {
  const [timeStr, setTimeStr] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toTimeString().split(" ")[0]);
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white border-b border-slate-200 px-6 py-3.5 sticky top-0 z-30 shadow-sm">
      <div className="flex items-center justify-between flex-wrap gap-4">
        {/* Left: Branding & Subject identity */}
        <div className="flex items-center space-x-3.5">
          <div className="h-9 w-9 bg-blue-600 rounded flex items-center justify-center text-white shadow-sm font-bold text-lg tracking-wider">
            IP
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base font-bold text-slate-900 tracking-tight">
                IPSIP Safety Command
              </h1>
            </div>
            <p className="text-xs text-slate-500 font-normal">
              Industrial PPE Safety Intelligence &amp; Hardware Bridge
            </p>
          </div>
        </div>

        {/* Center: Live edge status badge */}
        <div className="flex items-center space-x-3">
          <div
            className={`flex items-center space-x-2 px-3 py-1.5 rounded text-xs font-mono border ${
              isCameraRunning
                ? "bg-emerald-50 border-emerald-300 text-emerald-800"
                : "bg-slate-50 border-slate-200 text-slate-600"
            }`}
          >
            <Radio
              className={`w-3.5 h-3.5 ${
                isCameraRunning ? "text-emerald-600 animate-pulse" : "text-slate-400"
              }`}
            />
            <span className="font-semibold">
              {isCameraRunning ? "CAMERA STREAM ACTIVE" : "CAMERA STANDBY"}
            </span>
            {isCameraRunning && fps > 0 && (
              <>
                <span className="text-slate-400">|</span>
                <span>{fps.toFixed(1)} FPS</span>
              </>
            )}
          </div>
        </div>

        {/* Right: Clock & Direct Audit Export */}
        <div className="flex items-center space-x-3">
          <div className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1.5 rounded border border-slate-200 hidden sm:block">
            {timeStr}
          </div>

          {/* Official Audit Export Buttons */}
          <div className="flex items-center space-x-2">
            <button onClick={onOpenUpload} className="flex items-center space-x-1 px-2.5 py-1.5 rounded text-xs font-medium bg-blue-600 text-white">
              <Upload className="w-3.5 h-3.5" /><span>Inspect Video</span>
            </button>
            <button onClick={onOpenPolicy} className="flex items-center space-x-1 px-2.5 py-1.5 rounded text-xs font-medium bg-white border border-slate-300">
              <Sliders className="w-3.5 h-3.5" /><span>Safety Rules</span>
            </button>
            <a
              href={getExportAuditUrl("pdf", "all")}
              download
              title="Download Official OSHA 1910.132 &amp; ISO 45001 Compliance Audit Report (PDF)"
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-bold bg-rose-600 hover:bg-rose-700 text-white shadow-xs transition-colors"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Audit (PDF)</span>
            </a>

            <a
              href={getExportAuditUrl("excel", "all")}
              download
              title="Download formatted OSHA audit report for Microsoft Excel"
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white shadow-xs transition-colors"
            >
              <FileSpreadsheet className="w-3.5 h-3.5" />
              <span>Excel</span>
            </a>

            <a
              href={getExportAuditUrl("csv", "all")}
              download
              title="Download standard CSV audit ledger"
              className="flex items-center space-x-1 px-2.5 py-1.5 rounded text-xs font-medium bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>CSV</span>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}
