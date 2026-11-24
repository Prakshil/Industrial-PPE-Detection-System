import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import CameraFeed from "./components/CameraFeed";
import IncidentQueue from "./components/IncidentQueue";
import AttendanceRoster from "./components/AttendanceRoster";
import AnalyticsOverview from "./components/AnalyticsOverview";
import EvidenceModal from "./components/EvidenceModal";
import PolicyConfigModal from "./components/PolicyConfigModal";
import VideoUploadModal from "./components/VideoUploadModal";
import {
  fetchViolations,
  fetchTelemetry,
  triageViolation,
  deleteViolation,
  clearAllViolations,
  connectTelemetryWebSocket,
  fetchZones,
  updateZonePolicy,
  fetchAttendance,
  clearAttendance,
} from "./services/api";

export default function App() {
  const [activeTab, setActiveTab] = useState("violations"); // 'violations' | 'attendance'
  const [violations, setViolations] = useState([]);
  const [attendance, setAttendance] = useState([]);
  const [attendanceSummary, setAttendanceSummary] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [selectedViolationForEvidence, setSelectedViolationForEvidence] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraFps, setCameraFps] = useState(0);
  const [zones, setZones] = useState([]);
  const [policyOpen, setPolicyOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [processedVideoVersion, setProcessedVideoVersion] = useState(0);
  // Video stream state: null = no stream, { filename, zone_id } = stream active
  const [videoStreamInfo, setVideoStreamInfo] = useState(null);

  // Fetch initial data
  const loadData = async () => {
    try {
      const [vData, tData, zoneData, attData] = await Promise.all([
        fetchViolations("all"),
        fetchTelemetry(),
        fetchZones(),
        fetchAttendance(),
      ]);
      setViolations(vData || []);
      setTelemetry(tData || null);
      setZones(zoneData || []);
      if (attData) {
        setAttendance(attData.attendance || []);
        setAttendanceSummary(attData.summary || null);
      }
    } catch (err) {
      console.error("Error loading safety data:", err);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => {
      fetchViolations("all").then(setViolations).catch(() => {});
      fetchTelemetry().then(setTelemetry).catch(() => {});
      fetchAttendance().then((res) => {
        if (res) {
          setAttendance(res.attendance || []);
          if (res.summary) setAttendanceSummary(res.summary);
        }
      }).catch(() => {});
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  // Connect WebSocket for real-time telemetry from camera or video stream
  useEffect(() => {
    const disconnect = connectTelemetryWebSocket(
      (data) => {
        setLiveTelemetry(data);
        if (data.fps !== undefined) setCameraFps(data.fps);
        // Refresh violations & attendance when video stream pushes new events or active workers detected
        if (data.source === "video_stream" || (data.active_workers && data.active_workers.length > 0)) {
          fetchViolations("all").then(setViolations).catch(() => {});
          fetchAttendance().then((res) => {
            if (res) {
              setAttendance(res.attendance || []);
              if (res.summary) setAttendanceSummary(res.summary);
            }
          }).catch(() => {});
        }
        if (data.attendance_summary) {
          setAttendanceSummary(data.attendance_summary);
        }
      },
      (err) => {
        console.warn("Telemetry WS disconnected, polling fallback active.", err);
      }
    );
    return disconnect;
  }, []);

  // Delete single incident
  const handleDeleteViolation = async (id) => {
    try {
      await deleteViolation(id);
      setViolations((prev) => prev.filter((v) => v.id !== id));
      fetchTelemetry().then(setTelemetry).catch(() => {});
    } catch (err) {
      console.error("Failed to delete violation:", err);
    }
  };

  // Clear all incidents from ledger
  const handleClearAllViolations = async () => {
    try {
      await clearAllViolations();
      setViolations([]);
      fetchTelemetry().then(setTelemetry).catch(() => {});
    } catch (err) {
      console.error("Failed to clear violations:", err);
    }
  };

  // Clear shift attendance records
  const handleClearAttendance = async () => {
    try {
      await clearAttendance();
      setAttendance([]);
      setAttendanceSummary({ total_workers: 0, passed_workers: 0, flagged_workers: 0, pass_rate_percent: 100 });
      fetchTelemetry().then(setTelemetry).catch(() => {});
    } catch (err) {
      console.error("Failed to clear attendance:", err);
    }
  };

  const handleTriageViolation = async (id, action) => {
    await triageViolation(id, action);
    await loadData();
  };

  const handleUpdatePolicy = async (zoneId, policy) => {
    await updateZonePolicy(zoneId, policy);
    setZones(await fetchZones());
  };

  // Called when VideoUploadModal successfully starts the stream
  const handleStreamStarted = (info) => {
    setVideoStreamInfo(info);
    setProcessedVideoVersion(0); // Clear any old processed video overlay
    // Start refreshing violations & attendance immediately as stream runs
    loadData();
  };

  // Called when user clicks "Stop Stream" in CameraFeed
  const handleVideoStreamStopped = () => {
    setVideoStreamInfo(null);
    loadData(); // Refresh violations, telemetry & attendance after stream ends
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col font-sans text-slate-900">
      {/* Top Header Bar */}
      <Header
        isCameraRunning={isCameraActive || Boolean(cameraFps > 0) || Boolean(videoStreamInfo)}
        fps={cameraFps}
        onOpenPolicy={() => setPolicyOpen(true)}
        onOpenUpload={() => setUploadOpen(true)}
        videoStreamActive={Boolean(videoStreamInfo)}
      />

      {/* Main Command Center Layout */}
      <main className="flex-1 max-w-[1720px] w-full mx-auto p-4 sm:p-6 space-y-5">
        {/* Top Operational Section: Video Feed & (Incident Queue / Attendance Roster) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column: Live Camera Surveillance Feed */}
          <div className="lg:col-span-7 xl:col-span-7 2xl:col-span-7">
            <CameraFeed
              liveTelemetry={liveTelemetry}
              onCameraStateChange={(active, currentFps) => {
                setIsCameraActive(active);
                if (currentFps) setCameraFps(currentFps);
              }}
              processedVideoVersion={processedVideoVersion}
              videoStreamInfo={videoStreamInfo}
              onVideoStreamStopped={handleVideoStreamStopped}
            />
          </div>

          {/* Right Column: Tabbed Safety Violations & Shift Attendance */}
          <div className="lg:col-span-5 xl:col-span-5 2xl:col-span-5 flex flex-col space-y-2.5">
            {/* Clean Tab Switcher */}
            <div className="bg-slate-200/90 p-1 rounded-lg flex items-center space-x-1.5 border border-slate-300/80 shadow-2xs">
              <button
                type="button"
                onClick={() => setActiveTab("violations")}
                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold transition-all flex items-center justify-center space-x-2 cursor-pointer ${
                  activeTab === "violations"
                    ? "bg-white text-slate-900 shadow-xs border border-slate-200"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/70"
                }`}
              >
                <span>🚨 Safety Violations</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold ${
                    activeTab === "violations"
                      ? violations.length > 0
                        ? "bg-rose-100 text-rose-800 border border-rose-200"
                        : "bg-slate-100 text-slate-700 border border-slate-200"
                      : "bg-slate-300 text-slate-700"
                  }`}
                >
                  {violations.length}
                </span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab("attendance")}
                className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold transition-all flex items-center justify-center space-x-2 cursor-pointer ${
                  activeTab === "attendance"
                    ? "bg-white text-slate-900 shadow-xs border border-slate-200"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-100/70"
                }`}
              >
                <span>👷 Shift Attendance</span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold ${
                    activeTab === "attendance"
                      ? "bg-blue-100 text-blue-800 border border-blue-200"
                      : "bg-slate-300 text-slate-700"
                  }`}
                >
                  {attendanceSummary?.total_workers ?? attendance.length}
                </span>
              </button>
            </div>

            {/* Tab Panel Content */}
            <div className="flex-1 min-h-0">
              {activeTab === "violations" ? (
                <IncidentQueue
                  violations={violations}
                  onDeleteViolation={handleDeleteViolation}
                  onClearAllViolations={handleClearAllViolations}
                  onSelectViolationForEvidence={setSelectedViolationForEvidence}
                  onTriageViolation={handleTriageViolation}
                />
              ) : (
                <AttendanceRoster
                  attendance={attendance}
                  summary={attendanceSummary}
                  onClearAttendance={handleClearAttendance}
                  onRefresh={() => {
                    fetchAttendance().then((res) => {
                      if (res) {
                        setAttendance(res.attendance || []);
                        if (res.summary) setAttendanceSummary(res.summary);
                      }
                    }).catch(() => {});
                  }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Lower Section: Real Compliance Telemetry & KPIs */}
        <div>
          <AnalyticsOverview telemetry={telemetry} />
        </div>
      </main>

      {/* Evidence Crop Inspector Modal */}
      <EvidenceModal
        violation={selectedViolationForEvidence}
        onClose={() => setSelectedViolationForEvidence(null)}
        onDeleteViolation={handleDeleteViolation}
        onTriageViolation={handleTriageViolation}
      />
      <PolicyConfigModal
        isOpen={policyOpen}
        onClose={() => setPolicyOpen(false)}
        zones={zones}
        onUpdatePolicy={handleUpdatePolicy}
      />
      <VideoUploadModal
        isOpen={uploadOpen}
        onClose={() => setUploadOpen(false)}
        zones={zones}
        onStreamStarted={handleStreamStarted}
      />

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-3 text-center text-xs text-slate-500 font-mono">
        Industrial PPE Continuous Monitoring Platform &bull; ANSI Z535 / ISO 45001 Compliance &bull; SQLite WAL
      </footer>
    </div>
  );
}
