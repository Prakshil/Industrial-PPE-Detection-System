import React, { useState, useEffect, useRef } from "react";
import {
  Camera,
  RefreshCw,
  Video,
  Play,
  Square,
  Check,
  X,
  ShieldCheck,
  Cpu,
  AlertTriangle,
  Volume2,
  Radio,
  StopCircle,
} from "lucide-react";
import {
  getCameraStreamUrl,
  fetchLiveCameraStatus,
  startLiveCamera,
  stopLiveCamera,
  setLiveCameraMode,
  reconnectLiveSerial,
  setBuzzerPolarity,
  setLiveCameraProfile,
  getProcessedVideoUrl,
  getVideoStreamFeedUrl,
  fetchVideoStreamStatus,
  stopVideoStream,
} from "../services/api";

export default function CameraFeed({
  liveTelemetry,
  onCameraStateChange,
  processedVideoVersion,
  videoStreamInfo,          // { filename, zone_id } when a video stream is active
  onVideoStreamStopped,     // callback when user stops the video stream
}) {
  const [streamKey, setStreamKey] = useState(Date.now());
  const [isLiveRunning, setIsLiveRunning] = useState(false);
  const [testMode, setTestMode] = useState("all_ppe");
  const [serialConnected, setSerialConnected] = useState(false);
  const [buzzerActiveLow, setBuzzerActiveLow] = useState(true);
  const [isBusy, setIsBusy] = useState(false);
  const [fps, setFps] = useState(0);
  const [cameraProfile, setCameraProfile] = useState("upper_body");
  const [serviceError, setServiceError] = useState("");

  // Video stream state
  const [videoStatus, setVideoStatus] = useState(null);
  const videoStreamActive = Boolean(videoStreamInfo);

  // Sync live camera state from backend
  const syncStatus = async () => {
    try {
      const status = await fetchLiveCameraStatus();
      if (status) {
        setIsLiveRunning(Boolean(status.running));
        setTestMode(status.test_mode || "all_ppe");
        setSerialConnected(Boolean(status.serial_connected));
        setBuzzerActiveLow(Boolean(status.buzzer_active_low));
        setCameraProfile(status.camera_profile || "upper_body");
        setServiceError(status.error || "");
        if (status.fps) setFps(status.fps);
        if (onCameraStateChange) {
          onCameraStateChange(Boolean(status.running), status.fps || 0);
        }
      }
    } catch {
      // Background retry
    }
  };

  // Poll video stream status when active
  const syncVideoStatus = async () => {
    if (!videoStreamActive) return;
    try {
      const vs = await fetchVideoStreamStatus();
      setVideoStatus(vs);
      // Reflect video stream telemetry in the camera FPS counter too
      if (vs.fps && onCameraStateChange) {
        onCameraStateChange(vs.running || !vs.done, vs.fps);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    syncStatus();
    const timer = setInterval(syncStatus, 2500);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!videoStreamActive) {
      setVideoStatus(null);
      return;
    }
    syncVideoStatus();
    const timer = setInterval(syncVideoStatus, 800);
    return () => clearInterval(timer);
  }, [videoStreamActive]);

  const handleToggleCamera = async () => {
    setIsBusy(true);
    try {
      if (isLiveRunning) {
        await stopLiveCamera();
        setIsLiveRunning(false);
      } else {
        await startLiveCamera(0, "COM10", testMode, cameraProfile);
        setIsLiveRunning(true);
      }
      setStreamKey(Date.now());
      await syncStatus();
    } catch (e) {
      console.error("Failed to toggle camera:", e);
    } finally {
      setIsBusy(false);
    }
  };

  const handleProfileChange = async (profile) => {
    setIsBusy(true);
    try {
      await setLiveCameraProfile(profile);
      setCameraProfile(profile);
    } catch (error) {
      setServiceError(error.message);
    } finally {
      setIsBusy(false);
    }
  };

  const handleSwitchMode = async (newMode) => {
    setIsBusy(true);
    try {
      await setLiveCameraMode(newMode);
      setTestMode(newMode);
      await syncStatus();
    } catch (e) {
      console.error("Failed to set mode:", e);
    } finally {
      setIsBusy(false);
    }
  };

  const handleReconnectSerial = async () => {
    setIsBusy(true);
    try {
      await reconnectLiveSerial("COM10");
      await syncStatus();
    } catch (e) {
      console.error("Failed to reconnect serial:", e);
    } finally {
      setIsBusy(false);
    }
  };

  const handleToggleBuzzerPolarity = async () => {
    setIsBusy(true);
    try {
      const nextVal = !buzzerActiveLow;
      await setBuzzerPolarity(nextVal);
      setBuzzerActiveLow(nextVal);
    } catch (e) {
      console.error("Failed to toggle buzzer polarity:", e);
    } finally {
      setIsBusy(false);
    }
  };

  const handleStopVideoStream = async () => {
    setIsBusy(true);
    try {
      await stopVideoStream();
      if (onVideoStreamStopped) onVideoStreamStopped();
    } catch (e) {
      console.error("Failed to stop video stream:", e);
    } finally {
      setIsBusy(false);
    }
  };

  // ── Checklist data ──
  // If video stream is active use its checklist, otherwise use live telemetry
  const checklist = videoStreamActive && videoStatus
    ? videoStatus.checklist || {}
    : liveTelemetry?.checklist || {
        helmet: { present: false, confidence: 0.0 },
        vest: { present: false, confidence: 0.0 },
        boots: { present: false, confidence: 0.0 },
        gloves: { present: false, confidence: 0.0 },
      };

  const observablePpe = liveTelemetry?.observable_ppe || ["helmet", "vest", "gloves"];

  const isHelmet = Boolean(checklist.helmet?.present);
  const isVest   = Boolean(checklist.vest?.present);
  const isBoots  = Boolean(checklist.boots?.present);
  const isGloves = Boolean(checklist.gloves?.present);
  const bootsObservable = videoStreamActive ? true : observablePpe.includes("boots");

  const isCompliant =
    testMode === "helmet_only"
      ? isHelmet
      : isHelmet && isVest && isBoots && isGloves;

  const currentFps = videoStreamActive
    ? videoStatus?.fps ?? 0
    : liveTelemetry?.fps || fps || 0;

  const workers = videoStreamActive
    ? videoStatus?.workers || []
    : liveTelemetry?.workers || [];

  const streamUrl = `${getCameraStreamUrl("active_work_zone", "cam_live")}&t=${streamKey}`;
  const processedVideoUrl = processedVideoVersion
    ? `${getProcessedVideoUrl()}?t=${processedVideoVersion}`
    : "";
  const videoStreamUrl = `${getVideoStreamFeedUrl()}?t=${streamKey}`;

  // Progress info for video stream
  const framesProcessed = videoStatus?.frames_processed || 0;
  const totalFrames = videoStatus?.total_frames || 0;
  const progressPct = totalFrames > 0 ? Math.min(100, (framesProcessed / totalFrames) * 100) : 0;
  const videoWorkers = videoStatus?.unique_workers || 0;
  const videoViolations = videoStatus?.violations_logged || 0;

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm flex flex-col overflow-hidden">
      {/* Control Strip */}
      <div className="bg-slate-50 border-b border-slate-200 px-4 py-3 flex items-center justify-between flex-wrap gap-3">
        {/* Left: Controls */}
        <div className="flex items-center space-x-2 flex-wrap gap-2">
          {videoStreamActive ? (
            /* Video Stream Controls */
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-blue-600 text-white text-xs font-bold">
                <Radio className="w-3.5 h-3.5 animate-pulse" />
                <span>🎬 VIDEO STREAM ACTIVE</span>
              </div>
              <div className="text-xs text-slate-600 font-mono bg-white border border-slate-300 rounded px-2 py-1.5">
                {videoStreamInfo?.filename?.slice(0, 22) || "uploaded video"}
              </div>
              <button
                onClick={handleStopVideoStream}
                disabled={isBusy}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-sm"
              >
                <StopCircle className="w-3.5 h-3.5" />
                <span>Stop Stream</span>
              </button>
            </div>
          ) : (
            /* Live Camera Controls */
            <>
              <button
                onClick={handleToggleCamera}
                disabled={isBusy}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md font-bold text-xs shadow-sm transition-all cursor-pointer ${
                  isLiveRunning
                    ? "bg-rose-600 hover:bg-rose-700 text-white"
                    : "bg-emerald-600 hover:bg-emerald-700 text-white"
                }`}
              >
                {isLiveRunning ? (
                  <>
                    <Square className="w-3.5 h-3.5 fill-current" />
                    <span>⏹ Turn OFF Camera</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>▶ Turn ON Laptop Camera</span>
                  </>
                )}
              </button>

              <div className="flex items-center bg-white border border-slate-300 rounded p-0.5">
                <button
                  onClick={() => handleSwitchMode("all_ppe")}
                  disabled={isBusy}
                  title="Laptop upper-body audit: helmet, vest, and gloves are evaluated; boots are explicitly out of view."
                  className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors cursor-pointer ${
                    testMode === "all_ppe"
                      ? "bg-blue-600 text-white shadow-xs font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  💻 Upper-Body PPE Audit (Laptop)
                </button>
                <button
                  onClick={() => handleSwitchMode("helmet_only")}
                  disabled={isBusy}
                  title="Diagnostic mode for isolated helmet calibration"
                  className={`px-3 py-1.5 rounded text-xs font-semibold transition-colors cursor-pointer ${
                    testMode === "helmet_only"
                      ? "bg-blue-600 text-white shadow-xs font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  🪖 Helmet Diagnostic
                </button>
              </div>
              <select
                value={cameraProfile}
                onChange={(e) => handleProfileChange(e.target.value)}
                disabled={isBusy}
                className="text-xs border border-slate-300 rounded px-2 py-1.5 bg-white"
              >
                <option value="upper_body">Laptop / Upper Body</option>
                <option value="full_body">Full Body Camera</option>
                <option value="helmet_only">Helmet View</option>
              </select>
            </>
          )}
        </div>

        {/* Right: ESP32 Hardware Status */}
        <div className="flex items-center space-x-2">
          <div
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-mono font-semibold border ${
              serialConnected
                ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                : "bg-amber-50 text-amber-800 border-amber-300"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>ESP32: {serialConnected ? "COM10 LINKED" : "COM10 STANDBY"}</span>
          </div>

          <button
            onClick={handleToggleBuzzerPolarity}
            disabled={isBusy}
            title="Toggle Buzzer Polarity between Active-High (normal) and Active-Low (inverted for 3-pin modules)"
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 font-semibold text-xs transition-colors cursor-pointer"
          >
            <Volume2 className="w-3 h-3 text-slate-500" />
            <span>{buzzerActiveLow ? "Buzzer: Active-Low" : "Buzzer: Active-High"}</span>
          </button>

          <button
            onClick={handleReconnectSerial}
            disabled={isBusy}
            title="Reconnect serial port to ESP32 on COM10"
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 font-semibold text-xs transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3 h-3 text-slate-500" />
            <span>Reconnect</span>
          </button>
        </div>
      </div>

      {/* Video stream progress bar */}
      {videoStreamActive && totalFrames > 0 && (
        <div className="bg-blue-900 px-4 py-1.5 flex items-center space-x-3">
          <span className="text-blue-200 text-[11px] font-mono whitespace-nowrap">
            Frame {framesProcessed}/{totalFrames}
          </span>
          <div className="flex-1 bg-blue-700 rounded-full h-1.5">
            <div
              className="bg-blue-300 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <span className="text-blue-200 text-[11px] font-mono whitespace-nowrap">
            {videoWorkers} worker{videoWorkers !== 1 ? "s" : ""} • {videoViolations} violations
          </span>
          {videoStatus?.done && (
            <span className="text-emerald-300 text-[11px] font-bold">✓ COMPLETE</span>
          )}
        </div>
      )}

      {serviceError && (
        <div className="px-4 py-2 text-xs font-medium bg-rose-50 text-rose-800 border-b border-rose-200">
          Camera service: {serviceError}
        </div>
      )}

      {/* Main Video Viewport */}
      <div className="relative bg-slate-950 aspect-video w-full overflow-hidden flex items-center justify-center">
        {videoStreamActive ? (
          /* ── VIDEO STREAM MODE: live annotated MJPEG ── */
          <img
            key={videoStreamUrl}
            src={videoStreamUrl}
            alt="Video Stream — Live PPE Inspection"
            className="w-full h-full object-contain"
          />
        ) : processedVideoUrl ? (
          /* ── PROCESSED VIDEO playback ── */
          <video key={processedVideoUrl} src={processedVideoUrl} controls className="w-full h-full object-contain" />
        ) : isLiveRunning ? (
          /* ── LIVE WEBCAM MJPEG ── */
          <img
            key={streamKey}
            src={streamUrl}
            alt="Live Laptop Camera Feed"
            className="w-full h-full object-cover"
          />
        ) : (
          /* ── STANDBY ── */
          <div className="flex flex-col items-center justify-center p-8 space-y-4 text-center z-10 w-full h-full bg-slate-900/90">
            <div className="w-16 h-16 rounded-full bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
              <Camera className="w-8 h-8" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white tracking-wide">
                Live Laptop Camera (Integrated in Dashboard)
              </h3>
              <p className="text-xs text-slate-300 mt-1 max-w-md">
                Click below to turn on the camera — or upload a video via the header button to stream it with live bounding boxes, ESP32 alerts, and attendance recording.
              </p>
            </div>
            <button
              onClick={handleToggleCamera}
              disabled={isBusy}
              className="flex items-center space-x-2 px-6 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm shadow-xl transition-all transform hover:scale-105 active:scale-95 cursor-pointer"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>▶ Turn ON Laptop Camera</span>
            </button>
          </div>
        )}

        {/* Live FPS & Worker count overlay */}
        {(isLiveRunning || videoStreamActive) && (
          <div className="absolute bottom-3 left-3 bg-slate-900/85 backdrop-blur-xs text-slate-200 px-3 py-1.5 rounded text-xs font-mono border border-slate-700 space-x-3 flex items-center z-10">
            <span className="flex items-center space-x-1.5">
              <span className={`w-2 h-2 rounded-full inline-block animate-pulse ${videoStreamActive ? "bg-blue-400" : "bg-emerald-500"}`} />
              <span>{videoStreamActive ? "VIDEO STREAM" : "LIVE"} @ {currentFps.toFixed(1)} FPS</span>
            </span>
            <span className="text-slate-500">|</span>
            <span className={`font-semibold ${videoStreamActive ? "text-blue-300" : "text-emerald-400"}`}>
              {workers.length} Worker(s) Tracked
            </span>
            {videoStreamActive && videoStatus?.done && (
              <>
                <span className="text-slate-500">|</span>
                <span className="text-emerald-400 font-bold">✓ DONE</span>
              </>
            )}
          </div>
        )}

        {/* Video stream source label */}
        {videoStreamActive && (
          <div className="absolute top-8 right-3 bg-blue-700/80 backdrop-blur-xs text-blue-100 px-2.5 py-1 rounded text-[11px] font-mono border border-blue-500/40 z-10">
            📹 {videoStreamInfo?.filename?.slice(0, 24) || "video"}
          </div>
        )}
      </div>

      {/* Tactile 4-Item PPE Status Bar Below Video */}
      <div className="bg-slate-50 border-t border-slate-200 px-4 py-3">
        <div className="flex items-center justify-between flex-wrap gap-2.5">
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-blue-600" />
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              {videoStreamActive ? "Video Stream" : "Live Camera"} Checklist:
            </span>
          </div>

          <div className="flex items-center flex-wrap gap-2">
            {/* Helmet Badge */}
            <div
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-bold border transition-colors ${
                isHelmet
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {isHelmet ? (
                <Check className="w-3.5 h-3.5 text-emerald-600 stroke-[3]" />
              ) : (
                <X className="w-3.5 h-3.5 text-rose-600 stroke-[3]" />
              )}
              <span>⛑️ Helmet: {isHelmet ? "✓ WORN" : "✗ MISSING"}</span>
              {checklist.helmet?.confidence > 0 && (
                <span className="text-[9px] opacity-60 ml-0.5">
                  {(checklist.helmet.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Vest Badge */}
            <div
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-bold border transition-colors ${
                isVest
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {isVest ? (
                <Check className="w-3.5 h-3.5 text-emerald-600 stroke-[3]" />
              ) : (
                <X className="w-3.5 h-3.5 text-rose-600 stroke-[3]" />
              )}
              <span>🦺 Vest: {isVest ? "✓ WORN" : "✗ MISSING"}</span>
              {checklist.vest?.confidence > 0 && (
                <span className="text-[9px] opacity-60 ml-0.5">
                  {(checklist.vest.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Boots Badge */}
            <div
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-bold border transition-colors ${
                !bootsObservable
                  ? "bg-amber-50 text-amber-800 border-amber-300"
                  : isBoots
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {!bootsObservable ? (
                <span className="w-3.5 text-center">—</span>
              ) : isBoots ? (
                <Check className="w-3.5 h-3.5 text-emerald-600 stroke-[3]" />
              ) : (
                <X className="w-3.5 h-3.5 text-rose-600 stroke-[3]" />
              )}
              <span>🥾 Boots: {!bootsObservable ? "OUT OF VIEW" : isBoots ? "✓ WORN" : "✗ MISSING"}</span>
            </div>

            {/* Gloves Badge */}
            <div
              className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs font-bold border transition-colors ${
                isGloves
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                  : "bg-rose-50 text-rose-800 border-rose-300"
              }`}
            >
              {isGloves ? (
                <Check className="w-3.5 h-3.5 text-emerald-600 stroke-[3]" />
              ) : (
                <X className="w-3.5 h-3.5 text-rose-600 stroke-[3]" />
              )}
              <span>🧤 Gloves: {isGloves ? "✓ WORN" : "✗ MISSING"}</span>
              {checklist.gloves?.confidence > 0 && (
                <span className="text-[9px] opacity-60 ml-0.5">
                  {(checklist.gloves.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
