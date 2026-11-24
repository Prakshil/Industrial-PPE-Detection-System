import React, { useState, useEffect } from "react";
import {
  X,
  UploadCloud,
  Film,
  AlertCircle,
  Loader2,
  Sliders,
  Play,
  Radio,
  Clock,
  Sparkles,
} from "lucide-react";
import { startVideoStream, fetchSampleVideos } from "../services/api";

export default function VideoUploadModal({
  isOpen,
  onClose,
  zones,
  onStreamStarted,
}) {
  const [file, setFile] = useState(null);
  const [selectedZone, setSelectedZone] = useState("active_work_zone");
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [sampleVideos, setSampleVideos] = useState([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [launchingFilename, setLaunchingFilename] = useState(null);

  // Fetch available preloaded sample clips when modal opens
  useEffect(() => {
    if (isOpen) {
      setLoadingSamples(true);
      fetchSampleVideos()
        .then((data) => {
          setSampleVideos(Array.isArray(data) ? data : []);
        })
        .catch((err) => {
          console.warn("Failed to load sample clips:", err);
        })
        .finally(() => {
          setLoadingSamples(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setErrorMsg("");
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setErrorMsg("");
    }
  };

  // Launch a preloaded industrial CCTV clip with 1-click
  const handleLaunchSample = async (sampleFilename) => {
    setIsUploading(true);
    setLaunchingFilename(sampleFilename);
    setErrorMsg("");

    try {
      const data = await startVideoStream(null, selectedZone, sampleFilename);
      if (onStreamStarted) {
        onStreamStarted({ filename: sampleFilename, zone_id: selectedZone, ...data });
      }
      onClose();
    } catch (err) {
      setErrorMsg(err.message || "Failed to start sample video stream.");
      setIsUploading(false);
      setLaunchingFilename(null);
    }
  };

  // Launch a custom uploaded video file
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setErrorMsg("Please select a valid video file (.mp4, .avi, or .mov).");
      return;
    }

    setIsUploading(true);
    setLaunchingFilename(null);
    setErrorMsg("");

    try {
      const data = await startVideoStream(file, selectedZone);
      // Close modal immediately and switch the main feed to live video stream
      if (onStreamStarted) {
        onStreamStarted({ filename: file.name, zone_id: selectedZone, ...data });
      }
      onClose();
    } catch (err) {
      setErrorMsg(err.message || "Failed to start video stream. Check format and try again.");
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/65 backdrop-blur-xs p-4">
      <div className="bg-white rounded-xl shadow-2xl border border-slate-300 max-w-2xl w-full overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="px-5 py-3.5 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center space-x-2">
            <Radio className="w-5 h-5 text-blue-600 animate-pulse" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Upload Video &mdash; Live Stream Inspection
              </h3>
              <p className="text-xs text-slate-500">
                Streams annotated bounding boxes in real-time &bull; ESP32 + Attendance Registry + DB active
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isUploading}
            className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-200 transition-colors disabled:opacity-40"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* How it works banner */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-blue-800 text-[11px] space-y-1">
              <div className="font-bold text-xs flex items-center space-x-1.5">
                <Radio className="w-3.5 h-3.5" />
                <span>How Live Stream Mode Works</span>
              </div>
              <ul className="list-disc ml-4 space-y-0.5 text-blue-700">
                <li>Video is streamed &amp; processed <strong>frame-by-frame</strong> in real-time</li>
                <li>Annotated feed replaces the camera panel &mdash; bounding boxes, Track IDs, PPE badges</li>
                <li>ESP32 buzzer/light reacts to violations <strong>exactly like the live webcam</strong></li>
                <li>Tracked workers are automatically logged to the <strong>Shift Attendance &amp; PPE Registry</strong></li>
                <li>WebSocket telemetry &amp; live compliance dashboards update synchronously</li>
              </ul>
            </div>

            {/* Target Zone Selection */}
            <div className="space-y-1 bg-slate-50 p-3 rounded-lg border border-slate-200">
              <label className="font-bold text-slate-800 block text-xs">
                1. Target Plant Zone Policy:
              </label>
              <select
                value={selectedZone}
                onChange={(e) => setSelectedZone(e.target.value)}
                disabled={isUploading}
                className="w-full bg-white border border-slate-300 rounded p-2 text-slate-800 font-medium focus:ring-1 focus:ring-blue-500 focus:outline-none"
              >
                {zones.map((z) => (
                  <option key={z.zone_id} value={z.zone_id}>
                    {z.name} (Requires: {Array.isArray(z.required_ppe) ? z.required_ppe.join(", ") : z.required_ppe})
                  </option>
                ))}
              </select>
            </div>

            {/* Preloaded Industrial CCTV Clips (1-Click Selector) */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="font-bold text-slate-800 text-xs flex items-center space-x-1.5">
                  <Film className="w-4 h-4 text-blue-600" />
                  <span>Preloaded Industrial CCTV Clips (1-Click Instant Launch)</span>
                </label>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 font-bold border border-blue-200">
                  {sampleVideos.length} Clips Available
                </span>
              </div>

              {loadingSamples ? (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-center text-slate-500 flex items-center justify-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
                  <span>Loading preloaded sample clips...</span>
                </div>
              ) : sampleVideos.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  {sampleVideos.map((sample) => {
                    const isCurrentLaunching = launchingFilename === sample.filename;
                    return (
                      <div
                        key={sample.filename}
                        className="p-3 bg-slate-50 hover:bg-blue-50/50 border border-slate-200 hover:border-blue-300 rounded-lg transition-all flex flex-col justify-between space-y-2.5 shadow-2xs"
                      >
                        <div className="space-y-1">
                          <div className="flex items-start justify-between gap-1.5">
                            <span className="font-bold text-slate-900 text-xs line-clamp-1">
                              {sample.title || sample.filename}
                            </span>
                            <span className="shrink-0 text-[10px] font-mono px-1.5 py-0.2 bg-slate-200 text-slate-700 rounded font-semibold">
                              {sample.duration || "Demo"}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-500 line-clamp-2">
                            {sample.description}
                          </p>
                        </div>

                        <div className="flex items-center justify-between pt-1.5 border-t border-slate-200">
                          <span className="text-[10px] text-slate-400 font-mono">
                            {sample.size_mb ? `${sample.size_mb} MB &bull; ` : ""}{sample.category || "CCTV"}
                          </span>

                          <button
                            type="button"
                            disabled={isUploading}
                            onClick={() => handleLaunchSample(sample.filename)}
                            className="px-2.5 py-1 rounded text-xs font-bold bg-blue-600 hover:bg-blue-700 text-white flex items-center space-x-1.5 shadow-xs transition-colors cursor-pointer disabled:opacity-50"
                          >
                            {isCurrentLaunching ? (
                              <>
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                <span>Starting...</span>
                              </>
                            ) : (
                              <>
                                <Play className="w-3 h-3 fill-white" />
                                <span>1-Click Launch</span>
                              </>
                            )}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="p-3 bg-slate-50 border border-slate-200 rounded text-center text-slate-500 text-[11px]">
                  No preloaded sample clips found in upload directory.
                </div>
              )}
            </div>

            {/* Divider */}
            <div className="relative flex items-center py-1">
              <div className="flex-grow border-t border-slate-200"></div>
              <span className="shrink mx-3 text-slate-400 text-[10px] uppercase font-bold tracking-wider font-mono">
                Or Upload Custom Video File
              </span>
              <div className="flex-grow border-t border-slate-200"></div>
            </div>

            {/* Drag and Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
                file
                  ? "border-blue-400 bg-blue-50/40"
                  : "border-slate-300 hover:border-blue-400 bg-slate-50 hover:bg-blue-50/20"
              }`}
              onClick={() => document.getElementById("video-file-input-stream").click()}
            >
              <input
                id="video-file-input-stream"
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                className="hidden"
                onChange={handleFileChange}
              />
              <UploadCloud className="w-9 h-9 text-blue-600 mb-2" />
              {file ? (
                <div className="space-y-1">
                  <div className="font-semibold text-slate-900 text-xs">{file.name}</div>
                  <div className="text-[11px] text-slate-500 font-mono">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB &bull; {file.type || "video/mp4"}
                  </div>
                  <span className="text-xs text-blue-600 font-semibold underline block pt-1">
                    Click to replace file
                  </span>
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="font-semibold text-slate-800 text-xs">
                    Drop custom industrial CCTV clip here, or browse
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono">
                    Supports .MP4, .AVI, .MOV, .MKV, .WEBM
                  </div>
                </div>
              )}
            </div>

            {/* Error Alert */}
            {errorMsg && (
              <div className="bg-red-50 border border-red-200 text-red-800 px-3 py-2 rounded text-xs flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Upload Progress State */}
            {isUploading && (
              <div className="bg-blue-50 border border-blue-200 text-blue-900 p-4 rounded-lg space-y-2 text-center animate-pulse">
                <Loader2 className="w-6 h-6 animate-spin mx-auto text-blue-600" />
                <div className="font-semibold text-xs">
                  {launchingFilename ? `Starting stream for '${launchingFilename}'...` : "Uploading & starting live stream pipeline..."}
                </div>
                <div className="text-[11px] text-blue-700 font-mono">
                  Initializing YOLO &bull; ByteTrack &bull; ESP32 &bull; Attendance Registry
                </div>
              </div>
            )}

            {/* Submit Buttons */}
            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                disabled={isUploading}
                className="px-3.5 py-1.5 rounded text-xs font-medium text-slate-700 bg-white border border-slate-300 hover:bg-slate-100 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!file || isUploading}
                className="px-4 py-1.5 rounded text-xs font-semibold bg-blue-600 hover:bg-blue-700 text-white shadow-xs transition-colors disabled:opacity-50 flex items-center space-x-1.5"
              >
                {isUploading && !launchingFilename ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Starting Stream...</span>
                  </>
                ) : (
                  <>
                    <Radio className="w-3.5 h-3.5" />
                    <span>▶ Launch Uploaded Video</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
