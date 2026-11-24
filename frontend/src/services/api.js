/**
 * API service for communicating with the FastAPI industrial backend.
 * Dynamically resolves host to support local dev (Vite port 5173),
 * local production (FastAPI port 8000), and LAN/edge deployments.
 */

const isDev = typeof window !== "undefined" && window.location.port === "5173";
const host = typeof window !== "undefined" ? window.location.hostname || "localhost" : "localhost";
const protocol = typeof window !== "undefined" && window.location.protocol === "https:" ? "https:" : "http:";
const wsProtocol = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss:" : "ws:";

export const API_BASE = isDev
  ? `http://${host}:8000/api/v1`
  : typeof window !== "undefined"
  ? `${protocol}//${window.location.host}/api/v1`
  : "http://localhost:8000/api/v1";

export const WS_BASE = isDev
  ? `ws://${host}:8000/ws/telemetry`
  : typeof window !== "undefined"
  ? `${wsProtocol}//${window.location.host}/ws/telemetry`
  : "ws://localhost:8000/ws/telemetry";

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchZones() {
  const res = await fetch(`${API_BASE}/zones`);
  return res.json();
}

export async function updateZonePolicy(zoneId, policy) {
  const res = await fetch(`${API_BASE}/zones/${zoneId}/policy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(policy),
  });
  return res.json();
}

export async function fetchCameras() {
  const res = await fetch(`${API_BASE}/cameras`);
  return res.json();
}

export async function fetchViolations(zoneId = "all", status = "all", limit = 50) {
  const params = new URLSearchParams({ zone_id: zoneId, status, limit });
  const res = await fetch(`${API_BASE}/violations?${params.toString()}`);
  return res.json();
}

export async function fetchViolationById(id) {
  const res = await fetch(`${API_BASE}/violations/${id}`);
  return res.json();
}

export async function triageViolation(id, action, reviewedBy = "Safety Officer", notes = "") {
  const res = await fetch(`${API_BASE}/violations/${id}/action`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, reviewed_by: reviewedBy, notes }),
  });
  return res.json();
}

export async function deleteViolation(id) {
  const res = await fetch(`${API_BASE}/violations/${id}`, {
    method: "DELETE",
  });
  return res.json();
}

export async function clearAllViolations() {
  const res = await fetch(`${API_BASE}/violations`, {
    method: "DELETE",
  });
  return res.json();
}

export async function fetchTelemetry(hours = 24) {
  const res = await fetch(`${API_BASE}/telemetry?hours=${hours}`);
  return res.json();
}

export async function uploadAndProcessVideo(file, zoneId = "active_work_zone", maxFrames = 250) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("zone_id", zoneId);
  formData.append("max_frames", maxFrames.toString());

  const res = await fetch(`${API_BASE}/video/upload-and-process`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Video upload and analysis failed.");
  }
  return res.json();
}

export async function fetchLiveCameraStatus() {
  const res = await fetch(`${API_BASE}/camera/live/status`);
  return res.json();
}

export async function startLiveCamera(cameraIndex = 0, serialPort = "COM10", mode = "all_ppe", cameraProfile = "upper_body") {
  const res = await fetch(`${API_BASE}/camera/live/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_index: cameraIndex, serial_port: serialPort, mode, camera_profile: cameraProfile }),
  });
  return res.json();
}

export async function setLiveCameraProfile(cameraProfile) {
  const res = await fetch(`${API_BASE}/camera/live/profile`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_profile: cameraProfile }),
  });
  if (!res.ok) throw new Error("Unable to set camera profile.");
  return res.json();
}

export async function stopLiveCamera() {
  const res = await fetch(`${API_BASE}/camera/live/stop`, { method: "POST" });
  return res.json();
}

export async function setLiveCameraMode(mode = "all_ppe") {
  const res = await fetch(`${API_BASE}/camera/live/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  return res.json();
}

export async function setBuzzerPolarity(activeLow = false) {
  const res = await fetch(`${API_BASE}/camera/live/buzzer-polarity`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active_low: activeLow }),
  });
  return res.json();
}

export async function reconnectLiveSerial(port = "COM10") {
  const res = await fetch(`${API_BASE}/camera/live/reconnect-serial`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  return res.json();
}

export function getExportAuditUrl(format = "csv", zoneId = "all") {
  return `${API_BASE}/export/audit?format=${format}&zone_id=${zoneId}`;
}

export function getEvidenceImageUrl(filename) {
  if (!filename) return "";
  const cleanName = filename.replace(/^evidence[\/\\]/, "");
  return `${API_BASE}/evidence/${cleanName}`;
}

export function getCameraStreamUrl(zoneId = "active_work_zone", cameraId = "") {
  const camParam = cameraId ? `&camera_id=${encodeURIComponent(cameraId)}` : "";
  return `${API_BASE}/stream/feed?zone=${encodeURIComponent(zoneId)}${camParam}`;
}

export async function fetchAttendance(sessionId = "all", status = "all", zoneId = "all", limit = 100) {
  const params = new URLSearchParams({ session_id: sessionId, status, zone_id: zoneId, limit });
  const res = await fetch(`${API_BASE}/attendance?${params.toString()}`);
  return res.json();
}

export async function clearAttendance() {
  const res = await fetch(`${API_BASE}/attendance`, { method: "DELETE" });
  return res.json();
}

export function getExportAttendanceUrl(format = "csv", sessionId = "all", status = "all") {
  return `${API_BASE}/export/attendance?format=${format}&session_id=${sessionId}&status=${status}`;
}

export async function fetchSampleVideos() {
  const res = await fetch(`${API_BASE}/videos/sample-list`);
  return res.json();
}

export function getProcessedVideoUrl() {
  return `${API_BASE}/video/processed`;
}

export function getVideoStreamFeedUrl() {
  return `${API_BASE}/video/stream-feed`;
}

export async function startVideoStream(file, zoneId = "active_work_zone", sampleFilename = null) {
  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  }
  if (sampleFilename) {
    formData.append("sample_filename", sampleFilename);
  }
  formData.append("zone_id", zoneId);
  const res = await fetch(`${API_BASE}/video/start-stream`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start video stream.");
  }
  return res.json();
}

export async function fetchVideoStreamStatus() {
  const res = await fetch(`${API_BASE}/video/stream-status`);
  return res.json();
}

export async function stopVideoStream() {
  const res = await fetch(`${API_BASE}/video/stream-stop`, { method: "POST" });
  return res.json();
}

export function connectTelemetryWebSocket(onMessage, onError) {
  let ws = null;
  let retryTimer = null;

  function connect() {
    try {
      ws = new WebSocket(WS_BASE);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (e) {
          console.error("WS Parse error:", e);
        }
      };
      ws.onerror = (err) => {
        if (onError) onError(err);
      };
      ws.onclose = () => {
        retryTimer = setTimeout(connect, 3000);
      };
    } catch (e) {
      retryTimer = setTimeout(connect, 3000);
    }
  }

  connect();

  return () => {
    if (retryTimer) clearTimeout(retryTimer);
    if (ws) ws.close();
  };
}

