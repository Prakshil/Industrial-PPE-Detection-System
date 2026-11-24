"""
main.py — Launches the Unified Industrial PPE Dashboard & Edge Vision Platform.
Integrates the laptop webcam directly inside the web dashboard (http://localhost:8000/).
No separate popup OpenCV window!
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add project root and ppe_monitor directories to sys.path
_ROOT = Path(__file__).resolve().parent
_PM = _ROOT / "ppe_monitor"
_INNER_PM = _PM / "ppe_monitor"
for p in [str(_INNER_PM), str(_PM), str(_ROOT)]:
    if p in sys.path:
        sys.path.remove(p)
# Project root MUST be first so ppe_monitor.api is cleanly resolved
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_PM))
sys.path.append(str(_INNER_PM))


def start_dashboard(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True):
    """Starts the FastAPI backend and serves the integrated web dashboard."""
    import uvicorn
    from ppe_monitor.api.main import app

    url = f"http://127.0.0.1:{port}"
    print("=" * 80)
    print("   INDUSTRIAL PPE SAFETY INTELLIGENCE PLATFORM (EDGE v2.0)")
    print(f"   Dashboard URL:  {url}  [Important: Use HTTP, not HTTPS]")
    print("   Webcam Mode:    INTEGRATED IN DASHBOARD (No external window)")
    print("   Inspection:     Live 4-Parameter Checklist (Helmet, Vest, Boots, Gloves)")
    print("   Audit Export:   OSHA 1910.132 & ISO 45001 PDF Report Downloadable")
    print("   Hardware Link:  ESP32 Serial Bridge on COM10")
    print("=" * 80)

    if open_browser:
        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")


def run_cli_headless(
    source: str = "0",
    model_path: str = "PPE_detection/runs/ppe_exp_v3/ultra/weights/best.pt",
    zone_id: str = "active_work_zone",
    config_path: str = "config/zones.yaml",
    serial_port: str | None = "COM10",
    conf_threshold: float = 0.20,
    display: bool = False,
):
    """Headless CLI runner for headless edge servers or headless CI/CD."""
    from ppe_monitor.main import run as run_pipeline
    run_pipeline(
        source=source,
        model_path=model_path,
        zone_id=zone_id,
        config_path=config_path,
        serial_port=serial_port,
        conf_threshold=conf_threshold,
        display=display,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Industrial PPE Safety Command Center")
    ap.add_argument("--cli", action="store_true",
                    help="Run in headless CLI mode instead of web dashboard")
    ap.add_argument("--source", default="0",
                    help="Camera index or stream URL (for --cli mode)")
    ap.add_argument("--port", type=int, default=8000,
                    help="Web dashboard port (default: 8000)")
    ap.add_argument("--host", default="0.0.0.0",
                    help="Web dashboard host (default: 0.0.0.0)")
    ap.add_argument("--no-browser", action="store_true",
                    help="Do not auto-open browser on dashboard startup")
    ap.add_argument("--display", action="store_true",
                    help="Enable separate OpenCV window (only applicable in --cli mode)")
    args = ap.parse_args()

    if args.cli:
        run_cli_headless(source=args.source, display=args.display)
    else:
        start_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
