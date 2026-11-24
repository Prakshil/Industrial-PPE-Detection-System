"""Top-level package forwarder for ppe_monitor inner submodules."""

from pathlib import Path

_inner = Path(__file__).resolve().parent / "ppe_monitor"
__path__ = [str(_inner), str(Path(__file__).resolve().parent)]
