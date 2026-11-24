"""
Zone Policy Loader — reads YAML configuration and provides required PPE
lists and per-zone settings to the compliance engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ZonePolicy:
    zone_id: str
    description: str
    camera_id: str
    required_ppe: list[str]
    confirmation_frames: int
    alert_cooldown_seconds: int
    confirmation_seconds: float = 1.0


class PolicyLoader:
    """Loads and caches zone policies from a YAML file."""

    def __init__(self, yaml_path: str | Path):
        path = Path(yaml_path)
        if not path.is_file():
            # Fallback 1: relative to package root
            cand1 = Path(__file__).resolve().parent.parent.parent / yaml_path
            if cand1.is_file():
                path = cand1
            else:
                # Fallback 2: inside ppe_monitor/config/
                cand2 = Path(__file__).resolve().parent.parent.parent / "config" / Path(yaml_path).name
                if cand2.is_file():
                    path = cand2
        self._path = path
        self._body_regions: dict[str, tuple[float, float]] = {}
        self._policies: dict[str, ZonePolicy] = {}
        self._load()

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)

        # Load body region ratios
        for ppe_class, ratios in data.get("body_regions", {}).items():
            self._body_regions[ppe_class] = (float(ratios[0]), float(ratios[1]))

        # Load zone policies
        for zone_id, cfg in data.get("zones", {}).items():
            self._policies[zone_id] = ZonePolicy(
                zone_id=zone_id,
                description=cfg.get("description", ""),
                camera_id=cfg.get("camera_id", ""),
                required_ppe=cfg.get("required_ppe", []),
                confirmation_frames=int(cfg.get("confirmation_frames", 5)),
                alert_cooldown_seconds=int(cfg.get("alert_cooldown_seconds", 30)),
                confirmation_seconds=float(cfg.get("confirmation_seconds", 1.0)),
            )

    def get_policy(self, zone_id: str) -> ZonePolicy:
        if zone_id not in self._policies:
            raise KeyError(f"Zone '{zone_id}' not found in {self._path}")
        return self._policies[zone_id]

    def get_body_regions(self) -> dict[str, tuple[float, float]]:
        return dict(self._body_regions)

    def available_zones(self) -> list[str]:
        return list(self._policies.keys())

    def update_runtime_policy(
        self,
        zone_id: str,
        required_ppe: list[str] | None = None,
        confirmation_frames: int | None = None,
        confirmation_seconds: float | None = None,
        alert_cooldown_seconds: int | None = None,
    ) -> None:
        """Apply an already-validated dashboard policy to this edge process."""
        policy = self.get_policy(zone_id)
        if required_ppe is not None:
            policy.required_ppe = list(required_ppe)
        if confirmation_frames is not None:
            policy.confirmation_frames = confirmation_frames
        if confirmation_seconds is not None:
            policy.confirmation_seconds = confirmation_seconds
        if alert_cooldown_seconds is not None:
            policy.alert_cooldown_seconds = alert_cooldown_seconds

