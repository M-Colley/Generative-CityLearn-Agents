from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping


@dataclass
class DayEnergyPlan:
    """Daily high-level operating plan analogous to GTA's ``day_schedule``."""

    building_name: str
    day_index: int
    day_type: int
    description_of_today: List[str]
    solar_charging_window: str
    peak_avoidance_strategy: str
    ev_charging_plan: str
    raw_response: str = ""
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_name": self.building_name,
            "day_index": int(self.day_index),
            "day_type": int(self.day_type),
            "description_of_today": list(self.description_of_today),
            "solar_charging_window": self.solar_charging_window,
            "peak_avoidance_strategy": self.peak_avoidance_strategy,
            "ev_charging_plan": self.ev_charging_plan,
            "used_fallback": bool(self.used_fallback),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DayEnergyPlan":
        return cls(
            building_name=str(data.get("building_name", "Building")),
            day_index=int(data.get("day_index", 0) or 0),
            day_type=int(data.get("day_type", 1) or 1),
            description_of_today=[str(x) for x in (data.get("description_of_today") or [])],
            solar_charging_window=str(data.get("solar_charging_window", "")),
            peak_avoidance_strategy=str(data.get("peak_avoidance_strategy", "")),
            ev_charging_plan=str(data.get("ev_charging_plan", "")),
            raw_response=str(data.get("raw_response", "")),
            used_fallback=bool(data.get("used_fallback", False)),
        )
