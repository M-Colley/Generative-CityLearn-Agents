from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class BuildingSeedProfile:
    """Static equipment and load-profile seed attributes extracted for one building."""

    building_name: str
    occupant_archetype: str
    equipment_summary: Dict[str, Any]
    active_actions: List[str]
    action_bounds: Dict[str, List[float]]
    annual_demands_kwh: Dict[str, float]
    pv_nominal_power_kw: float = 0.0
    battery_capacity_kwh: float = 0.0
    battery_nominal_power_kw: float = 0.0
    battery_min_soc: float = 0.0
    ev_chargers: List[Dict[str, Any]] = field(default_factory=list)
    deferrable_appliances: List[Dict[str, Any]] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_name": self.building_name,
            "occupant_archetype": self.occupant_archetype,
            "equipment_summary": dict(self.equipment_summary),
            "active_actions": list(self.active_actions),
            "action_bounds": {k: list(v) for k, v in self.action_bounds.items()},
            "annual_demands_kwh": dict(self.annual_demands_kwh),
            "pv_nominal_power_kw": float(self.pv_nominal_power_kw),
            "battery_capacity_kwh": float(self.battery_capacity_kwh),
            "battery_nominal_power_kw": float(self.battery_nominal_power_kw),
            "battery_min_soc": float(self.battery_min_soc),
            "ev_chargers": [dict(item) for item in self.ev_chargers],
            "deferrable_appliances": [dict(item) for item in self.deferrable_appliances],
            "custom_attributes": dict(self.custom_attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BuildingSeedProfile":
        return cls(
            building_name=str(data.get("building_name", "Building")),
            occupant_archetype=str(data.get("occupant_archetype", "prosumer")),
            equipment_summary=dict(data.get("equipment_summary") or {}),
            active_actions=[str(x) for x in (data.get("active_actions") or [])],
            action_bounds={
                str(k): [float(v[0]), float(v[1])]
                for k, v in (data.get("action_bounds") or {}).items()
            },
            annual_demands_kwh={
                str(k): float(v) for k, v in (data.get("annual_demands_kwh") or {}).items()
            },
            pv_nominal_power_kw=float(data.get("pv_nominal_power_kw", 0.0) or 0.0),
            battery_capacity_kwh=float(data.get("battery_capacity_kwh", 0.0) or 0.0),
            battery_nominal_power_kw=float(data.get("battery_nominal_power_kw", 0.0) or 0.0),
            battery_min_soc=float(data.get("battery_min_soc", 0.0) or 0.0),
            ev_chargers=[dict(x) for x in (data.get("ev_chargers") or [])],
            deferrable_appliances=[dict(x) for x in (data.get("deferrable_appliances") or [])],
            custom_attributes=dict(data.get("custom_attributes") or {}),
        )
