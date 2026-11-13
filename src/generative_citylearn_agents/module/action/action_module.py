from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


class ActionModule:
    """Extract physical building observations/action bounds and format CityLearn actions.

    Analogous to ``module/action/action_module.py`` in Generative-Traffic-Agents.
    """

    def __init__(self, *, include_all_observations: bool = False):
        self.include_all_observations = bool(include_all_observations)
        self._action_specs_cache: Dict[int, Tuple[Tuple[str, ...], List[Dict[str, Any]]]] = {}

    def clear_cache(self) -> None:
        self._action_specs_cache.clear()

    def building_observations_dict(self, building: Any) -> Dict[str, float]:
        raw = building.observations(
            include_all=self.include_all_observations,
            normalize=False,
            periodic_normalization=False,
        )
        cleaned: Dict[str, float] = {}
        for key, value in (raw or {}).items():
            try:
                scalar = float(value)
                cleaned[str(key)] = scalar if math.isfinite(scalar) else 0.0
            except (TypeError, ValueError):
                continue
        return cleaned

    def building_action_specs(self, building: Any) -> List[Dict[str, Any]]:
        active_actions = tuple(str(a) for a in (getattr(building, "active_actions", []) or []))
        cached = self._action_specs_cache.get(id(building))
        if cached is not None and cached[0] == active_actions:
            return cached[1]

        space = building.action_space
        charger_by_action: Dict[str, Any] = {}
        for charger in getattr(building, "electric_vehicle_chargers", None) or []:
            cid = str(getattr(charger, "charger_id", ""))
            charger_by_action[f"electric_vehicle_storage_{cid}"] = charger

        bess = getattr(building, "electrical_storage", None)
        bess_dod = float(getattr(bess, "depth_of_discharge", 1.0) or 1.0)
        bess_min_soc = max(0.0, min(1.0, 1.0 - bess_dod)) if 0.0 <= bess_dod <= 1.0 else 0.0

        specs: List[Dict[str, Any]] = []
        for idx, act_name in enumerate(active_actions):
            low = float(space.low[idx])
            high = float(space.high[idx])
            spec: Dict[str, Any] = {
                "action_name": act_name,
                "low": low,
                "high": high,
            }
            if act_name == "electrical_storage" and bess is not None:
                spec["capacity_kwh"] = float(getattr(bess, "capacity", 0.0) or 0.0)
                spec["nominal_power_kw"] = float(getattr(bess, "nominal_power", 0.0) or 0.0)
                spec["min_soc"] = bess_min_soc
            elif act_name in charger_by_action:
                charger = charger_by_action[act_name]
                max_p = float(getattr(charger, "max_charging_power", 0.0) or 0.0)
                min_p = float(getattr(charger, "min_charging_power", 0.0) or 0.0)
                spec["max_charging_power_kw"] = max_p
                spec["min_charging_power_kw"] = min_p
                spec["min_effective_action"] = (min_p / max_p) if max_p > 0.0 and min_p > 0.0 else 0.0
            specs.append(spec)

        self._action_specs_cache[id(building)] = (active_actions, specs)
        return specs

    @staticmethod
    def format_env_actions(
        per_building_actions: Sequence[Sequence[float]],
        *,
        central_agent: bool,
    ) -> List[List[float]]:
        if central_agent:
            return [[float(val) for vec in per_building_actions for val in vec]]
        return [[float(val) for val in vec] for vec in per_building_actions]
