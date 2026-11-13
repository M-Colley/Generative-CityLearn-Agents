from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

SENSOR_FAULT_SENTINEL_THRESHOLD = -9990.0
_EV_SENTINEL_SUFFIXES = (
    "_soc",
    "_required_soc_departure",
    "_estimated_departure_time",
    "_estimated_arrival_time",
    "_estimated_soc_arrival",
)


class ActionModule:
    """Extract physical building observations/action bounds and format CityLearn actions.

    Analogous to ``module/action/action_module.py`` in Generative-Traffic-Agents, with
    CityLearn-specific guards for:
    - strict ``agent_payload`` extraction (preserving robustness sensor blackouts and
      fair ``Comparison`` manifests),
    - both ``interface='flat'`` and ``interface='entity'`` observation/action contracts,
    - sentinel decoding (``-9999.0`` sensor faults and ``-0.1`` / ``-1`` unplugged EV sentinels),
    - hardware deadbands, strict ``> trigger_threshold`` deferrable triggers, and escalator bands.
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

    def extract_env_observations_by_building(
        self,
        env: Any,
        observations: Any,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
        """Extract per-building telemetry and structured diagnostics from ``env`` + ``observations``."""
        unwrapped = getattr(env, "unwrapped", env)
        buildings = list(getattr(unwrapped, "buildings", []) or [])
        obs_by_building: Dict[str, Dict[str, float]] = {}
        diag_by_building: Dict[str, Dict[str, Any]] = {b.name: {} for b in buildings}

        if self.include_all_observations or observations is None:
            for b in buildings:
                raw_obs = self.building_observations_dict(b)
                clean_obs, diag = self.sanitize_telemetry_for_prompt(raw_obs)
                obs_by_building[b.name] = clean_obs
                diag_by_building[b.name] = diag
            return obs_by_building, diag_by_building

        if isinstance(observations, Mapping) and "tables" in observations:
            raw_map, entity_diag = self._extract_from_entity_payload(unwrapped, observations)
            for b in buildings:
                raw_obs = raw_map.get(b.name)
                if raw_obs is None:
                    raw_obs = self.building_observations_dict(b)
                clean_obs, diag = self.sanitize_telemetry_for_prompt(raw_obs)
                merged_diag = {**diag, **entity_diag.get(b.name, {})}
                obs_by_building[b.name] = clean_obs
                diag_by_building[b.name] = merged_diag
            return obs_by_building, diag_by_building

        flat_map = self._extract_from_flat_payload(unwrapped, observations)
        for b in buildings:
            raw_obs = flat_map.get(b.name)
            if raw_obs is None:
                raw_obs = self.building_observations_dict(b)
            clean_obs, diag = self.sanitize_telemetry_for_prompt(raw_obs)
            obs_by_building[b.name] = clean_obs
            diag_by_building[b.name] = diag
        return obs_by_building, diag_by_building

    @staticmethod
    def sanitize_telemetry_for_prompt(
        raw_obs: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Separate CityLearn fault/disconnected sentinels from physical numeric telemetry."""
        cleaned: Dict[str, float] = {}
        faulted_sensors: List[str] = []
        disconnected_chargers: set[str] = set()

        for key, val in raw_obs.items():
            k = str(key)
            try:
                scalar = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar):
                continue

            if scalar <= SENSOR_FAULT_SENTINEL_THRESHOLD:
                faulted_sensors.append(k)
                continue

            if k.startswith("electric_vehicle_charger_") and scalar < 0.0:
                if any(k.endswith(suffix) for suffix in _EV_SENTINEL_SUFFIXES):
                    prefix = k[len("electric_vehicle_charger_") :]
                    for suffix in _EV_SENTINEL_SUFFIXES:
                        if prefix.endswith(suffix):
                            disconnected_chargers.add(prefix[: -len(suffix)])
                            break
                    continue

            cleaned[k] = scalar

        for k, v in cleaned.items():
            if k.startswith("electric_vehicle_charger_") and k.endswith("_connected_state") and v < 0.5:
                cid = k[len("electric_vehicle_charger_") : -len("_connected_state")]
                disconnected_chargers.add(cid)

        diag: Dict[str, Any] = {}
        if faulted_sensors:
            diag["faulted_sensors"] = sorted(faulted_sensors)
        if disconnected_chargers:
            diag["disconnected_ev_chargers"] = sorted(disconnected_chargers)
        return cleaned, diag

    @staticmethod
    def _extract_from_flat_payload(
        env: Any,
        observations: Any,
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        buildings = list(getattr(env, "buildings", []) or [])
        obs_names = getattr(env, "observation_names", None)
        if not isinstance(observations, Sequence) or not isinstance(obs_names, Sequence):
            return result

        central = bool(getattr(env, "central_agent", False))
        if not central and len(observations) == len(buildings) and len(obs_names) == len(buildings):
            for building, names, values in zip(buildings, obs_names, observations):
                if isinstance(names, Sequence) and isinstance(values, (Sequence, np.ndarray)):
                    row: Dict[str, float] = {}
                    for name, val in zip(names, values):
                        try:
                            scalar = float(val)
                            if math.isfinite(scalar):
                                row[str(name)] = scalar
                        except (TypeError, ValueError):
                            continue
                    result[building.name] = row
            return result

        if central and len(observations) == 1 and len(obs_names) == 1:
            shared_names = list(obs_names[0])
            shared_vals = list(observations[0])
            per_building_active = [list(getattr(b, "active_observations", []) or []) for b in buildings]
            if sum(len(a) for a in per_building_active) == len(shared_vals):
                cursor = 0
                for building, active in zip(buildings, per_building_active):
                    slice_vals = shared_vals[cursor : cursor + len(active)]
                    cursor += len(active)
                    result[building.name] = {
                        str(k): float(v)
                        for k, v in zip(active, slice_vals)
                        if isinstance(v, (int, float, np.number)) and math.isfinite(float(v))
                    }
            else:
                shared_dict = {
                    str(k): float(v)
                    for k, v in zip(shared_names, shared_vals)
                    if isinstance(v, (int, float, np.number)) and math.isfinite(float(v))
                }
                for building in buildings:
                    result[building.name] = dict(shared_dict)
        return result

    @staticmethod
    def _extract_from_entity_payload(
        env: Any,
        payload: Mapping[str, Any],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
        obs_by_building: Dict[str, Dict[str, float]] = {}
        diag_by_building: Dict[str, Dict[str, Any]] = {}
        specs = getattr(env, "entity_specs", None)
        tables = payload.get("tables")
        if not isinstance(specs, Mapping) or not isinstance(tables, Mapping):
            return obs_by_building, diag_by_building

        table_specs = specs.get("tables") or {}
        building_spec = table_specs.get("building") or {}
        building_ids = list(building_spec.get("entity_ids") or [])
        building_features = list(building_spec.get("features") or [])
        building_arr = tables.get("building")

        community_row: Dict[str, float] = {}
        comm_spec = table_specs.get("community") or {}
        comm_features = list(comm_spec.get("features") or [])
        comm_arr = tables.get("community")
        if comm_arr is not None and len(comm_arr) > 0:
            for idx, feat in enumerate(comm_features):
                if idx < comm_arr.shape[1]:
                    community_row[str(feat)] = float(comm_arr[0, idx])

        meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
        dr_meta = meta.get("demand_response") if isinstance(meta.get("demand_response"), Mapping) else None

        for row_idx, b_id in enumerate(building_ids):
            row_dict: Dict[str, float] = dict(community_row)
            if building_arr is not None and row_idx < len(building_arr):
                for col_idx, feat in enumerate(building_features):
                    if col_idx < building_arr.shape[1]:
                        row_dict[str(feat)] = float(building_arr[row_idx, col_idx])
            obs_by_building[str(b_id)] = row_dict
            b_diag: Dict[str, Any] = {}
            if dr_meta:
                b_diag["demand_response"] = dict(dr_meta)
            diag_by_building[str(b_id)] = b_diag

        storage_spec = table_specs.get("storage") or {}
        storage_ids = list(storage_spec.get("entity_ids") or [])
        storage_feats = list(storage_spec.get("features") or [])
        storage_arr = tables.get("storage")
        if storage_arr is not None:
            for s_idx, s_id in enumerate(storage_ids):
                if s_idx >= len(storage_arr):
                    continue
                b_name = str(s_id).split("/", 1)[0]
                target = obs_by_building.get(b_name)
                if target is None:
                    continue
                s_row = {str(f): float(storage_arr[s_idx, c]) for c, f in enumerate(storage_feats) if c < storage_arr.shape[1]}
                if "soc" in s_row:
                    target.setdefault("electrical_storage_soc", s_row["soc"])

        charger_spec = table_specs.get("charger") or {}
        charger_ids = list(charger_spec.get("entity_ids") or [])
        charger_feats = list(charger_spec.get("features") or [])
        charger_arr = tables.get("charger")
        if charger_arr is not None:
            for c_idx, c_id in enumerate(charger_ids):
                if c_idx >= len(charger_arr) or "/" not in str(c_id):
                    continue
                b_name, cid = str(c_id).split("/", 1)
                target = obs_by_building.get(b_name)
                if target is None:
                    continue
                c_row = {str(f): float(charger_arr[c_idx, c]) for c, f in enumerate(charger_feats) if c < charger_arr.shape[1]}
                if "is_connected" in c_row:
                    target[f"electric_vehicle_charger_{cid}_connected_state"] = c_row["is_connected"]
                if "connected_ev_soc" in c_row and c_row.get("is_connected", 0.0) >= 0.5:
                    target[f"electric_vehicle_charger_{cid}_soc"] = c_row["connected_ev_soc"]
                if "connected_ev_required_soc" in c_row and c_row.get("is_connected", 0.0) >= 0.5:
                    target[f"electric_vehicle_charger_{cid}_required_soc_departure"] = c_row["connected_ev_required_soc"]
                if "hours_until_departure" in c_row and c_row.get("is_connected", 0.0) >= 0.5:
                    target[f"electric_vehicle_charger_{cid}_estimated_departure_time"] = c_row["hours_until_departure"]
                clip_reasons = [
                    k.replace("clip_reason_", "")
                    for k, v in c_row.items()
                    if k.startswith("clip_reason_") and 0.5 <= v <= 1.5
                ]
                if clip_reasons:
                    diag_by_building.setdefault(b_name, {}).setdefault("previous_step_clips", {})[cid] = clip_reasons

        return obs_by_building, diag_by_building

    def building_action_specs(self, building: Any) -> List[Dict[str, Any]]:
        active_actions = tuple(str(a) for a in (getattr(building, "active_actions", []) or []))
        cached = self._action_specs_cache.get(id(building))
        if cached is not None and cached[0] == active_actions:
            return cached[1]

        space = building.action_space
        charger_by_action: Dict[str, Any] = {}
        charger_phase_map = getattr(building, "_charger_phase_map", None) or {}
        for charger in getattr(building, "electric_vehicle_chargers", None) or []:
            cid = str(getattr(charger, "charger_id", ""))
            charger_by_action[f"electric_vehicle_storage_{cid}"] = charger

        deferrable_by_action: Dict[str, Any] = {}
        for idx_app, appliance in enumerate(getattr(building, "deferrable_appliances", None) or []):
            deferrable_by_action[f"deferrable_appliance_{idx_app}"] = appliance

        escalator_by_action: Dict[str, Any] = {}
        for idx_esc, esc in enumerate(getattr(building, "escalators", None) or []):
            escalator_by_action[f"escalator_{idx_esc}"] = esc

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
                cid = str(getattr(charger, "charger_id", ""))
                max_p = float(getattr(charger, "max_charging_power", 0.0) or 0.0)
                min_p = float(getattr(charger, "min_charging_power", 0.0) or 0.0)
                max_dis_p = float(getattr(charger, "max_discharging_power", 0.0) or 0.0)
                min_dis_p = float(getattr(charger, "min_discharging_power", 0.0) or 0.0)
                phase = charger_phase_map.get(cid) or getattr(charger, "phase_connection", None)
                spec["max_charging_power_kw"] = max_p
                spec["min_charging_power_kw"] = min_p
                spec["min_effective_action"] = (min_p / max_p) if max_p > 0.0 and min_p > 0.0 else 0.0
                if low < 0.0 and max_dis_p > 0.0:
                    spec["max_discharging_power_kw"] = max_dis_p
                    spec["min_discharging_power_kw"] = min_dis_p
                    spec["min_effective_discharge_action"] = (
                        -(min_dis_p / max_dis_p) if min_dis_p > 0.0 else 0.0
                    )
                if phase:
                    spec["phase_connection"] = str(phase)
            elif act_name in deferrable_by_action:
                appliance = deferrable_by_action[act_name]
                trig = float(getattr(appliance, "trigger_threshold", 0.5) or 0.5)
                spec["trigger_threshold_strict"] = trig
                spec["recommended_start_action"] = min(high, max(trig + 0.25, 1.0))
                spec["nominal_power_kw"] = float(getattr(appliance, "nominal_power", 0.0) or 0.0)
            elif act_name in escalator_by_action:
                esc = escalator_by_action[act_name]
                spec["quantization_bands"] = {
                    "standby": "[0.0, 0.333)",
                    "slow": "[0.333, 0.667)",
                    "normal": "[0.667, 1.0]",
                }
                spec["minimum_state_steps"] = int(getattr(esc, "minimum_state_steps", 1) or 1)
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

    @staticmethod
    def format_entity_actions(
        env: Any,
        per_building_actions: Sequence[Sequence[float]],
        action_specs_by_building: Sequence[Sequence[Mapping[str, Any]]],
    ) -> Dict[str, Any]:
        """Format actions for ``CityLearnEnv(interface='entity')`` using canonical entity map keys."""
        unwrapped = getattr(env, "unwrapped", env)
        action_space = unwrapped.action_space
        tables: Dict[str, np.ndarray] = {}
        if isinstance(action_space, Mapping) and "tables" in action_space:
            for table_name, box in action_space["tables"].items():
                tables[str(table_name)] = np.zeros(box.shape, dtype=np.float32)

        entity_map: Dict[str, Dict[str, float]] = {}
        for building, vec, specs in zip(unwrapped.buildings, per_building_actions, action_specs_by_building):
            b_payload: Dict[str, float] = {}
            for val, spec in zip(vec, specs):
                act_name = str(spec["action_name"])
                scalar = float(val)
                if act_name.startswith("electric_vehicle_storage_"):
                    cid = act_name[len("electric_vehicle_storage_") :]
                    entity_map[f"charger:{building.name}/{cid}"] = {
                        "electric_vehicle_storage": scalar
                    }
                else:
                    b_payload[act_name] = scalar
            if b_payload:
                entity_map[f"building:{building.name}"] = b_payload

        return {"tables": tables, "map": entity_map}
