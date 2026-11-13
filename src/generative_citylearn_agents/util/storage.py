from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from generative_citylearn_agents.model.agent import StepDecisionRecord
from generative_citylearn_agents.model.building import BuildingProfile
from generative_citylearn_agents.model.day_schedule import DayEnergyPlan


class Storage:
    """Persist and reload GTA-style stage checkpoints and decision logs.

    Mirrors ``util.storage.Storage`` from ``ciao-group/Generative-Traffic-Agents``:
    - ``1_descriptions.json`` / ``1_no_description.json``
    - ``2_day_schedules.json`` / ``2_no_day_schedule.json``
    - ``3_decisions.json``
    - ``4_kpi_summary.json``
    """

    DESCRIPTIONS_FILENAME = "1_descriptions.json"
    NO_DESCRIPTIONS_FILENAME = "1_no_description.json"
    DAY_SCHEDULES_FILENAME = "2_day_schedules.json"
    NO_DAY_SCHEDULES_FILENAME = "2_no_day_schedule.json"
    DECISIONS_FILENAME = "3_decisions.json"
    KPI_SUMMARY_FILENAME = "4_kpi_summary.json"

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        *,
        load_from_storage: bool = False,
        auto_flush: bool = True,
    ):
        self.storage_dir: Optional[Path] = None if storage_dir is None else Path(storage_dir)
        self.load_from_storage = bool(load_from_storage)
        self.auto_flush = bool(auto_flush)

        self.descriptions: Dict[str, Dict[str, Any]] = {}
        self.failed_descriptions: List[Dict[str, Any]] = []
        self.day_schedules: List[Dict[str, Any]] = []
        self.failed_day_schedules: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.kpi_summary: List[Dict[str, Any]] = []

        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            if self.load_from_storage:
                self._load_existing()

    def _load_existing(self) -> None:
        if self.descriptions_path and self.descriptions_path.is_file():
            loaded = json.loads(self.descriptions_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.descriptions.update(loaded)
        if self.day_schedules_path and self.day_schedules_path.is_file():
            loaded = json.loads(self.day_schedules_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                self.day_schedules.extend(loaded)

    @property
    def descriptions_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.DESCRIPTIONS_FILENAME

    @property
    def no_descriptions_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.NO_DESCRIPTIONS_FILENAME

    @property
    def day_schedules_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.DAY_SCHEDULES_FILENAME

    @property
    def no_day_schedules_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.NO_DAY_SCHEDULES_FILENAME

    @property
    def decisions_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.DECISIONS_FILENAME

    @property
    def kpi_summary_path(self) -> Optional[Path]:
        return None if self.storage_dir is None else self.storage_dir / self.KPI_SUMMARY_FILENAME

    def reset_episode_logs(self, *, keep_descriptions: bool = True) -> None:
        if not keep_descriptions:
            self.descriptions.clear()
            self.failed_descriptions.clear()
        self.day_schedules.clear()
        self.failed_day_schedules.clear()
        self.decisions.clear()
        self.kpi_summary.clear()
        if self.auto_flush:
            self.flush()

    def record_profile(self, profile: BuildingProfile) -> None:
        payload = profile.to_dict()
        self.descriptions[profile.building_name] = payload
        if profile.used_fallback:
            self.failed_descriptions.append(payload)
        if self.auto_flush:
            self._write_json(self.descriptions_path, self.descriptions)
            if self.failed_descriptions:
                self._write_json(self.no_descriptions_path, self.failed_descriptions)

    def record_day_plan(self, day_plan: DayEnergyPlan) -> None:
        payload = day_plan.to_dict()
        self.day_schedules.append(payload)
        if day_plan.used_fallback:
            self.failed_day_schedules.append(payload)
        if self.auto_flush:
            self._write_json(self.day_schedules_path, self.day_schedules)
            if self.failed_day_schedules:
                self._write_json(self.no_day_schedules_path, self.failed_day_schedules)

    def record_step_decision(self, record: StepDecisionRecord) -> None:
        self.decisions.append(record.to_dict())
        if self.auto_flush:
            self._write_json(self.decisions_path, self.decisions)

    def record_kpi_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.kpi_summary = [dict(r) for r in rows]
        if self.auto_flush:
            self._write_json(self.kpi_summary_path, self.kpi_summary)

    def flush(self) -> None:
        if self.storage_dir is None:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.descriptions_path, self.descriptions)
        self._write_json(self.no_descriptions_path, self.failed_descriptions)
        self._write_json(self.day_schedules_path, self.day_schedules)
        self._write_json(self.no_day_schedules_path, self.failed_day_schedules)
        self._write_json(self.decisions_path, self.decisions)
        if self.kpi_summary:
            self._write_json(self.kpi_summary_path, self.kpi_summary)

    def summarize_decisions(self) -> Dict[str, Any]:
        action_totals: Dict[str, List[float]] = {}
        fallback_count = 0
        clamped_count = 0
        sample_rationales: List[Dict[str, Any]] = []

        for record in self.decisions:
            for decision in record.get("decisions", []):
                name = str(decision.get("action_name", ""))
                val = float(decision.get("value", 0.0))
                action_totals.setdefault(name, []).append(val)
                if decision.get("used_fallback"):
                    fallback_count += 1
                if decision.get("clamped"):
                    clamped_count += 1
                if len(sample_rationales) < 5 and not record.get("reused_previous_step", False):
                    sample_rationales.append(
                        {
                            "time_step": record.get("time_step"),
                            "building_name": record.get("building_name"),
                            "action_name": name,
                            "value": val,
                            "reasoning": decision.get("reasoning", ""),
                        }
                    )

        mean_actions = {
            name: (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in sorted(action_totals.items())
        }
        return {
            "buildings_profiled": len(self.descriptions),
            "day_schedules_recorded": len(self.day_schedules),
            "step_records_recorded": len(self.decisions),
            "fallback_decisions": fallback_count,
            "clamped_decisions": clamped_count,
            "mean_action_by_name": mean_actions,
            "sample_rationales": sample_rationales,
        }

    @staticmethod
    def _write_json(path: Optional[Path], payload: Any) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


AgentArtifactStorage = Storage
