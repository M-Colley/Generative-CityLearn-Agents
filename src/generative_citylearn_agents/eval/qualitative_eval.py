from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from generative_citylearn_agents.util.storage import Storage


def evaluate_qualitative_artifacts(storage_dir: Union[str, Path]) -> Dict[str, Any]:
    """Inspect saved persona, daily schedule, and rationale artifacts (GTA ``qualitative_eval.py``)."""
    base = Path(storage_dir)
    descriptions_file = base / Storage.DESCRIPTIONS_FILENAME
    schedules_file = base / Storage.DAY_SCHEDULES_FILENAME
    decisions_file = base / Storage.DECISIONS_FILENAME

    descriptions: Dict[str, Any] = (
        json.loads(descriptions_file.read_text(encoding="utf-8"))
        if descriptions_file.is_file()
        else {}
    )
    schedules: List[Dict[str, Any]] = (
        json.loads(schedules_file.read_text(encoding="utf-8"))
        if schedules_file.is_file()
        else []
    )
    decisions: List[Dict[str, Any]] = (
        json.loads(decisions_file.read_text(encoding="utf-8"))
        if decisions_file.is_file()
        else []
    )

    archetypes: Dict[str, int] = {}
    for prof in descriptions.values():
        seed = prof.get("seed_profile") or {}
        arch = str(seed.get("occupant_archetype", "unknown"))
        archetypes[arch] = archetypes.get(arch, 0) + 1

    total_act_decisions = 0
    clamped = 0
    fallbacks = 0
    rationales: List[str] = []
    for rec in decisions:
        for dec in rec.get("decisions", []):
            total_act_decisions += 1
            if dec.get("clamped"):
                clamped += 1
            if dec.get("used_fallback"):
                fallbacks += 1
            reason = str(dec.get("reasoning", "")).strip()
            if reason and len(rationales) < 10:
                rationales.append(f"[{rec.get('building_name')} t={rec.get('time_step')}] {reason}")

    return {
        "storage_dir": str(base),
        "num_personas": len(descriptions),
        "archetype_counts": archetypes,
        "num_day_schedules": len(schedules),
        "num_step_records": len(decisions),
        "total_actuator_decisions": total_act_decisions,
        "clamped_ratio": (clamped / total_act_decisions) if total_act_decisions else 0.0,
        "fallback_ratio": (fallbacks / total_act_decisions) if total_act_decisions else 0.0,
        "sample_rationales": rationales,
    }
