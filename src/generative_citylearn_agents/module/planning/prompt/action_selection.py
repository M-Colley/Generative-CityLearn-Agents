from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from generative_citylearn_agents.llm.base_chat_api import format_context_block
from generative_citylearn_agents.model.agent import StepDecisionRecord
from generative_citylearn_agents.model.building import BuildingProfile
from generative_citylearn_agents.model.day_schedule import DayEnergyPlan


def get_action_selection_prompt(
    *,
    profile: BuildingProfile,
    day_plan: Optional[DayEnergyPlan],
    time_step: int,
    observations: Mapping[str, float],
    action_specs: Sequence[Mapping[str, Any]],
    recent_decisions: Optional[Sequence[StepDecisionRecord]] = None,
) -> str:
    """Build Stage-3 step-level control decision prompt modeled after GTA's ``PlanningModule``."""
    history_tail: List[Dict[str, Any]] = []
    for rec in (recent_decisions or [])[-2:]:
        history_tail.append(
            {
                "time_step": rec.time_step,
                "hour": rec.hour,
                "actions": {d.action_name: d.value for d in rec.decisions},
            }
        )

    context = {
        "prompt_stage": "action_selection",
        "building_name": profile.building_name,
        "time_step": int(time_step),
        "persona_description": profile.persona_description,
        "energy_priorities": profile.energy_priorities,
        "day_plan": None
        if day_plan is None
        else {
            "day_index": day_plan.day_index,
            "description_of_today": day_plan.description_of_today,
            "solar_charging_window": day_plan.solar_charging_window,
            "peak_avoidance_strategy": day_plan.peak_avoidance_strategy,
            "ev_charging_plan": day_plan.ev_charging_plan,
        },
        "observations": dict(observations),
        "action_specs": [dict(spec) for spec in action_specs],
        "recent_decisions": history_tail,
    }

    return (
        f"Select continuous control actions for building '{profile.building_name}' "
        f"at time_step={time_step}.\n\n"
        "Physical action semantics:\n"
        "- Storage actions (`electrical_storage`, `cooling_storage`, `heating_storage`, `dhw_storage`): "
        "fraction of storage capacity charged (>0) or discharged (<0) in this step, bounded by `[low, high]`.\n"
        "- EV charger actions (`electric_vehicle_storage_<id>`): fraction of charger power (>0 charges, "
        "<0 V2G discharge when supported). Note `min_effective_action`: positive values strictly below "
        "`min_effective_action` fall inside the hardware deadband and deliver 0 kW, so use `0.0` or "
        "`>= min_effective_action`.\n"
        "- Deferrable appliances (`deferrable_appliance_<idx>`): `>= 0.5` starts a cycle if pending.\n"
        "- HVAC device actions (`cooling_device`, `heating_device`, `cooling_or_heating_device`): "
        "fraction of nominal device power.\n\n"
        f"{format_context_block(context)}\n\n"
        "Return a JSON object containing one entry per item in `action_specs`:\n"
        "{\n"
        '  "decisions": [\n'
        "    {\n"
        '      "action_name": "<exact action_name from action_specs>",\n'
        '      "reasoning": "<1 concise sentence citing current observations/plan>",\n'
        '      "value": 0.0\n'
        "    }\n"
        "  ]\n"
        "}"
    )
