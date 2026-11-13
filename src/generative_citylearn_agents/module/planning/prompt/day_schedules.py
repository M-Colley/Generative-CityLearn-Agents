from __future__ import annotations

from typing import Mapping

from generative_citylearn_agents.llm.base_chat_api import format_context_block
from generative_citylearn_agents.model.building import BuildingProfile


def get_day_schedule_prompt(
    profile: BuildingProfile,
    day_index: int,
    observations: Mapping[str, float],
) -> str:
    """Build Stage-2 daily energy schedule prompt modeled after GTA's ``day_schedules.py``."""
    day_type = int(round(float(observations.get("day_type", 1.0))))
    context = {
        "prompt_stage": "day_schedule",
        "building_name": profile.building_name,
        "day_index": int(day_index),
        "day_type": day_type,
        "persona_description": profile.persona_description,
        "energy_priorities": profile.energy_priorities,
        "current_observations": dict(observations),
    }
    return (
        f"Create a high-level 24-hour energy operating schedule for building "
        f"'{profile.building_name}' on simulation day {day_index} (day_type={day_type}).\n\n"
        f"{format_context_block(context)}\n\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "description_of_today": [\n'
        '    "00:00-06:00: <overnight off-peak EV/battery plan>",\n'
        '    "06:00-10:00: <morning ramp plan>",\n'
        '    "10:00-16:00: <midday solar self-consumption & pre-conditioning plan>",\n'
        '    "16:00-22:00: <evening peak shaving & comfort protection plan>",\n'
        '    "22:00-24:00: <late evening recovery plan>"\n'
        "  ],\n"
        '  "solar_charging_window": "<target hours and SoC goal for PV charging>",\n'
        '  "peak_avoidance_strategy": "<how battery/thermal/loads avoid high tariff imports>",\n'
        '  "ev_charging_plan": "<how EV departure SoC targets are guaranteed>"\n'
        "}"
    )
