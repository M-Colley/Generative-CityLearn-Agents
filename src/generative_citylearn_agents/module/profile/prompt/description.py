from __future__ import annotations

from generative_citylearn_agents.llm.base_chat_api import format_context_block
from generative_citylearn_agents.model.seed import BuildingSeedProfile


def get_description_prompt(seed_profile: BuildingSeedProfile) -> str:
    """Build Stage-1 persona prompt modeled after GTA's ``module/profile/prompt/description.py``."""
    context = {
        "prompt_stage": "persona_description",
        "seed_profile": seed_profile.to_dict(),
    }
    return (
        f"Generate a realistic, physically grounded energy-management persona for "
        f"building '{seed_profile.building_name}' based on its installed equipment, "
        f"annual load profile, and occupant archetype.\n\n"
        f"{format_context_block(context)}\n\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "persona_description": "<2-4 sentences describing household/facility occupancy, comfort habits, and DER flexibility>",\n'
        '  "energy_priorities": ["<priority 1>", "<priority 2>", "<priority 3>"],\n'
        '  "comfort_preference": "<concise thermal/appliance comfort guideline>",\n'
        '  "ev_flexibility_preference": "<concise EV departure readiness and charging strategy>"\n'
        "}"
    )
