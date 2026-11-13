from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from generative_citylearn_agents.model.seed import BuildingSeedProfile


@dataclass
class BuildingProfile:
    """Grounded natural-language persona and operating priorities for a building."""

    building_name: str
    persona_description: str
    energy_priorities: List[str]
    comfort_preference: str
    ev_flexibility_preference: str
    seed_profile: BuildingSeedProfile
    raw_response: str = ""
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_name": self.building_name,
            "persona_description": self.persona_description,
            "energy_priorities": list(self.energy_priorities),
            "comfort_preference": self.comfort_preference,
            "ev_flexibility_preference": self.ev_flexibility_preference,
            "seed_profile": self.seed_profile.to_dict(),
            "used_fallback": bool(self.used_fallback),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BuildingProfile":
        return cls(
            building_name=str(data.get("building_name", "Building")),
            persona_description=str(data.get("persona_description", "")),
            energy_priorities=[str(x) for x in (data.get("energy_priorities") or [])],
            comfort_preference=str(data.get("comfort_preference", "")),
            ev_flexibility_preference=str(data.get("ev_flexibility_preference", "")),
            seed_profile=BuildingSeedProfile.from_dict(data.get("seed_profile") or {}),
            raw_response=str(data.get("raw_response", "")),
            used_fallback=bool(data.get("used_fallback", False)),
        )
