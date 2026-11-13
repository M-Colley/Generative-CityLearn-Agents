from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from generative_citylearn_agents.llm.base_chat_api import BaseChatAPI
from generative_citylearn_agents.model.agent import GenerativeBuildingAgentState
from generative_citylearn_agents.model.building import BuildingProfile
from generative_citylearn_agents.model.seed import BuildingSeedProfile
from generative_citylearn_agents.module.profile.prompt.description import (
    get_description_prompt,
)
from generative_citylearn_agents.module.profile.seed.building_seed_generator import (
    BuildingSeedGenerator,
)
from generative_citylearn_agents.util.json import extract_json_from_polluted_string

LOGGER = logging.getLogger(__name__)


class ProfileModule:
    """Stage-1 persona synthesis module modeled directly after GTA's ``ProfileModule``."""

    def __init__(
        self,
        llm_client: BaseChatAPI,
        *,
        seed_generator: Optional[BuildingSeedGenerator] = None,
        max_retries: int = 1,
        preloaded_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        self.llm_client = llm_client
        self.seed_generator = seed_generator or BuildingSeedGenerator()
        self.max_retries = max(int(max_retries), 0)
        self._profiles: Dict[str, BuildingProfile] = {}

        for name, payload in (preloaded_profiles or {}).items():
            try:
                self._profiles[str(name)] = BuildingProfile.from_dict(payload)
            except Exception:
                continue

    @property
    def profiles(self) -> Dict[str, BuildingProfile]:
        return dict(self._profiles)

    def clear(self) -> None:
        self._profiles.clear()

    @staticmethod
    def generate_seeded_agents(
        seed_generator: BuildingSeedGenerator,
        buildings: Sequence[Any],
    ) -> List[GenerativeBuildingAgentState]:
        """Create seeded agent states for a list of CityLearn buildings (GTA style)."""
        return [
            GenerativeBuildingAgentState(
                building_name=str(getattr(b, "name", "Building")),
                seed=seed_generator.generate_seed(b),
            )
            for b in buildings
        ]

    def generate_descriptions_multithreaded(
        self,
        agents: Sequence[GenerativeBuildingAgentState],
        max_workers: int = 4,
    ) -> Tuple[List[GenerativeBuildingAgentState], List[GenerativeBuildingAgentState]]:
        """Synthesize personas concurrently across seeded agents (mirroring GTA)."""
        workers = max(int(max_workers), 1)
        if workers == 1 or len(agents) <= 1:
            profiles = [self.generate_profile_from_seed(a.seed) for a in agents]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                profiles = list(pool.map(lambda a: self.generate_profile_from_seed(a.seed), agents))

        described: List[GenerativeBuildingAgentState] = []
        fallback_only: List[GenerativeBuildingAgentState] = []
        for agent_state, profile in zip(agents, profiles):
            agent_state.profile = profile
            self._profiles[profile.building_name] = profile
            if profile.used_fallback:
                fallback_only.append(agent_state)
            else:
                described.append(agent_state)
        return described, fallback_only

    def get_or_create_profile(self, building: Any) -> BuildingProfile:
        name = str(getattr(building, "name", "Building"))
        if name in self._profiles:
            return self._profiles[name]

        seed = self.seed_generator.generate_seed(building)
        profile = self.generate_profile_from_seed(seed)
        self._profiles[name] = profile
        return profile

    def generate_profile_from_seed(self, seed: BuildingSeedProfile) -> BuildingProfile:
        if seed.building_name in self._profiles:
            return self._profiles[seed.building_name]

        prompt = get_description_prompt(seed)
        raw_text = ""
        for _ in range(self.max_retries + 1):
            try:
                raw_text = self.llm_client.get_completion(prompt)
                parsed = extract_json_from_polluted_string(raw_text)
                if isinstance(parsed, Mapping) and str(parsed.get("persona_description", "")).strip():
                    priorities_raw = parsed.get("energy_priorities") or []
                    if isinstance(priorities_raw, str):
                        priorities = [priorities_raw]
                    elif isinstance(priorities_raw, list):
                        priorities = [str(item) for item in priorities_raw if str(item).strip()]
                    else:
                        priorities = []

                    return BuildingProfile(
                        building_name=seed.building_name,
                        persona_description=str(parsed["persona_description"]).strip(),
                        energy_priorities=priorities
                        or [
                            "Maximize solar self-consumption",
                            "Protect occupant comfort and EV readiness",
                        ],
                        comfort_preference=str(
                            parsed.get(
                                "comfort_preference",
                                "Maintain indoor comfort within setpoint band.",
                            )
                        ),
                        ev_flexibility_preference=str(
                            parsed.get(
                                "ev_flexibility_preference",
                                "Ensure EV departure SoC target before scheduled departure.",
                            )
                        ),
                        seed_profile=seed,
                        raw_response=raw_text,
                        used_fallback=False,
                    )
            except Exception as exc:
                LOGGER.debug("ProfileModule fallback retry for %s: %s", seed.building_name, exc)

        return self._fallback_profile(seed, raw_response=raw_text)

    @staticmethod
    def _fallback_profile(seed: BuildingSeedProfile, raw_response: str = "") -> BuildingProfile:
        return BuildingProfile(
            building_name=seed.building_name,
            persona_description=(
                f"{seed.building_name} is a {seed.occupant_archetype} managing "
                f"{seed.pv_nominal_power_kw:.1f} kWp PV and {seed.battery_capacity_kwh:.1f} kWh storage "
                f"to balance comfort, EV departure readiness, and grid cost."
            ),
            energy_priorities=[
                "Meet EV departure SoC targets reliably",
                "Store solar surplus and low-price electricity",
                "Reduce peak grid imports",
            ],
            comfort_preference="Keep indoor temperature comfortable around setpoints.",
            ev_flexibility_preference="Prioritize timely EV charging above charger minimum power.",
            seed_profile=seed,
            raw_response=raw_response,
            used_fallback=True,
        )
