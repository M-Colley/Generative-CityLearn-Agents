from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv

from generative_citylearn_agents.llm import BaseChatAPI, resolve_llm_client
from generative_citylearn_agents.model.agent import StepDecisionRecord
from generative_citylearn_agents.model.building import BuildingProfile
from generative_citylearn_agents.model.day_schedule import DayEnergyPlan
from generative_citylearn_agents.module.action.action_module import ActionModule
from generative_citylearn_agents.module.planning.planning_module import PlanningModule
from generative_citylearn_agents.module.profile.profile_module import ProfileModule
from generative_citylearn_agents.module.profile.seed.building_seed_generator import (
    BuildingSeedGenerator,
)
from generative_citylearn_agents.util.storage import Storage


class CityLearnAdapter:
    """Environment lifecycle wrapper analogous to ``SumoAdapter`` in GTA."""

    def __init__(
        self,
        dataset_or_schema: Union[str, Path, Mapping[str, Any]],
        *,
        buildings: Optional[Sequence[str]] = None,
        episode_time_steps: Optional[int] = None,
        interface: str = "flat",
        central_agent: bool = False,
        render_mode: str = "none",
        **env_kwargs: Any,
    ):
        kwargs: Dict[str, Any] = {
            "interface": interface,
            "central_agent": central_agent,
            "render_mode": render_mode,
            **env_kwargs,
        }
        if buildings is not None:
            kwargs["buildings"] = list(buildings)
        if episode_time_steps is not None:
            kwargs["episode_time_steps"] = int(episode_time_steps)
        self.env = CityLearnEnv(dataset_or_schema, **kwargs)

    @property
    def buildings(self) -> Sequence[Any]:
        return self.env.unwrapped.buildings

    def reset(self, seed: int = 0):
        return self.env.reset(seed=seed)

    def step(self, actions: List[List[float]]):
        return self.env.step(actions)

    def evaluate_kpis(self):
        return self.env.evaluate_v2()

    def close(self) -> None:
        self.env.close()


class LLMAgent(Agent):
    """First-class CityLearn ``Agent`` driven by ``ProfileModule`` + ``PlanningModule`` + ``ActionModule``.

    Can be instantiated programmatically in Python OR referenced directly in a
    CityLearn ``schema.json`` via ``"type": "generative_citylearn_agents.LLMAgent"``.
    """

    def __init__(
        self,
        env: CityLearnEnv,
        llm_client: Optional[Union[BaseChatAPI, Any]] = None,
        backend: str = "mock",
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        n_predict: int = 700,
        gpu_id: int = 0,
        storage_dir: Optional[Union[str, Path]] = None,
        load_from_storage: bool = False,
        enable_day_planning: bool = True,
        decision_interval_steps: int = 1,
        persona_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        include_all_observations: bool = False,
        max_retries: int = 1,
        workers: int = 1,
        auto_flush_storage: bool = True,
        **kwargs: Any,
    ):
        self.backend = str(backend or "mock").strip().lower()
        self.model_id = model_id
        self.base_url = base_url
        self.temperature = float(temperature)
        self.n_predict = int(n_predict)
        self.gpu_id = int(gpu_id)
        self.enable_day_planning = bool(enable_day_planning)
        self.decision_interval_steps = max(int(decision_interval_steps), 1)
        self.include_all_observations = bool(include_all_observations)
        self.max_retries = max(int(max_retries), 0)
        self.workers = max(int(workers), 1)

        self.llm_client: BaseChatAPI = resolve_llm_client(
            llm_client=llm_client,
            backend=self.backend,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=self.temperature,
            n_predict=self.n_predict,
            gpu_id=self.gpu_id,
        )
        self.storage = Storage(
            storage_dir=storage_dir,
            load_from_storage=load_from_storage,
            auto_flush=auto_flush_storage,
        )
        self.seed_generator = BuildingSeedGenerator(persona_overrides=persona_overrides)
        self.profile_module = ProfileModule(
            self.llm_client,
            seed_generator=self.seed_generator,
            max_retries=self.max_retries,
            preloaded_profiles=self.storage.descriptions if load_from_storage else None,
        )
        self.planning_module = PlanningModule(
            self.llm_client,
            max_retries=self.max_retries,
        )
        self.action_module = ActionModule(include_all_observations=self.include_all_observations)

        self._day_plans_by_building: Dict[str, DayEnergyPlan] = {}
        self._last_day_key_by_building: Dict[str, int] = {}
        self._decision_history_by_building: Dict[str, List[StepDecisionRecord]] = {}
        self._last_step_record_by_building: Dict[str, StepDecisionRecord] = {}
        self._llm_topology_version: int = int(getattr(env.unwrapped, "topology_version", 0))

        super().__init__(env, **kwargs)
        self.observation_space = self.env.unwrapped.flat_observation_space
        self.action_space = self.env.unwrapped.flat_action_space
        self.reset()

    def reset(self):
        super().reset()
        if not hasattr(self, "_day_plans_by_building"):
            return

        self._day_plans_by_building.clear()
        self._last_day_key_by_building.clear()
        self._decision_history_by_building.clear()
        self._last_step_record_by_building.clear()
        self.action_module.clear_cache()
        self.storage.reset_episode_logs(keep_descriptions=True)

        env = self.env.unwrapped
        self._llm_topology_version = int(getattr(env, "topology_version", 0))
        if self.workers > 1 and len(env.buildings) > 1:
            missing = [b for b in env.buildings if b.name not in self.profile_module.profiles]
            if missing:
                seeded = ProfileModule.generate_seeded_agents(self.seed_generator, missing)
                described, fallbacks = self.profile_module.generate_descriptions_multithreaded(
                    seeded, max_workers=self.workers
                )
                for st in [*described, *fallbacks]:
                    if st.profile is not None:
                        self.storage.record_profile(st.profile)
        for building in env.buildings:
            self._ensure_building_profile(building)

    def comparison_identity(self) -> Dict[str, Any]:
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "version": "0.1.0",
            "parameters": {
                "backend": self.backend,
                "client_class": type(self.llm_client).__qualname__,
                "model_id": getattr(self.llm_client, "model_id", self.model_id),
                "temperature": self.temperature,
                "n_predict": self.n_predict,
                "enable_day_planning": self.enable_day_planning,
                "decision_interval_steps": self.decision_interval_steps,
                "include_all_observations": self.include_all_observations,
            },
            "observation_access": (
                "all_building_observations" if self.include_all_observations else "active_only"
            ),
        }

    def predict(
        self,
        observations: Any,
        deterministic: Optional[bool] = None,
    ) -> List[List[float]]:
        env = self.env.unwrapped
        if env.time_step == 0 and self.time_step != 0:
            self.reset()

        if getattr(env, "topology_mode", "static") == "dynamic":
            version = int(getattr(env, "topology_version", 0))
            if version != self._llm_topology_version:
                self.action_space = env.flat_action_space
                self.action_names = env.action_names
                self.observation_space = env.flat_observation_space
                self.observation_names = env.observation_names
                self.building_metadata = env.get_metadata()["buildings"]
                self.action_module.clear_cache()
                self._llm_topology_version = version

        steps_per_day = max(1, int(round(86400.0 / max(float(env.seconds_per_time_step), 1.0))))
        day_index = int(env.time_step // steps_per_day)

        per_building_actions = [
            self._predict_building_actions(
                building=building,
                time_step=int(env.time_step),
                day_index=day_index,
            )
            for building in env.buildings
        ]
        actions = self.action_module.format_env_actions(
            per_building_actions,
            central_agent=bool(env.central_agent),
        )
        self.actions = actions
        self.next_time_step()
        return actions

    def _ensure_building_profile(self, building: Any) -> BuildingProfile:
        existed = building.name in self.profile_module.profiles
        profile = self.profile_module.get_or_create_profile(building)
        if not existed or building.name not in self.storage.descriptions:
            self.storage.record_profile(profile)
        return profile

    def _predict_building_actions(
        self,
        *,
        building: Any,
        time_step: int,
        day_index: int,
    ) -> List[float]:
        profile = self._ensure_building_profile(building)
        obs_dict = self.action_module.building_observations_dict(building)
        action_specs = self.action_module.building_action_specs(building)

        day_plan: Optional[DayEnergyPlan] = None
        if self.enable_day_planning:
            if self._last_day_key_by_building.get(building.name) != day_index:
                day_plan = self.planning_module.plan_day_schedule(
                    profile=profile,
                    day_index=day_index,
                    observations=obs_dict,
                )
                self._day_plans_by_building[building.name] = day_plan
                self._last_day_key_by_building[building.name] = day_index
                self.storage.record_day_plan(day_plan)
            else:
                day_plan = self._day_plans_by_building.get(building.name)

        prev_record = self._last_step_record_by_building.get(building.name)
        should_reuse = (
            self.decision_interval_steps > 1
            and (time_step % self.decision_interval_steps) != 0
            and prev_record is not None
            and len(prev_record.action_vector) == len(action_specs)
        )

        if should_reuse:
            reclamped = [
                max(float(spec["low"]), min(float(spec["high"]), float(val)))
                for val, spec in zip(prev_record.action_vector, action_specs)
            ]
            record = StepDecisionRecord.clone_for_reused_step(
                prev_record,
                time_step=time_step,
                observations=obs_dict,
                clamped_vector=reclamped,
            )
            vector = reclamped
        else:
            recent = self._decision_history_by_building.get(building.name, [])
            vector, record = self.planning_module.select_actions(
                profile=profile,
                day_plan=day_plan,
                time_step=time_step,
                observations=obs_dict,
                action_specs=action_specs,
                recent_decisions=recent,
            )

        history = self._decision_history_by_building.setdefault(building.name, [])
        history.append(record)
        if len(history) > 8:
            del history[:-8]
        self._last_step_record_by_building[building.name] = record
        self.storage.record_step_decision(record)
        return vector


GenerativeBuildingAgent = LLMAgent
