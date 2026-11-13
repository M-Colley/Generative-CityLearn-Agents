from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from citylearn.multi_community import MultiCommunityEnv

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

    def step(self, actions: Any, *, agent: Optional["LLMAgent"] = None):
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        if agent is not None:
            agent.observe_transition(info=info, rewards=rewards)
        return obs, rewards, terminated, truncated, info

    def evaluate_kpis(self):
        return self.env.evaluate_v2()

    def close(self) -> None:
        self.env.close()


class LLMAgent(Agent):
    """First-class CityLearn ``Agent`` driven by ``ProfileModule`` + ``PlanningModule`` + ``ActionModule``.

    Supports both ``interface='flat'`` and ``interface='entity'`` (including dynamic
    prosumer topology, demand response, and step-level actuator clipping feedback),
    concurrent multi-building LLM inference via ``workers > 1``, and pickle-safe
    serialization for ``citylearn.__main__.Simulator``.
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
        self._state_lock = threading.RLock()

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
        self._last_execution_feedback_by_building: Dict[str, Dict[str, Any]] = {}
        self._llm_topology_version: int = int(getattr(env.unwrapped, "topology_version", 0))

        super().__init__(env, **kwargs)
        self.observation_space = self.env.unwrapped.flat_observation_space
        self.action_space = self.env.unwrapped.flat_action_space
        self.reset()

    def reset(self):
        super().reset()
        if not hasattr(self, "_day_plans_by_building"):
            return

        with self._state_lock:
            self._day_plans_by_building.clear()
            self._last_day_key_by_building.clear()
            self._decision_history_by_building.clear()
            self._last_step_record_by_building.clear()
            self._last_execution_feedback_by_building.clear()
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
                "privileged_simulator_state" if self.include_all_observations else "agent_payload"
            ),
        }

    def observe_transition(
        self,
        *,
        info: Optional[Mapping[str, Any]] = None,
        rewards: Optional[Any] = None,
    ) -> None:
        """Ingest step-level execution diagnostics (actuator clipping, topology events) from ``env.step``."""
        if not isinstance(info, Mapping):
            return

        feedback_by_b: Dict[str, Dict[str, Any]] = {}
        exec_audit = info.get("entity_action_execution")
        if isinstance(exec_audit, Mapping):
            for entity_group, entries in exec_audit.items():
                if not isinstance(entries, Mapping):
                    continue
                for entity_id, audit in entries.items():
                    if not isinstance(audit, Mapping):
                        continue
                    b_name = str(entity_id).split("/", 1)[0]
                    reasons = audit.get("clip_reasons") or [
                        k.replace("clip_reason_", "")
                        for k, v in audit.items()
                        if str(k).startswith("clip_reason_") and bool(v)
                    ]
                    req_p = audit.get("requested_power_kw")
                    lim_p = audit.get("limited_power_kw")
                    was_clipped = bool(reasons) or (
                        isinstance(req_p, (int, float))
                        and isinstance(lim_p, (int, float))
                        and abs(float(req_p) - float(lim_p)) > 1e-4
                    )
                    if was_clipped:
                        feedback_by_b.setdefault(b_name, {}).setdefault("clipped_entities", {})[
                            f"{entity_group}:{entity_id}"
                        ] = {
                            "requested_power_kw": req_p,
                            "limited_power_kw": lim_p,
                            "clip_reasons": list(reasons),
                        }

        topo_events = info.get("topology_events_applied")
        if isinstance(topo_events, Sequence) and len(topo_events) > 0:
            for ev_item in topo_events:
                if isinstance(ev_item, Mapping):
                    b_target = str(ev_item.get("building") or ev_item.get("target_building") or "")
                    if b_target:
                        feedback_by_b.setdefault(b_target, {}).setdefault("topology_events", []).append(
                            dict(ev_item)
                        )

        with self._state_lock:
            self._last_execution_feedback_by_building = feedback_by_b

    def learn(
        self,
        episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        deterministic_finish: Optional[bool] = None,
        logging_level: Optional[int] = None,
    ):
        """Run rollout episodes while preserving ``step()`` ``info`` feedback for the LLM."""
        total_episodes = 1 if episodes is None else max(int(episodes), 1)
        det_finish = False if deterministic_finish is None else bool(deterministic_finish)

        for ep in range(total_episodes):
            det = bool(deterministic) or (det_finish and ep == total_episodes - 1)
            observations, _ = self.env.reset()
            self.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                actions = self.predict(observations, deterministic=det)
                observations, rewards, terminated, truncated, info = self.env.step(actions)
                self.observe_transition(info=info, rewards=rewards)

    def predict(
        self,
        observations: Any,
        deterministic: Optional[bool] = None,
    ) -> Any:
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
        current_step = int(env.time_step)
        day_index = int(current_step // steps_per_day)
        buildings = list(env.buildings)

        obs_by_b, diag_by_b = self.action_module.extract_env_observations_by_building(env, observations)
        specs_by_building: List[List[Dict[str, Any]]] = []
        profiles_by_b: Dict[str, BuildingProfile] = {}
        for building in buildings:
            profiles_by_b[building.name] = self._ensure_building_profile(building)
            specs_by_building.append(self.action_module.building_action_specs(building))
            exec_fb = self._last_execution_feedback_by_building.get(building.name)
            if exec_fb:
                diag_by_b.setdefault(building.name, {})["execution_feedback"] = dict(exec_fb)

        def _solve_index(idx: int) -> Tuple[List[float], Optional[DayEnergyPlan], StepDecisionRecord]:
            b = buildings[idx]
            return self._compute_building_decision(
                profile=profiles_by_b[b.name],
                obs_dict=obs_by_b.get(b.name, {}),
                telemetry_diagnostics=diag_by_b.get(b.name),
                action_specs=specs_by_building[idx],
                time_step=current_step,
                day_index=day_index,
            )

        if self.workers > 1 and len(buildings) > 1:
            pool_size = min(self.workers, len(buildings))
            with ThreadPoolExecutor(max_workers=pool_size) as pool:
                results = list(pool.map(_solve_index, range(len(buildings))))
        else:
            results = [_solve_index(i) for i in range(len(buildings))]

        per_building_actions: List[List[float]] = []
        with self._state_lock:
            for b, (vec, new_plan, record) in zip(buildings, results):
                if new_plan is not None:
                    self._day_plans_by_building[b.name] = new_plan
                    self._last_day_key_by_building[b.name] = day_index
                    self.storage.record_day_plan(new_plan)
                history = self._decision_history_by_building.setdefault(b.name, [])
                history.append(record)
                if len(history) > 8:
                    del history[:-8]
                self._last_step_record_by_building[b.name] = record
                self.storage.record_step_decision(record)
                per_building_actions.append(vec)

        flat_actions = self.action_module.format_env_actions(
            per_building_actions,
            central_agent=bool(env.central_agent),
        )
        self.actions = flat_actions
        self.next_time_step()

        if getattr(env, "interface", "flat") == "entity":
            return self.action_module.format_entity_actions(
                env,
                per_building_actions,
                specs_by_building,
            )
        return flat_actions

    def _ensure_building_profile(self, building: Any) -> BuildingProfile:
        with self._state_lock:
            existed = building.name in self.profile_module.profiles
        profile = self.profile_module.get_or_create_profile(building)
        with self._state_lock:
            if not existed or building.name not in self.storage.descriptions:
                self.storage.record_profile(profile)
        return profile

    def _compute_building_decision(
        self,
        *,
        profile: BuildingProfile,
        obs_dict: Mapping[str, float],
        telemetry_diagnostics: Optional[Mapping[str, Any]],
        action_specs: Sequence[Mapping[str, Any]],
        time_step: int,
        day_index: int,
    ) -> Tuple[List[float], Optional[DayEnergyPlan], StepDecisionRecord]:
        b_name = profile.building_name
        with self._state_lock:
            cached_day = self._last_day_key_by_building.get(b_name)
            existing_plan = self._day_plans_by_building.get(b_name)
            prev_record = self._last_step_record_by_building.get(b_name)
            recent = list(self._decision_history_by_building.get(b_name, []))

        new_plan: Optional[DayEnergyPlan] = None
        day_plan: Optional[DayEnergyPlan] = None
        if self.enable_day_planning:
            if cached_day != day_index:
                new_plan = self.planning_module.plan_day_schedule(
                    profile=profile,
                    day_index=day_index,
                    observations=obs_dict,
                )
                day_plan = new_plan
            else:
                day_plan = existing_plan

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
            return reclamped, new_plan, record

        vector, record = self.planning_module.select_actions(
            profile=profile,
            day_plan=day_plan,
            time_step=time_step,
            observations=obs_dict,
            action_specs=action_specs,
            recent_decisions=recent,
            telemetry_diagnostics=telemetry_diagnostics,
        )
        return vector, new_plan, record

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_state_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._state_lock = threading.RLock()


class MultiCommunityLLMAdapter:
    """Generative LLM orchestrator for ``citylearn.multi_community.MultiCommunityEnv``."""

    def __init__(
        self,
        env_or_communities: Union[MultiCommunityEnv, Sequence[Mapping[str, Any]]],
        *,
        workers: int = 1,
        storage_root: Optional[Union[str, Path]] = None,
        **agent_kwargs: Any,
    ):
        if isinstance(env_or_communities, MultiCommunityEnv):
            self.env = env_or_communities
            self._owns_env = False
        else:
            normalized_comms = []
            for entry in env_or_communities:
                item = dict(entry)
                if "community_id" not in item and "id" in item:
                    item["community_id"] = item.pop("id")
                normalized_comms.append(item)
            self.env = MultiCommunityEnv(communities=normalized_comms)
            self._owns_env = True

        self.workers = max(int(workers), 1)
        self.agents: Dict[str, LLMAgent] = {}
        root_path = Path(storage_root) if storage_root is not None else None
        for cid, child_env in self.env.envs.items():
            comm_storage = (root_path / cid) if root_path is not None else None
            self.agents[cid] = LLMAgent(
                child_env,
                storage_dir=comm_storage,
                workers=self.workers,
                **agent_kwargs,
            )

    @property
    def community_ids(self) -> List[str]:
        return list(self.env.community_ids)

    def reset(self, seed: int = 0):
        obs_by_comm, info_by_comm = self.env.reset(seed=seed)
        for cid, agent in self.agents.items():
            agent.reset()
        return obs_by_comm, info_by_comm

    def predict(
        self,
        observations_by_community: Mapping[str, Any],
        deterministic: Optional[bool] = None,
    ) -> Dict[str, Any]:
        cids = self.community_ids

        def _predict_comm(cid: str) -> Any:
            return self.agents[cid].predict(
                observations_by_community.get(cid),
                deterministic=deterministic,
            )

        if self.workers > 1 and len(cids) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(cids))) as pool:
                preds = list(pool.map(_predict_comm, cids))
        else:
            preds = [_predict_comm(cid) for cid in cids]
        return dict(zip(cids, preds))

    def step(self, actions_by_community: Mapping[str, Any]):
        obs, rewards, terminated, truncated, info = self.env.step(actions_by_community)
        if isinstance(info, Mapping):
            for cid, agent in self.agents.items():
                comm_info = info.get(cid)
                comm_rew = rewards.get(cid) if isinstance(rewards, Mapping) else None
                if isinstance(comm_info, Mapping):
                    agent.observe_transition(info=comm_info, rewards=comm_rew)
        return obs, rewards, terminated, truncated, info

    def evaluate_kpis(self):
        return self.env.evaluate_v2()

    def close(self) -> None:
        for agent in self.agents.values():
            agent.storage.flush()
        if self._owns_env:
            self.env.close()


GenerativeBuildingAgent = LLMAgent
