from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union
from unittest.mock import MagicMock, patch

import citylearn
from citylearn.citylearn import CityLearnEnv
import pytest

from generative_citylearn_agents import (
    ActionModule,
    BuildingSeedGenerator,
    CallableChatAPI,
    GenerativeBuildingAgent,
    HuggingfaceChatAPI,
    LLMAgent,
    MultiCommunityLLMAdapter,
    OpenAICompatibleChatAPI,
    PlanningModule,
    ProfileModule,
    RuleGuidedMockChatAPI,
    Storage,
    build_comparison_table,
    evaluate_qualitative_artifacts,
    extract_json_from,
    extract_json_from_polluted_string,
    run_energy_simulacra,
)


def _resolve_dataset(dataset_name: str) -> Union[str, Path]:
    repo_root = Path(citylearn.__file__).resolve().parents[1]
    candidate = repo_root / "data" / "datasets" / dataset_name / "schema.json"
    if candidate.is_file():
        return candidate
    return dataset_name


class TestJsonExtraction:
    def test_extracts_from_think_tags_and_code_fence(self):
        polluted = (
            "<think>\nLet us check battery SoC (0.42) and price {high: 0.31}.\n</think>\n"
            "Here is my control decision:\n"
            "```json\n"
            '{"decisions": [{"action_name": "electrical_storage", "reasoning": "Discharge {peak}", "value": -0.25}]}\n'
            "```\n"
            "Hope this helps!"
        )
        parsed = extract_json_from_polluted_string(polluted)
        assert parsed["decisions"][0]["action_name"] == "electrical_storage"
        assert parsed["decisions"][0]["value"] == pytest.approx(-0.25)
        assert extract_json_from(polluted) == parsed

    def test_repairs_trailing_commas_and_nested_braces_in_strings(self):
        noisy = (
            'Reasoning summary before JSON: '
            '{"persona_description": "Household {A} with PV & EV", '
            '"energy_priorities": ["PV self-consumption", "EV readiness",],}'
        )
        parsed = extract_json_from(noisy)
        assert parsed["persona_description"] == "Household {A} with PV & EV"
        assert len(parsed["energy_priorities"]) == 2

    def test_raises_value_error_on_non_json_output(self):
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            extract_json_from_polluted_string("No braces or JSON payload anywhere here.")


class TestChatBackends:
    def test_openai_compatible_chat_api_posts_and_parses_completion(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"decisions": [{"action_name": "electrical_storage", "value": 0.2, "reasoning": "Solar"}]}'
                    }
                }
            ]
        }
        client = OpenAICompatibleChatAPI(
            base_url="http://localhost:11434/v1/",
            model_id="qwen2.5:7b-instruct",
            api_key="test-key",
            temperature=0.1,
            n_predict=256,
        )
        with patch(
            "generative_citylearn_agents.llm.openai_compatible_chat_api.requests.post",
            return_value=mock_response,
        ) as post_mock:
            out = client.get_completion("Select action")

        assert '"electrical_storage"' in out
        assert post_mock.call_args[0][0] == "http://localhost:11434/v1/chat/completions"
        assert post_mock.call_args[1]["headers"]["Authorization"] == "Bearer test-key"

    def test_huggingface_chat_api_injected_pipeline_and_pickle_roundtrip(self):
        fake_pipe = MagicMock(
            return_value=[
                {
                    "generated_text": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "prompt"},
                        {"role": "assistant", "content": '{"persona_description": "Solar prosumer"}'},
                    ]
                }
            ]
        )
        hf_client = HuggingfaceChatAPI(
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            n_predict=128,
            temperature=0.0,
            pipeline_instance=fake_pipe,
        )
        text = hf_client.get_completion("Describe building")
        assert extract_json_from(text)["persona_description"] == "Solar prosumer"

        restored = pickle.loads(pickle.dumps(hf_client))
        assert restored.model_id == "Qwen/Qwen3-4B-Instruct-2507"
        assert restored._pipeline is None


class TestProfileAndPlanningModules:
    def test_malformed_llm_outputs_trigger_safe_bounded_fallbacks(self):
        broken_client = CallableChatAPI(lambda prompt: "I refuse to output valid JSON!")
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")
        env = CityLearnEnv(
            dataset,
            buildings=["Building_1"],
            episode_time_steps=4,
            render_mode="none",
        )
        try:
            building = env.buildings[0]
            seed_gen = BuildingSeedGenerator(
                persona_overrides={building.name: {"occupant_archetype": "custom commuter"}}
            )
            prof_mod = ProfileModule(broken_client, seed_generator=seed_gen, max_retries=0)
            profile = prof_mod.get_or_create_profile(building)
            assert profile.used_fallback is True
            assert "custom commuter" in profile.persona_description

            plan_mod = PlanningModule(broken_client, max_retries=0)
            obs = building.observations(include_all=False, normalize=False, periodic_normalization=False)
            day_plan = plan_mod.plan_day_schedule(profile=profile, day_index=0, observations=obs)
            assert day_plan.used_fallback is True
            assert len(day_plan.description_of_today) >= 1

            specs = [
                {"action_name": "electrical_storage", "low": -0.5, "high": 0.5},
                {"action_name": "positive_only_device", "low": 0.2, "high": 0.8},
            ]
            vec, record = plan_mod.select_actions(
                profile=profile,
                day_plan=day_plan,
                time_step=0,
                observations=obs,
                action_specs=specs,
            )
            assert vec == [pytest.approx(0.0), pytest.approx(0.2)]
            assert all(d.used_fallback for d in record.decisions)
        finally:
            env.close()

    def test_multithreaded_persona_and_schedule_generation(self):
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")
        env = CityLearnEnv(
            dataset,
            buildings=["Building_1", "Building_2"],
            episode_time_steps=4,
            render_mode="none",
        )
        try:
            env.reset(seed=0)
            seed_gen = BuildingSeedGenerator()
            seeded_agents = ProfileModule.generate_seeded_agents(seed_gen, env.buildings)
            prof_mod = ProfileModule(RuleGuidedMockChatAPI())
            described, fallbacks = prof_mod.generate_descriptions_multithreaded(
                seeded_agents, max_workers=2
            )
            assert len(described) == 2
            assert len(fallbacks) == 0

            plan_mod = PlanningModule(RuleGuidedMockChatAPI())
            obs_map = {
                b.name: b.observations(include_all=False, normalize=False, periodic_normalization=False)
                for b in env.buildings
            }
            scheduled, sched_fallbacks = plan_mod.generate_day_schedules_multithreaded(
                described, obs_map, day_index=0, max_workers=2
            )
            assert len(scheduled) == 2
            assert len(sched_fallbacks) == 0
        finally:
            env.close()


class TestSimulacraAndEval:
    def test_simulacra_orchestrator_storage_reload_and_eval(self, tmp_path: Path):
        storage_dir = tmp_path / "simulacra_run"
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")

        result = run_energy_simulacra(
            {
                "dataset": str(dataset),
                "buildings": ["Building_1", "Building_2"],
                "episode_time_steps": 6,
                "backend": "mock",
                "workers": 2,
                "storage_path": str(storage_dir),
            }
        )
        assert GenerativeBuildingAgent is LLMAgent
        assert not result["kpis"].empty
        assert (storage_dir / Storage.DESCRIPTIONS_FILENAME).is_file()
        assert (storage_dir / Storage.DAY_SCHEDULES_FILENAME).is_file()
        assert (storage_dir / Storage.DECISIONS_FILENAME).is_file()
        assert (storage_dir / Storage.KPI_SUMMARY_FILENAME).is_file()

        qual = evaluate_qualitative_artifacts(storage_dir)
        assert qual["num_personas"] == 2
        assert qual["num_day_schedules"] == 2
        assert qual["num_step_records"] == 10
        assert qual["fallback_ratio"] == 0.0

        # Verify load_from_storage reuses saved personas without re-calling persona LLM stage
        persona_calls = {"count": 0}
        mock_api = RuleGuidedMockChatAPI()

        def counting_fn(prompt: str, sys_prompt: str) -> str:
            if '"prompt_stage": "persona_description"' in prompt:
                persona_calls["count"] += 1
            return mock_api.get_completion(prompt, system_prompt=sys_prompt)

        env = CityLearnEnv(
            dataset,
            buildings=["Building_1", "Building_2"],
            episode_time_steps=4,
            render_mode="none",
        )
        try:
            agent = LLMAgent(
                env,
                llm_client=counting_fn,
                storage_dir=storage_dir,
                load_from_storage=True,
            )
            obs, _ = env.reset(seed=0)
            while not (env.terminated or env.truncated):
                obs, _, _, _, _ = env.step(agent.predict(obs))
            assert persona_calls["count"] == 0
        finally:
            env.close()

        comp_df = build_comparison_table(
            dataset,
            buildings=["Building_1"],
            episode_time_steps=5,
            output_csv=tmp_path / "comparison.csv",
        )
        assert not comp_df.empty
        assert (tmp_path / "comparison.csv").is_file()

    def test_entity_and_dynamic_topology_and_schema_load_agent(self, tmp_path: Path):
        dyn_path = _resolve_dataset("citylearn_three_phase_dynamic_topology_demo")
        if isinstance(dyn_path, Path) and dyn_path.is_file():
            dyn_schema: Dict[str, Any] = json.loads(dyn_path.read_text(encoding="utf-8"))
            dyn_schema = copy.deepcopy(dyn_schema)
            dyn_schema["root_directory"] = str(dyn_path.parent)
            dyn_schema["topology_mode"] = "dynamic"
            for event, step in zip(dyn_schema.get("topology_events", []), [2, 3, 4, 5, 6, 7]):
                event["time_step"] = step

            env = CityLearnEnv(
                dyn_schema,
                episode_time_steps=8,
                interface="entity",
                render_mode="none",
            )
            try:
                agent = LLMAgent(env, backend="mock", decision_interval_steps=2)
                obs, _ = env.reset(seed=0)
                initial_count = len(agent.storage.descriptions)
                while not (env.terminated or env.truncated):
                    obs, _, _, _, _ = env.step(agent.predict(obs))
                assert len(agent.storage.descriptions) > initial_count
            finally:
                env.close()

        # Verify CityLearn schema.json `"type": "generative_citylearn_agents.LLMAgent"`
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")
        if isinstance(dataset, Path) and dataset.is_file():
            schema_dict = json.loads(dataset.read_text(encoding="utf-8"))
            schema_dict["root_directory"] = str(dataset.parent)
            schema_dict["buildings"] = {
                k: v for idx, (k, v) in enumerate(schema_dict["buildings"].items()) if idx == 0
            }
            schema_dict["agent"] = {
                "type": "generative_citylearn_agents.LLMAgent",
                "attributes": {
                    "backend": "mock",
                    "storage_dir": str(tmp_path / "schema_loaded"),
                },
            }
            env2 = CityLearnEnv(schema_dict, episode_time_steps=4, render_mode="none")
            try:
                loaded_agent = env2.load_agent()
                assert isinstance(loaded_agent, LLMAgent)
                loaded_agent.learn(episodes=1)
                assert (tmp_path / "schema_loaded" / "3_decisions.json").is_file()
            finally:
                env2.close()


class TestLLMBlockerMitigations:
    def test_sentinel_decoding_strips_faults_and_negative_ev_sentinels(self):
        raw = {
            "hour": 14.0,
            "indoor_dry_bulb_temperature": -9999.0,
            "outdoor_dry_bulb_temperature": 26.5,
            "electric_vehicle_charger_c1_connected_state": 0.0,
            "electric_vehicle_charger_c1_soc": -0.1,
            "electric_vehicle_charger_c1_estimated_departure_time": -1.0,
        }
        cleaned, diag = ActionModule.sanitize_telemetry_for_prompt(raw)
        assert "indoor_dry_bulb_temperature" not in cleaned
        assert "electric_vehicle_charger_c1_soc" not in cleaned
        assert cleaned["electric_vehicle_charger_c1_connected_state"] == 0.0
        assert diag["faulted_sensors"] == ["indoor_dry_bulb_temperature"]
        assert diag["disconnected_ev_chargers"] == ["c1"]

    def test_parallel_step_prediction_execution_feedback_and_pickle(self):
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")
        env = CityLearnEnv(
            dataset,
            buildings=["Building_1", "Building_2", "Building_3"],
            episode_time_steps=4,
            render_mode="none",
        )
        seen_prompts = []
        mock_api = RuleGuidedMockChatAPI()

        def spy_fn(prompt: str, sys_prompt: str) -> str:
            seen_prompts.append(prompt)
            return mock_api.get_completion(prompt, system_prompt=sys_prompt)

        try:
            agent = LLMAgent(env, llm_client=spy_fn, workers=3)
            assert agent.comparison_identity()["observation_access"] == "agent_payload"
            obs, _ = env.reset(seed=0)
            first_actions = agent.predict(obs)
            assert len(first_actions) == 3

            agent.observe_transition(
                info={
                    "entity_action_execution": {
                        "charger": {
                            "Building_1/c1": {
                                "requested_power_kw": 7.2,
                                "limited_power_kw": 3.6,
                                "clip_reasons": ["phase_headroom"],
                            }
                        }
                    }
                }
            )
            obs, _, _, _, _ = env.step(first_actions)
            second_actions = agent.predict(obs)
            assert len(second_actions) == 3
            assert any("execution_feedback" in p and "phase_headroom" in p for p in seen_prompts)

            # Verify pickle safety even with non-pickleable local closure spy_fn
            unpickled = pickle.loads(pickle.dumps(agent))
            assert isinstance(unpickled, LLMAgent)
        finally:
            env.close()

    def test_multi_community_llm_adapter_rollout_and_portfolio_kpis(self, tmp_path: Path):
        dataset = _resolve_dataset("citylearn_challenge_2022_phase_all_plus_evs")
        communities = [
            {
                "id": "north_comm",
                "schema": dataset,
                "env_kwargs": {
                    "buildings": ["Building_1"],
                    "episode_time_steps": 4,
                    "render_mode": "none",
                },
            },
            {
                "id": "south_comm",
                "schema": dataset,
                "env_kwargs": {
                    "buildings": ["Building_2"],
                    "episode_time_steps": 4,
                    "render_mode": "none",
                },
            },
        ]
        mc = MultiCommunityLLMAdapter(
            communities,
            backend="mock",
            workers=2,
            storage_root=tmp_path / "mc_logs",
        )
        try:
            obs_map, _ = mc.reset(seed=0)
            assert mc.community_ids == ["north_comm", "south_comm"]
            done = False
            while not done:
                act_map = mc.predict(obs_map)
                assert set(act_map.keys()) == {"north_comm", "south_comm"}
                obs_map, _, term, trunc, _ = mc.step(act_map)
                done = bool(term or trunc)
            kpis = mc.evaluate_kpis()
            assert not kpis.empty
            assert "__portfolio__" in set(kpis["community_id"])
        finally:
            mc.close()

