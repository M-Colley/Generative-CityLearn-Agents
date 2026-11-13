from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from citylearn.agents.baseline import BusinessAsUsualAgent
from citylearn.citylearn import CityLearnEnv
import pandas as pd

from generative_citylearn_agents.module.action.citylearn_adapter import LLMAgent

DEFAULT_SELECTED_KPIS = (
    "district_cost_total_control_eur",
    "district_energy_grid_total_import_control_kwh",
    "district_energy_grid_peak_import_control_kw",
    "district_emissions_total_control_kgco2",
    "district_ev_performance_departure_min_acceptable_feasible_ratio",
    "district_electrical_service_phase_violations_energy_total_kwh",
)


def _run_controller_episode(
    dataset_or_schema: Union[str, Path, Mapping[str, Any]],
    *,
    controller_factory: Any,
    buildings: Optional[Sequence[str]] = None,
    episode_time_steps: int = 24,
    seed: int = 0,
) -> pd.DataFrame:
    kwargs: dict[str, Any] = {
        "episode_time_steps": int(episode_time_steps),
        "interface": "flat",
        "render_mode": "none",
    }
    if buildings is not None:
        kwargs["buildings"] = list(buildings)

    env = CityLearnEnv(dataset_or_schema, **kwargs)
    try:
        agent = controller_factory(env)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            actions = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(actions)
        return env.evaluate_v2()
    finally:
        env.close()


def build_comparison_table(
    dataset_or_schema: Union[str, Path, Mapping[str, Any]],
    *,
    buildings: Optional[Sequence[str]] = None,
    episode_time_steps: int = 24,
    seed: int = 0,
    llm_kwargs: Optional[Mapping[str, Any]] = None,
    selected_kpis: Sequence[str] = DEFAULT_SELECTED_KPIS,
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Compare ``LLMAgent`` against ``BusinessAsUsualAgent`` on district KPIs (GTA ``eval/`` style)."""
    llm_opts = dict(llm_kwargs or {"backend": "mock"})
    bau_kpis = _run_controller_episode(
        dataset_or_schema,
        controller_factory=lambda env: BusinessAsUsualAgent(env),
        buildings=buildings,
        episode_time_steps=episode_time_steps,
        seed=seed,
    )
    llm_kpis = _run_controller_episode(
        dataset_or_schema,
        controller_factory=lambda env: LLMAgent(env, **llm_opts),
        buildings=buildings,
        episode_time_steps=episode_time_steps,
        seed=seed,
    )

    bau_district = bau_kpis.loc[
        (bau_kpis["level"] == "district") & bau_kpis["cost_function"].isin(selected_kpis),
        ["cost_function", "value"],
    ].rename(columns={"value": "business_as_usual"})
    llm_district = llm_kpis.loc[
        (llm_kpis["level"] == "district") & llm_kpis["cost_function"].isin(selected_kpis),
        ["cost_function", "value"],
    ].rename(columns={"value": "generative_llm_agent"})

    merged = pd.merge(bau_district, llm_district, on="cost_function", how="outer")
    merged["delta_llm_minus_bau"] = merged["generative_llm_agent"] - merged["business_as_usual"]

    if output_csv is not None:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)

    return merged
