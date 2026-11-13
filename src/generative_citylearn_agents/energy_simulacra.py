"""End-to-end generative CityLearn simulation orchestrator modeled after GTA's ``traffic_simulacra.py``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from generative_citylearn_agents.config.config import get_default_config
from generative_citylearn_agents.module.action.citylearn_adapter import (
    CityLearnAdapter,
    LLMAgent,
)
from generative_citylearn_agents.util.logging import log_info
from generative_citylearn_agents.util.time import Timer


def run_energy_simulacra(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Execute the full three-stage generative agent workflow on a CityLearn scenario."""
    cfg: Dict[str, Any] = get_default_config("fast_demo")
    if config is not None:
        cfg.update(dict(config))

    log_info("Starting CityLearn generative energy simulacra")
    timer = Timer()
    timer.start()

    log_info("Initialising CityLearn adapter and generative LLM controller...")
    adapter = CityLearnAdapter(
        cfg["dataset"],
        buildings=cfg.get("buildings"),
        episode_time_steps=cfg.get("episode_time_steps", 24),
        interface=cfg.get("interface", "flat"),
        central_agent=bool(cfg.get("central_agent", False)),
        render_mode="none",
    )

    try:
        agent = LLMAgent(
            adapter.env,
            backend=cfg.get("backend", "mock"),
            model_id=cfg.get("model_id"),
            base_url=cfg.get("base_url"),
            api_key=cfg.get("api_key"),
            temperature=float(cfg.get("temperature", 0.2)),
            n_predict=int(cfg.get("n_predict", 700)),
            gpu_id=int(cfg.get("gpu_id", 0)),
            storage_dir=cfg.get("storage_path"),
            load_from_storage=bool(cfg.get("load_from_storage", False)),
            enable_day_planning=bool(cfg.get("enable_day_planning", True)),
            decision_interval_steps=int(cfg.get("decision_interval_steps", 1)),
            workers=int(cfg.get("workers", 1)),
        )

        observations, _ = adapter.reset(seed=int(cfg.get("seed", 0)))
        log_info(
            f"[DESCRIPTION] {len(agent.storage.descriptions)} building personas active "
            f"({len(agent.storage.failed_descriptions)} fallback)."
        )

        terminated = truncated = False
        while not (terminated or truncated):
            actions = agent.predict(observations)
            observations, _, terminated, truncated, _ = adapter.step(actions)

        log_info(
            f"[DAY_SCHEDULE] {len(agent.storage.day_schedules)} daily schedules generated "
            f"({len(agent.storage.failed_day_schedules)} fallback)."
        )
        log_info(f"[DECISIONS] {len(agent.storage.decisions)} step decision records logged.")

        kpis_df = adapter.evaluate_kpis()
        district_rows = kpis_df.loc[
            kpis_df["level"] == "district", ["cost_function", "value"]
        ].to_dict(orient="records")
        agent.storage.record_kpi_rows(district_rows)
        agent.storage.flush()

        elapsed = timer.stop()
        log_info(f"[TIME] Total runtime: {elapsed}.")
        return {
            "runtime": elapsed,
            "summary": agent.storage.summarize_decisions(),
            "kpis": kpis_df,
            "storage_dir": None if agent.storage.storage_dir is None else str(agent.storage.storage_dir),
        }
    finally:
        adapter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["fast_demo", "phase_all_evs", "ollama_local"],
        default="fast_demo",
        help="Configuration preset from config/config.py.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name or path to schema.json.")
    parser.add_argument(
        "--backend",
        choices=["mock", "huggingface", "openai_compatible"],
        default=None,
        help="Override LLM backend.",
    )
    parser.add_argument("--model-id", default=None, help="Model identifier.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint URL.")
    parser.add_argument("--steps", type=int, default=None, help="Episode time steps.")
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=None,
        help="Query LLM every N control steps.",
    )
    parser.add_argument(
        "--buildings",
        nargs="*",
        default=None,
        help="Optional subset of building names.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=None,
        help="Directory for 1_descriptions.json, 2_day_schedules.json, 3_decisions.json, and 4_kpi_summary.json.",
    )
    parser.add_argument(
        "--load-from-storage",
        action="store_true",
        help="Reuse existing personas from storage_path if present.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Concurrent LLM worker threads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_default_config(args.preset)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.backend is not None:
        cfg["backend"] = args.backend
    if args.model_id is not None:
        cfg["model_id"] = args.model_id
    if args.base_url is not None:
        cfg["base_url"] = args.base_url
    if args.steps is not None:
        cfg["episode_time_steps"] = args.steps
    if args.decision_interval is not None:
        cfg["decision_interval_steps"] = args.decision_interval
    if args.buildings is not None:
        cfg["buildings"] = args.buildings if len(args.buildings) > 0 else None
    if args.storage_path is not None:
        cfg["storage_path"] = str(args.storage_path)
    if args.load_from_storage:
        cfg["load_from_storage"] = True
    if args.workers is not None:
        cfg["workers"] = args.workers

    result = run_energy_simulacra(cfg)
    kpis_df = result["kpis"]
    selected = [
        "district_cost_total_control_eur",
        "district_energy_grid_total_import_control_kwh",
        "district_ev_performance_departure_min_acceptable_feasible_ratio",
        "district_electrical_service_phase_violations_energy_total_kwh",
    ]
    print("\n=== District KPI Summary ===")
    print(
        kpis_df.loc[
            (kpis_df["level"] == "district") & kpis_df["cost_function"].isin(selected),
            ["cost_function", "value"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
