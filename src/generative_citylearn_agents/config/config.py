from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

config_fast_demo: Dict[str, Any] = {
    "dataset": "citylearn_challenge_2022_phase_all_plus_evs",
    "buildings": ["Building_1", "Building_2"],
    "episode_time_steps": 24,
    "interface": "flat",
    "central_agent": False,
    "backend": "mock",
    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
    "base_url": None,
    "temperature": 0.2,
    "n_predict": 700,
    "gpu_id": 0,
    "workers": 4,
    "enable_day_planning": True,
    "decision_interval_steps": 1,
    "load_from_storage": False,
    "storage_path": "storage/fast_demo",
    "seed": 0,
}

config_phase_all_evs: Dict[str, Any] = {
    **config_fast_demo,
    "buildings": None,
    "episode_time_steps": 168,
    "backend": "huggingface",
    "storage_path": "storage/phase_all_evs",
}

config_ollama_local: Dict[str, Any] = {
    **config_fast_demo,
    "backend": "openai_compatible",
    "base_url": "http://localhost:11434/v1",
    "model_id": "qwen2.5:7b-instruct",
    "storage_path": "storage/ollama_local",
}


def get_default_config(preset: str = "fast_demo") -> Dict[str, Any]:
    presets = {
        "fast_demo": config_fast_demo,
        "phase_all_evs": config_phase_all_evs,
        "ollama_local": config_ollama_local,
    }
    if preset not in presets:
        raise ValueError(f"Unknown config preset {preset!r}. Available: {sorted(presets)}")
    return deepcopy(presets[preset])
