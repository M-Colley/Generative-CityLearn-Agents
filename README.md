# Generative CityLearn Agents

Generative LLM-driven building and energy-community controllers for [CityLearn](https://github.com/citylearn-project/CityLearn), modeled directly after the modular cognitive architecture of [GTA: Generative Traffic Agents (`ciao-group/Generative-Traffic-Agents`)](https://github.com/ciao-group/Generative-Traffic-Agents).

## Overview

Just as **Generative Traffic Agents (GTA)** decouples persona synthesis (`ProfileModule`), daily activity scheduling (`PlanningModule`), and simulator execution (`SumoAdapter` + JSON checkpoint artifacts), **Generative CityLearn Agents** structures LLM control of renewable energy communities into three inspectable stages:

1. **Stage 1 — Grounded Persona Synthesis (`module/profile/`)**
   - `BuildingSeedGenerator` inspects static building metadata (rooftop PV kWp, battery kWh/kW & reserve SoC, thermal storage tanks, EV chargers & minimum power deadbands, deferrable appliances, and annual demand estimates) without accessing future episode trajectories.
   - `ProfileModule` prompts the LLM (supporting multithreaded batch generation via `generate_descriptions_multithreaded`) and saves grounded prosumer personas to `1_descriptions.json` (and fallbacks to `1_no_description.json`).
2. **Stage 2 — Daily Energy Schedule Generation (`module/planning/`)**
   - `PlanningModule` synthesizes a 24-hour energy operating plan (`description_of_today`, solar charging window, peak tariff avoidance strategy, and EV departure readiness plan) at episode start and each day rollover, persisted to `2_day_schedules.json`.
3. **Stage 3 — Closed-Loop Step Action Selection & CityLearn Execution (`module/action/`)**
   - At each control step (or every $N$ steps via `decision_interval_steps`), `PlanningModule.select_actions` queries the LLM with active unnormalized physical observations (`observation_access="active_only"`), parses `<think>...</think>` reasoning blocks and fenced JSON via `extract_json_from_polluted_string`, clamps actions to `[low, high]`, and logs rationales to `3_decisions.json` and KPIs to `4_kpi_summary.json`.

## Repository Structure

Mirrors `ciao-group/Generative-Traffic-Agents`:

```text
src/generative_citylearn_agents/
├── config/
│   └── config.py                        # Presets (fast_demo, phase_all_evs, ollama_local)
├── eval/
│   ├── build_comparison_tables.py       # KPI comparison vs. CityLearn BusinessAsUsualAgent
│   └── qualitative_eval.py              # Persona, schedule & rationale audit summaries
├── llm/
│   ├── base_chat_api.py                 # BaseChatAPI & CallableChatAPI
│   ├── huggingface_chat_api.py          # Local Transformers backend (default: Qwen/Qwen3-4B-Instruct-2507)
│   ├── openai_compatible_chat_api.py    # Ollama, vLLM, LiteLLM, OpenRouter & OpenAI HTTP backend
│   └── mock_chat_api.py                 # Deterministic physics-grounded offline responder
├── model/
│   ├── seed.py                          # BuildingSeedProfile
│   ├── building.py                      # BuildingProfile
│   ├── day_schedule.py                  # DayEnergyPlan
│   └── agent.py                         # ActionDecision, StepDecisionRecord, GenerativeBuildingAgentState
├── module/
│   ├── profile/                         # BuildingSeedGenerator, ProfileModule & description prompt
│   ├── planning/                        # PlanningModule, day schedule & action selection prompts
│   └── action/                          # ActionModule, CityLearnAdapter & LLMAgent(Agent)
├── util/
│   ├── json.py                          # extract_json_from_polluted_string
│   ├── storage.py                       # Storage checkpointing & load_from_storage reload
│   ├── logging.py                       # Structured timestamped logging
│   └── time.py                          # Wall-clock runtime timer
└── energy_simulacra.py                  # End-to-end CLI & Python orchestrator
```

## Installation

```bash
pip install -e .
```

To enable local open-weight HuggingFace `transformers` inference (`Qwen/Qwen3-4B-Instruct-2507`):

```bash
pip install -e ".[llm]"
```

## Quickstart

### 1. Run the Simulacra CLI (GTA `traffic_simulacra.py` style)

Offline deterministic run (no API keys or GPU required):

```bash
citylearn-simulacra --preset fast_demo --backend mock --steps 24 --storage-path storage/demo
```

Local open-weight model via HuggingFace `transformers` (`Qwen/Qwen3-4B-Instruct-2507`):

```bash
citylearn-simulacra --preset phase_all_evs --backend huggingface --model-id Qwen/Qwen3-4B-Instruct-2507 --steps 24
```

Local Ollama / vLLM / OpenAI-compatible server:

```bash
citylearn-simulacra --preset ollama_local --base-url http://localhost:11434/v1 --model-id qwen2.5:7b-instruct --steps 24
```

### 2. Use `LLMAgent` Directly in Any CityLearn Script

`LLMAgent` subclasses `citylearn.agents.base.Agent` and works out-of-the-box with `central_agent=True/False`, `interface="flat"` and `interface="entity"`, and `topology_mode="dynamic"`:

```python
from citylearn.citylearn import CityLearnEnv
from generative_citylearn_agents import LLMAgent

env = CityLearnEnv(
    "citylearn_challenge_2022_phase_all_plus_evs",
    buildings=["Building_1", "Building_2"],
    episode_time_steps=24,
    interface="flat",
    render_mode="none",
)

agent = LLMAgent(
    env,
    backend="openai_compatible",          # or "huggingface" / "mock"
    base_url="http://localhost:11434/v1",
    model_id="qwen2.5:7b-instruct",
    storage_dir="storage/my_experiment",
    enable_day_planning=True,
    decision_interval_steps=1,
    workers=4,
)

observations, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    actions = agent.predict(observations)
    observations, rewards, terminated, truncated, info = env.step(actions)

print(agent.storage.summarize_decisions())
print(env.evaluate_v2().head())
env.close()
```

### 3. Use in CityLearn `schema.json` / CLI

Once `generative-citylearn-agents` is installed in the same Python environment as `citylearn`, `env.load_agent()` and `citylearn simulate` can load `LLMAgent` directly from `schema.json`:

```json
{
  "agent": {
    "type": "generative_citylearn_agents.LLMAgent",
    "attributes": {
      "backend": "openai_compatible",
      "base_url": "http://localhost:11434/v1",
      "model_id": "qwen2.5:7b-instruct",
      "storage_dir": "storage/cli_run",
      "enable_day_planning": true
    }
  }
}
```

### 4. Quantitative & Qualitative Evaluation (`eval/`)

Compare `LLMAgent` against CityLearn's `BusinessAsUsualAgent` and inspect saved rationales:

```python
from generative_citylearn_agents import build_comparison_table, evaluate_qualitative_artifacts

comparison_df = build_comparison_table(
    "citylearn_challenge_2022_phase_all_plus_evs",
    buildings=["Building_1", "Building_2"],
    episode_time_steps=24,
    llm_kwargs={"backend": "mock", "storage_dir": "storage/eval_run"},
)
print(comparison_df.to_string(index=False))
print(evaluate_qualitative_artifacts("storage/eval_run"))
```

## License

Released under the [MIT License](LICENSE).
