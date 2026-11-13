from generative_citylearn_agents.config import (
    config_fast_demo,
    config_ollama_local,
    config_phase_all_evs,
    get_default_config,
)
from generative_citylearn_agents.eval import (
    build_comparison_table,
    evaluate_qualitative_artifacts,
)
from generative_citylearn_agents.llm import (
    BaseChatAPI,
    CallableChatAPI,
    HuggingfaceChatAPI,
    OpenAICompatibleChatAPI,
    RuleGuidedMockChatAPI,
    resolve_llm_client,
)
from generative_citylearn_agents.model import (
    ActionDecision,
    BuildingProfile,
    BuildingSeedProfile,
    DayEnergyPlan,
    GenerativeBuildingAgentState,
    StepDecisionRecord,
)
from generative_citylearn_agents.module import (
    ActionModule,
    BuildingSeedGenerator,
    CityLearnAdapter,
    GenerativeBuildingAgent,
    LLMAgent,
    MultiCommunityLLMAdapter,
    PlanningModule,
    ProfileModule,
)
from generative_citylearn_agents.util import (
    AgentArtifactStorage,
    Storage,
    Timer,
    extract_json_from,
    extract_json_from_polluted_string,
)

__version__ = "0.1.0"


def run_energy_simulacra(config=None):
    from generative_citylearn_agents.energy_simulacra import (
        run_energy_simulacra as _run,
    )

    return _run(config)


__all__ = [
    "ActionDecision",
    "ActionModule",
    "AgentArtifactStorage",
    "BaseChatAPI",
    "BuildingProfile",
    "BuildingSeedGenerator",
    "BuildingSeedProfile",
    "CallableChatAPI",
    "CityLearnAdapter",
    "DayEnergyPlan",
    "GenerativeBuildingAgent",
    "GenerativeBuildingAgentState",
    "HuggingfaceChatAPI",
    "LLMAgent",
    "MultiCommunityLLMAdapter",
    "OpenAICompatibleChatAPI",
    "PlanningModule",
    "ProfileModule",
    "RuleGuidedMockChatAPI",
    "StepDecisionRecord",
    "Storage",
    "Timer",
    "build_comparison_table",
    "config_fast_demo",
    "config_ollama_local",
    "config_phase_all_evs",
    "evaluate_qualitative_artifacts",
    "extract_json_from",
    "extract_json_from_polluted_string",
    "get_default_config",
    "resolve_llm_client",
    "run_energy_simulacra",
]
