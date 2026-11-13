from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from generative_citylearn_agents.model.building import BuildingProfile
from generative_citylearn_agents.model.day_schedule import DayEnergyPlan
from generative_citylearn_agents.model.seed import BuildingSeedProfile


@dataclass
class ActionDecision:
    """Validated single-actuator decision and natural-language rationale."""

    action_name: str
    value: float
    raw_value: Optional[float]
    low: float
    high: float
    reasoning: str
    clamped: bool = False
    used_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "value": float(self.value),
            "raw_value": None if self.raw_value is None else float(self.raw_value),
            "low": float(self.low),
            "high": float(self.high),
            "reasoning": self.reasoning,
            "clamped": bool(self.clamped),
            "used_fallback": bool(self.used_fallback),
        }


@dataclass
class StepDecisionRecord:
    """Step-level decision log analogous to GTA's decision artifacts."""

    time_step: int
    building_name: str
    hour: float
    day_type: int
    decisions: List[ActionDecision]
    action_vector: List[float]
    observations_snapshot: Dict[str, float]
    day_plan_summary: Optional[str] = None
    raw_response: str = ""
    reused_previous_step: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": int(self.time_step),
            "building_name": self.building_name,
            "hour": float(self.hour),
            "day_type": int(self.day_type),
            "action_vector": [float(v) for v in self.action_vector],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "observations_snapshot": {k: float(v) for k, v in self.observations_snapshot.items()},
            "day_plan_summary": self.day_plan_summary,
            "reused_previous_step": bool(self.reused_previous_step),
        }

    @classmethod
    def clone_for_reused_step(
        cls,
        previous: "StepDecisionRecord",
        *,
        time_step: int,
        observations: Mapping[str, float],
        clamped_vector: List[float],
    ) -> "StepDecisionRecord":
        cloned_decisions: List[ActionDecision] = []
        for idx, decision in enumerate(previous.decisions):
            val = clamped_vector[idx] if idx < len(clamped_vector) else decision.value
            cloned_decisions.append(
                ActionDecision(
                    action_name=decision.action_name,
                    value=float(val),
                    raw_value=decision.raw_value,
                    low=decision.low,
                    high=decision.high,
                    reasoning=decision.reasoning,
                    clamped=bool(val != decision.value or decision.clamped),
                    used_fallback=decision.used_fallback,
                )
            )
        return cls(
            time_step=int(time_step),
            building_name=previous.building_name,
            hour=float(observations.get("hour", previous.hour)),
            day_type=int(round(float(observations.get("day_type", previous.day_type)))),
            decisions=cloned_decisions,
            action_vector=list(clamped_vector),
            observations_snapshot={str(k): float(v) for k, v in observations.items()},
            day_plan_summary=previous.day_plan_summary,
            raw_response="",
            reused_previous_step=True,
        )


@dataclass
class GenerativeBuildingAgentState:
    """Container pairing a seeded building with its persona, schedules, and decisions."""

    building_name: str
    seed: BuildingSeedProfile
    profile: Optional[BuildingProfile] = None
    day_schedules: Dict[int, DayEnergyPlan] = field(default_factory=dict)
    decisions: List[StepDecisionRecord] = field(default_factory=list)
