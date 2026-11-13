from __future__ import annotations

import abc
import inspect
import json
import math
import pickle
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert generative building energy management agent operating in a "
    "CityLearn renewable energy community. Reason carefully about occupant comfort, "
    "solar self-consumption, electricity tariffs, carbon intensity, battery reserve "
    "margins, EV departure deadlines, and electrical service limits. Always respond "
    "with strictly valid JSON matching the requested schema."
)

CONTEXT_START_MARKER = "<<<CITYLEARN_CONTEXT_JSON"
CONTEXT_END_MARKER = "CITYLEARN_CONTEXT_JSON>>>"


def round_clean(value: Any, digits: int = 4) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else 0.0
    if isinstance(value, Mapping):
        return {str(k): round_clean(v, digits=digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_clean(v, digits=digits) for v in value]
    try:
        scalar = float(value)
        return round(scalar, digits) if math.isfinite(scalar) else 0.0
    except (TypeError, ValueError):
        return str(value)


def format_context_block(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(round_clean(payload), indent=2, sort_keys=True)
    return f"{CONTEXT_START_MARKER}\n{serialized}\n{CONTEXT_END_MARKER}"


class BaseChatAPI(abc.ABC):
    """Abstract interface for LLM chat backends."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else str(system_prompt)

    @abc.abstractmethod
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a completion string for a single user prompt."""

    def get_completions(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        return [self.get_completion(prompt, system_prompt=system_prompt) for prompt in prompts]


class CallableChatAPI(BaseChatAPI):
    """Wrap any Python callable ``fn(prompt[, system_prompt]) -> str`` as a ``BaseChatAPI``."""

    def __init__(self, fn: Callable[..., str], system_prompt: Optional[str] = None):
        if not callable(fn):
            raise TypeError("CallableChatAPI requires a callable `fn`.")
        super().__init__(system_prompt=system_prompt)
        self._fn = fn
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            self._accepts_system = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params
            ) or len(params) >= 2
        except (TypeError, ValueError):
            self._accepts_system = False

    def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self._fn is None:
            raise RuntimeError("CallableChatAPI callable was not pickleable and must be re-attached after unpickling.")
        active_system = self.system_prompt if system_prompt is None else str(system_prompt)
        if self._accepts_system:
            return str(self._fn(prompt, active_system))
        return str(self._fn(prompt))

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        try:
            pickle.dumps(self._fn)
        except Exception:
            state["_fn"] = None
        return state
