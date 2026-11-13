from __future__ import annotations

from typing import Any, Optional, Union

from generative_citylearn_agents.llm.base_chat_api import (
    CONTEXT_END_MARKER,
    CONTEXT_START_MARKER,
    DEFAULT_SYSTEM_PROMPT,
    BaseChatAPI,
    CallableChatAPI,
    format_context_block,
)
from generative_citylearn_agents.llm.huggingface_chat_api import HuggingfaceChatAPI
from generative_citylearn_agents.llm.mock_chat_api import RuleGuidedMockChatAPI
from generative_citylearn_agents.llm.openai_compatible_chat_api import OpenAICompatibleChatAPI


def resolve_llm_client(
    *,
    llm_client: Optional[Union[BaseChatAPI, Any]] = None,
    backend: str = "mock",
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    n_predict: int = 700,
    gpu_id: int = 0,
) -> BaseChatAPI:
    if isinstance(llm_client, BaseChatAPI):
        return llm_client
    if callable(llm_client):
        return CallableChatAPI(llm_client, system_prompt=system_prompt)
    if llm_client is not None:
        raise TypeError("llm_client must be a BaseChatAPI instance, a callable, or None.")

    normalized = str(backend or "mock").strip().lower()
    if normalized in {"mock", "rule_guided_mock", "offline"}:
        return RuleGuidedMockChatAPI(system_prompt=system_prompt)
    if normalized in {"huggingface", "hf", "transformers"}:
        return HuggingfaceChatAPI(
            model_id=model_id,
            system_prompt=system_prompt,
            n_predict=n_predict,
            gpu_id=gpu_id,
            temperature=temperature,
        )
    if normalized in {"openai_compatible", "openai", "ollama", "vllm", "openrouter"}:
        return OpenAICompatibleChatAPI(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
        )
    raise ValueError(
        f"Unsupported LLM backend {backend!r}. Expected 'mock', 'huggingface', or 'openai_compatible'."
    )


__all__ = [
    "BaseChatAPI",
    "CONTEXT_END_MARKER",
    "CONTEXT_START_MARKER",
    "CallableChatAPI",
    "DEFAULT_SYSTEM_PROMPT",
    "HuggingfaceChatAPI",
    "OpenAICompatibleChatAPI",
    "RuleGuidedMockChatAPI",
    "format_context_block",
    "resolve_llm_client",
]
