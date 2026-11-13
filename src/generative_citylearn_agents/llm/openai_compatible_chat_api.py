from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional

import requests

from generative_citylearn_agents.llm.base_chat_api import BaseChatAPI


class OpenAICompatibleChatAPI(BaseChatAPI):
    """HTTP client for OpenAI, Ollama, vLLM, LiteLLM, or OpenRouter ``/chat/completions``."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL_ID = "qwen2.5:7b-instruct"

    def __init__(
        self,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        n_predict: int = 700,
        temperature: float = 0.2,
        timeout_seconds: float = 60.0,
        max_http_retries: int = 2,
        extra_body: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(system_prompt=system_prompt)
        env_url = os.environ.get("CITYLEARN_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        resolved_url = base_url or env_url or self.DEFAULT_BASE_URL
        self.base_url = str(resolved_url).rstrip("/")
        env_model = os.environ.get("CITYLEARN_LLM_MODEL")
        self.model_id = str(model_id or env_model or self.DEFAULT_MODEL_ID)
        self.api_key = (
            api_key
            if api_key is not None
            else (os.environ.get("CITYLEARN_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
        )
        self.n_predict = max(int(n_predict), 16)
        self.temperature = max(float(temperature), 0.0)
        self.timeout_seconds = max(float(timeout_seconds), 1.0)
        self.max_http_retries = max(int(max_http_retries), 0)
        self.extra_body = {} if extra_body is None else dict(extra_body)

    def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        active_system = self.system_prompt if system_prompt is None else str(system_prompt)
        endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": active_system},
                {"role": "user", "content": str(prompt)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.n_predict,
            **self.extra_body,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_http_retries + 1):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise ValueError(f"Empty choices in LLM response: {body!r}")
                message = choices[0].get("message") or {}
                if "content" in message and message["content"] is not None:
                    return str(message["content"])
                if "text" in choices[0]:
                    return str(choices[0]["text"])
                raise ValueError(f"Missing completion content in LLM response: {body!r}")
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_http_retries:
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
        raise RuntimeError(f"OpenAICompatibleChatAPI request to {endpoint} failed: {last_exc}") from last_exc
