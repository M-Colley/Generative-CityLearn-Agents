from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from generative_citylearn_agents.llm.base_chat_api import BaseChatAPI


class HuggingfaceChatAPI(BaseChatAPI):
    """Local HuggingFace ``transformers`` chat backend modeled after GTA's ``huggingface_chat_api.py``.

    Defaults to ``Qwen/Qwen3-4B-Instruct-2507`` and loads ``transformers`` + ``torch``
    lazily on first inference so initialization and pickling stay lightweight.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

    def __init__(
        self,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        n_predict: int = 700,
        gpu_id: int = 0,
        temperature: float = 0.2,
        do_sample: Optional[bool] = None,
        device_map: Optional[str] = None,
        torch_dtype: Any = "auto",
        pipeline_instance: Any = None,
    ):
        super().__init__(system_prompt=system_prompt)
        self.model_id = self.DEFAULT_MODEL_ID if not model_id else str(model_id)
        self.n_predict = max(int(n_predict), 16)
        self.gpu_id = int(gpu_id)
        self.temperature = max(float(temperature), 0.0)
        self.do_sample = (self.temperature > 0.0) if do_sample is None else bool(do_sample)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._pipeline = pipeline_instance

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "HuggingfaceChatAPI requires `transformers` and `torch`. "
                "Install optional dependencies with `pip install 'generative-citylearn-agents[llm]'`."
            ) from exc

        pipe_kwargs: Dict[str, Any] = {
            "task": "text-generation",
            "model": self.model_id,
            "torch_dtype": self.torch_dtype,
        }
        if self.device_map is not None:
            pipe_kwargs["device_map"] = self.device_map
        elif torch.cuda.is_available() and self.gpu_id >= 0:
            pipe_kwargs["device"] = self.gpu_id
        else:
            pipe_kwargs["device"] = -1

        self._pipeline = pipeline(**pipe_kwargs)
        return self._pipeline

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        active_system = self.system_prompt if system_prompt is None else str(system_prompt)
        return [
            {"role": "system", "content": active_system},
            {"role": "user", "content": str(prompt)},
        ]

    @staticmethod
    def _extract_generated_content(output: Any) -> str:
        item = output[0] if isinstance(output, list) and len(output) > 0 else output
        if isinstance(item, Mapping):
            generated = item.get("generated_text", "")
            if isinstance(generated, list) and len(generated) > 0:
                last_msg = generated[-1]
                if isinstance(last_msg, Mapping):
                    return str(last_msg.get("content", ""))
                return str(last_msg)
            return str(generated)
        return str(item)

    def get_completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pipe = self._ensure_pipeline()
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.n_predict,
            "do_sample": self.do_sample,
        }
        if self.do_sample and self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        try:
            import torch

            with torch.inference_mode():
                raw_out = pipe(messages, **gen_kwargs)
        except ImportError:
            raw_out = pipe(messages, **gen_kwargs)
        return self._extract_generated_content(raw_out)

    def get_completions(
        self,
        prompts: Sequence[str],
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        if len(prompts) == 0:
            return []
        pipe = self._ensure_pipeline()
        batch_messages = [self._build_messages(p, system_prompt=system_prompt) for p in prompts]
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.n_predict,
            "do_sample": self.do_sample,
        }
        if self.do_sample and self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        try:
            import torch

            with torch.inference_mode():
                outputs = pipe(batch_messages, **gen_kwargs)
        except ImportError:
            outputs = pipe(batch_messages, **gen_kwargs)
        return [self._extract_generated_content(out) for out in outputs]

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_pipeline"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pipeline = None
