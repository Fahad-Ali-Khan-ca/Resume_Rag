"""
local_llm.py - a tiny, in-process abstraction for running LLMs locally.

Design goals:
  * Pure Python program. No Ollama / LM Studio / external server or daemon.
    (The model runs *inside* this process, as a library.)
  * Same code on Linux / macOS / Windows, CPU or GPU. No "works on my machine".
  * Swap MODELS by changing one string. Swap the RUNTIME per host via backends.

One interface (`LLM`), two pluggable backends:
  * "transformers" - Hugging Face + PyTorch. Purest Python, any HF model id.
  * "llamacpp"     - llama-cpp-python (GGUF). Quantized, runs on modest hardware.


Run:
  python local_llm.py gemma-2b
  python local_llm.py qwen-14b-q4 "Explain RAG in one sentence."
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


# --------------------------------------------------------------------------- #
# Shared types
# --------------------------------------------------------------------------- #

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.9  # 0.0 => greedy / reproducible
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class LLM(ABC):
    """The single interface the rest of your codebase depends on."""

    name: str

    @abstractmethod
    def generate(
        self, messages: list[Message], config: GenConfig | None = None
    ) -> str: ...

    def stream(
        self, messages: list[Message], config: GenConfig | None = None
    ) -> Iterator[str]:
        # Backends override this with real streaming; default yields once.
        yield self.generate(messages, config)

    def __call__(self, prompt: str, **kw) -> str:
        return self.generate(
            [{"role": "user", "content": prompt}], GenConfig(**kw) if kw else None
        )


def _merge_system_into_user(messages: list[Message]) -> list[Message]:
    """Lowest-common-denominator chat: fold any system text into the first user
    turn. Fallback for templates (e.g. Gemma) that reject a system role."""
    sys = " ".join(m["content"] for m in messages if m["role"] == "system").strip()
    rest = [m for m in messages if m["role"] != "system"]
    if sys and rest and rest[0]["role"] == "user":
        print(f"sys : {sys} and rest : {rest}")
        return [
            {"role": "user", "content": f"{sys}\n\n{rest[0]['content']}"},
            *rest[1:],
        ]
    if sys:
        print(f"sys : {sys}")
        return [{"role": "user", "content": sys}, *rest]
    return rest


# --------------------------------------------------------------------------- #
# Backend 1: Hugging Face transformers + PyTorch
# --------------------------------------------------------------------------- #


class TransformersLLM(LLM):
    def __init__(
        self, model_id: str, *, device: str | None = None, dtype: str | None = None
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self._torch = torch
        self.device = device or self._pick_device(torch)
        print(f"transformers backend: device={self.device}, dtype={dtype or 'default'}")
        torch_dtype = self._pick_dtype(torch, self.device, dtype)

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch_dtype, device_map="auto"
        )
        self.model.eval()

    @staticmethod
    def _pick_device(torch) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _pick_dtype(torch, device: str, override: str | None):
        if override:
            return getattr(torch, override)
        # fp16 on CPU is slow / partially unsupported -> keep CPU in fp32.
        return torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    def _encode(self, messages: list[Message]):
        try:
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        except Exception:
            # Template doesn't accept a system role -> degrade gracefully.
            return self.tok.apply_chat_template(
                _merge_system_into_user(messages),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)

    def _gen_kwargs(self, cfg: GenConfig) -> dict:
        pad = self.tok.pad_token_id
        if pad is None:  # some models (e.g. Llama 3) -> use eos
            eos = self.tok.eos_token_id
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        kw = dict(
            max_new_tokens=cfg.max_new_tokens, pad_token_id=pad, do_sample=cfg.do_sample
        )
        if cfg.do_sample:  # only pass sampling args when sampling
            kw.update(temperature=cfg.temperature, top_p=cfg.top_p)
        return kw

    def generate(self, messages, config=None):
        cfg = config or GenConfig()
        inputs = self._encode(messages)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(cfg))
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages, config=None):
        from transformers import TextIteratorStreamer

        cfg = config or GenConfig()
        inputs = self._encode(messages)
        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(**inputs, **self._gen_kwargs(cfg), streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for piece in streamer:
            yield piece
        thread.join()


# --------------------------------------------------------------------------- #
# Backend 2: llama-cpp-python (GGUF, quantized, very portable)
# --------------------------------------------------------------------------- #


class LlamaCppLLM(LLM):
    def __init__(
        self,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        model_path: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ):
        from llama_cpp import Llama

        self.name = model_path or f"{repo_id}/{filename}"
        common = dict(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        if model_path:
            self.llm = Llama(model_path=model_path, **common)
        else:
            # Auto-downloads the matching GGUF from the Hub on first run.
            self.llm = Llama.from_pretrained(
                repo_id=repo_id, filename=filename, **common
            )

    def generate(self, messages, config=None):
        cfg = config or GenConfig()
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
        )
        return out["choices"][0]["message"]["content"].strip()

    def stream(self, messages, config=None):
        cfg = config or GenConfig()
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
            stream=True,
        ):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta


# --------------------------------------------------------------------------- #
# Registry: swap models by name. This is your "seamless swap" surface.
# --------------------------------------------------------------------------- #


@dataclass
class Spec:
    backend: str  # "transformers" | "llamacpp"
    params: dict = field(default_factory=dict)



REGISTRY: dict[str, Spec] = {
    # transformers (full precision; needs the RAM; great on GPU / Apple Silicon)
    "gemma-2b": Spec(
        "transformers", {"model_id": "google/gemma-2-2b-it"}
    ),
    "gemma-9b": Spec("transformers", {"model_id": "google/gemma-2-9b-it"}),
    "gemma-4-31b": Spec("transformers", {"model_id": "google/gemma-4-31B-it"}),
    "gemma-4e": Spec("transformers", {"model_id": "google/gemma-4-E4B-it"}),
    "qwen-7b": Spec("transformers", {"model_id": "Qwen/Qwen2.5-7B-Instruct"}),
    # llama.cpp (quantized GGUF; runs on modest CPUs / laptops)
    "gemma-9b-q4": Spec(
        "llamacpp",
        {"repo_id": "bartowski/gemma-2-9b-it-GGUF", "filename": "*Q4_K_M.gguf"},
    ),
    "qwen-14b-q4": Spec(
        "llamacpp",
        {"repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF", "filename": "*Q4_K_M.gguf"},
    ),
}

_BACKENDS = {"transformers": TransformersLLM, "llamacpp": LlamaCppLLM}


def load(name: str) -> LLM:
    if name not in REGISTRY:
        raise KeyError(f"{name!r} not in registry. Available: {list(REGISTRY)}")
    spec = REGISTRY[name]
    return _BACKENDS[spec.backend](**spec.params)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"
    prompt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Why are people sad? Explain in one sentence."
    )

    print(f"loading: {model_name}")
    llm = load(model_name)
    print(f"backend: {type(llm).__name__}  |  model: {llm.name}\n")

    msgs = [{"role": "user", "content": prompt}]

    print("--- generate ---")
    print(llm.generate(msgs, GenConfig(max_new_tokens=128)))

    print("\n--- stream ---")
    for tok in llm.stream(msgs, GenConfig(max_new_tokens=128)):
        print(tok, end="", flush=True)
    print()
