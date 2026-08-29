from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Literal


# --------------------------------------------------------------------------- #
# Shared types
# --------------------------------------------------------------------------- #

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}

AcceleratorKind = Literal[
    "nvidia-cuda",
    "amd-rocm",
    "apple-mps",
    "cpu",
]


@dataclass
class GenConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    stop: list[str] = field(default_factory=list)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


@dataclass(frozen=True)
class AcceleratorInfo:
    kind: AcceleratorKind
    device: str
    name: str
    runtime_version: str | None = None


class LLM(ABC):

    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> str:
        ...

    def stream(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> Iterator[str]:
        yield self.generate(messages, config)

    def __call__(self, prompt: str, **kwargs) -> str:
        config = GenConfig(**kwargs) if kwargs else None
        return self.generate(
            [{"role": "user", "content": prompt}],
            config,
        )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _merge_system_into_user(messages: list[Message]) -> list[Message]:

    system_text = " ".join(
        message["content"]
        for message in messages
        if message["role"] == "system"
    ).strip()

    remaining = [
        message
        for message in messages
        if message["role"] != "system"
    ]

    if system_text and remaining and remaining[0]["role"] == "user":
        return [
            {
                "role": "user",
                "content": f"{system_text}\n\n{remaining[0]['content']}",
            },
            *remaining[1:],
        ]

    if system_text:
        return [
            {"role": "user", "content": system_text},
            *remaining,
        ]

    return remaining


# --------------------------------------------------------------------------- #
# Backend 1: Hugging Face Transformers + PyTorch
# --------------------------------------------------------------------------- #

class TransformersLLM(LLM):
    """
    Hugging Face Transformers backend.

    NVIDIA and AMD intentionally share device="cuda":
      * NVIDIA build -> torch.version.cuda is populated
      * AMD ROCm     -> torch.version.hip is populated
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self._torch = torch

        self.accelerator = self._detect_accelerator(torch)
        self.device = self._resolve_device(
            torch,
            requested=device,
            detected=self.accelerator,
        )

        torch_dtype = self._pick_dtype(
            torch,
            device=self.device,
            override=dtype,
        )

        self._print_hardware_summary(
            torch_dtype=torch_dtype,
        )

        self.tok = AutoTokenizer.from_pretrained(
            model_id,
        )

        model_kwargs = {
            "dtype": torch_dtype,
        }

        # device_map="auto" works for both CUDA and ROCm because PyTorch ROCm
        # deliberately exposes AMD GPUs through the CUDA device interface.
        #
        # It also permits CPU offload if the model is larger than VRAM.
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        # CPU and Apple MPS are simpler when explicitly moved as a whole.
        if self.device in {"cpu", "mps"}:
            self.model.to(self.device)

        self.model.eval()

        # For automatically dispatched/offloaded models, determine where the
        # input embedding layer actually lives rather than blindly assuming
        # that every tensor belongs on cuda:0.
        self.input_device = self._find_input_device()

    # ------------------------------------------------------------------ #
    # Hardware detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_accelerator(torch) -> AcceleratorInfo:
        """
        Detect the available PyTorch accelerator.

        ROCm note:
        AMD GPUs still report through torch.cuda.is_available() and use
        device="cuda". torch.version.hip is the reliable distinction.
        """

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

            hip_version = getattr(
                torch.version,
                "hip",
                None,
            )

            if hip_version:
                return AcceleratorInfo(
                    kind="amd-rocm",
                    device="cuda",
                    name=gpu_name,
                    runtime_version=str(hip_version),
                )

            cuda_version = getattr(
                torch.version,
                "cuda",
                None,
            )

            return AcceleratorInfo(
                kind="nvidia-cuda",
                device="cuda",
                name=gpu_name,
                runtime_version=(
                    str(cuda_version)
                    if cuda_version
                    else None
                ),
            )

        mps_backend = getattr(
            torch.backends,
            "mps",
            None,
        )

        if mps_backend and mps_backend.is_available():
            return AcceleratorInfo(
                kind="apple-mps",
                device="mps",
                name="Apple Metal Performance Shaders",
            )

        return AcceleratorInfo(
            kind="cpu",
            device="cpu",
            name="CPU",
        )

    @staticmethod
    def _resolve_device(
        torch,
        *,
        requested: str | None,
        detected: AcceleratorInfo,
    ) -> str:
        """
        Resolve optional user device overrides.

        Accepted aliases:
          auto / None
          cuda / nvidia
          amd / rocm / hip
          mps
          cpu

        AMD ROCm still resolves to "cuda".
        """

        if requested is None:
            return detected.device

        value = requested.strip().lower()

        if value == "auto":
            return detected.device

        if value in {"cuda", "nvidia"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU requested, but this PyTorch build "
                    "does not expose a GPU."
                )

            if value == "nvidia" and getattr(
                torch.version,
                "hip",
                None,
            ):
                raise RuntimeError(
                    "NVIDIA was requested, but the installed "
                    "PyTorch build is ROCm/HIP for AMD."
                )

            return "cuda"

        if value in {"amd", "rocm", "hip"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "AMD ROCm GPU requested, but PyTorch reports "
                    "no GPU accelerator."
                )

            if not getattr(
                torch.version,
                "hip",
                None,
            ):
                raise RuntimeError(
                    "AMD ROCm GPU requested, but the installed "
                    "PyTorch build is not a ROCm/HIP build."
                )

            # ROCm intentionally uses the CUDA device string.
            return "cuda"

        if value == "mps":
            mps_backend = getattr(
                torch.backends,
                "mps",
                None,
            )

            if not (
                mps_backend
                and mps_backend.is_available()
            ):
                raise RuntimeError(
                    "MPS requested but unavailable."
                )

            return "mps"

        if value == "cpu":
            return "cpu"

        raise ValueError(
            f"Unsupported device {requested!r}. "
            "Use auto, cuda, nvidia, amd, rocm, hip, mps, or cpu."
        )

    @staticmethod
    def _pick_dtype(
        torch,
        *,
        device: str,
        override: str | None,
    ):

        if override:
            if not hasattr(torch, override):
                raise ValueError(
                    f"Unknown PyTorch dtype: {override!r}"
                )

            return getattr(
                torch,
                override,
            )

        if device == "cuda":
            bf16_checker = getattr(
                torch.cuda,
                "is_bf16_supported",
                None,
            )

            try:
                if (
                    callable(bf16_checker)
                    and bf16_checker()
                ):
                    return torch.bfloat16
            except Exception:
                # Some backends/drivers may not implement the probe cleanly.
                pass

            return torch.float16

        if device == "mps":
            return torch.float16

        return torch.float32

    def _print_hardware_summary(
        self,
        *,
        torch_dtype,
    ) -> None:
        runtime = ""

        if self.accelerator.runtime_version:
            runtime = (
                f", runtime={self.accelerator.runtime_version}"
            )

        print(
            "transformers backend: "
            f"accelerator={self.accelerator.kind}, "
            f"device={self.device}, "
            f"gpu={self.accelerator.name}, "
            f"dtype={torch_dtype}"
            f"{runtime}"
        )

    # ------------------------------------------------------------------ #
    # Model/input placement
    # ------------------------------------------------------------------ #

    def _find_input_device(self) -> str:
        """
        Return the device holding the model's input embedding weights.

        This matters when Accelerate has split/offloaded a model across GPU
        and CPU due to limited VRAM.
        """

        try:
            embeddings = self.model.get_input_embeddings()

            if embeddings is not None:
                device = embeddings.weight.device

                if device.type != "meta":
                    return str(device)

        except Exception:
            pass

        try:
            device = self.model.device

            if device.type != "meta":
                return str(device)

        except Exception:
            pass

        return self.device

    # ------------------------------------------------------------------ #
    # Prompt encoding / generation
    # ------------------------------------------------------------------ #

    def _encode(
        self,
        messages: list[Message],
    ):
        try:
            encoded = self.tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

        except Exception:
            encoded = self.tok.apply_chat_template(
                _merge_system_into_user(messages),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

        return encoded.to(
            self.input_device,
        )

    def _gen_kwargs(
        self,
        cfg: GenConfig,
    ) -> dict:
        pad_token_id = self.tok.pad_token_id

        if pad_token_id is None:
            eos_token_id = self.tok.eos_token_id

            pad_token_id = (
                eos_token_id[0]
                if isinstance(
                    eos_token_id,
                    (list, tuple),
                )
                else eos_token_id
            )

        kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "pad_token_id": pad_token_id,
            "do_sample": cfg.do_sample,
        }

        if cfg.do_sample:
            kwargs.update(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

        return kwargs

    def generate(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> str:
        cfg = config or GenConfig()

        inputs = self._encode(
            messages,
        )

        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **self._gen_kwargs(cfg),
            )

        prompt_length = inputs["input_ids"].shape[1]

        new_tokens = output[
            0,
            prompt_length:,
        ]

        return self.tok.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    def stream(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        cfg = config or GenConfig()

        inputs = self._encode(
            messages,
        )

        streamer = TextIteratorStreamer(
            self.tok,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        kwargs = {
            **inputs,
            **self._gen_kwargs(cfg),
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=kwargs,
            daemon=True,
        )

        thread.start()

        for piece in streamer:
            yield piece

        thread.join()

class Gemma4LLM(LLM):
    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        dtype: str | None = None,
    ):
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForMultimodalLM,
        )

        self.name = model_id
        self._torch = torch

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if dtype:
            torch_dtype = getattr(torch, dtype)

        elif self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

        else:
            torch_dtype = torch.float32

        print(
            f"gemma4 backend: "
            f"device={self.device}, "
            f"dtype={torch_dtype}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map="auto",
            )
        )

        self.model.eval()

    def generate(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> str:

        cfg = config or GenConfig()

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            self.model.device
        )

        with self._torch.inference_mode():

            output = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=(
                    cfg.temperature
                    if cfg.do_sample
                    else None
                ),
                top_p=(
                    cfg.top_p
                    if cfg.do_sample
                    else None
                ),
            )

        prompt_length = inputs["input_ids"].shape[1]

        generated = output[
            0,
            prompt_length:
        ]

        return self.processor.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

# --------------------------------------------------------------------------- #
# Backend 2: llama-cpp-python (GGUF)
# --------------------------------------------------------------------------- #

class LlamaCppLLM(LLM):
    """
    llama.cpp backend.

    GPU acceleration depends on how llama-cpp-python itself was compiled.
    The Python interface here remains the same.
    """

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

        if not model_path and not (
            repo_id
            and filename
        ):
            raise ValueError(
                "Provide either model_path or both repo_id and filename."
            )

        self.name = (
            model_path
            or f"{repo_id}/{filename}"
        )

        common = {
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }

        if model_path:
            self.llm = Llama(
                model_path=model_path,
                **common,
            )
        else:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **common,
            )

    def generate(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> str:
        cfg = config or GenConfig()

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
        )

        content = output[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

        return (
            content.strip()
            if content
            else ""
        )

    def stream(
        self,
        messages: list[Message],
        config: GenConfig | None = None,
    ) -> Iterator[str]:
        cfg = config or GenConfig()

        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            stop=cfg.stop or None,
            stream=True,
        ):
            delta = chunk[
                "choices"
            ][0][
                "delta"
            ].get(
                "content"
            )

            if delta:
                yield delta


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

@dataclass
class Spec:
    backend: str
    params: dict = field(
        default_factory=dict,
    )

REGISTRY: dict[str, Spec] = {

    # ------------------------------------------------------------------ #
    # Transformers - Gemma 2 / Qwen
    # ------------------------------------------------------------------ #

    "gemma-2b": Spec(
        "transformers",
        {
            "model_id": "google/gemma-2-2b-it",
        },
    ),

    "gemma-9b": Spec(
        "transformers",
        {
            "model_id": "google/gemma-2-9b-it",
        },
    ),

    "qwen-7b": Spec(
        "transformers",
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
        },
    ),

    # ------------------------------------------------------------------ #
    # Gemma 4
    # ------------------------------------------------------------------ #

    "gemma-4-e2b": Spec(
        "gemma4",
        {
            "model_id": "google/gemma-4-E2B-it",
        },
    ),

    "gemma-4-e4b": Spec(
        "gemma4",
        {
            "model_id": "google/gemma-4-E4B-it",
        },
    ),

    "gemma-4-12b": Spec(
        "gemma4",
        {
            "model_id": "google/gemma-4-12B-it",
        },
    ),

    "gemma-4-26b-a4b": Spec(
        "gemma4",
        {
            "model_id": "google/gemma-4-26B-A4B-it",
        },
    ),

    "gemma-4-31b": Spec(
        "gemma4",
        {
            "model_id": "google/gemma-4-31B-it",
        },
    ),

    # ------------------------------------------------------------------ #
    # llama.cpp / GGUF
    # ------------------------------------------------------------------ #

    "gemma-9b-q4": Spec(
        "llamacpp",
        {
            "repo_id": "bartowski/gemma-2-9b-it-GGUF",
            "filename": "*Q4_K_M.gguf",
        },
    ),

    "qwen-14b-q4": Spec(
        "llamacpp",
        {
            "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
            "filename": "*Q4_K_M.gguf",
        },
    ),
}
_BACKENDS = {
    "transformers": TransformersLLM,
    "llamacpp": LlamaCppLLM,
    "gemma4": Gemma4LLM,

}


def load(
    name: str,
    **overrides,
) -> LLM:
    """
    Load a registered model.

    Optional keyword overrides are useful for tests:

        load("gemma-2b", device="cpu")
        load("gemma-2b", device="amd")
        load("gemma-2b", dtype="float16")
    """

    if name not in REGISTRY:
        raise KeyError(
            f"{name!r} not in registry. "
            f"Available: {list(REGISTRY)}"
        )

    spec = REGISTRY[name]

    params = {
        **spec.params,
        **overrides,
    }

    return _BACKENDS[
        spec.backend
    ](
        **params,
    )


# --------------------------------------------------------------------------- #
# Demo / hardware sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    model_name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "gemma-2b"
    )

    prompt = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Explain RAG in one sentence."
    )

    print(
        f"loading: {model_name}"
    )

    llm = load(
        model_name
    )

    print(
        f"backend: {type(llm).__name__} "
        f"| model: {llm.name}"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    print("\n--- generate ---")

    print(
        llm.generate(
            messages,
            GenConfig(
                max_new_tokens=128,
                temperature=0.0,
            ),
        )
    )

    print("\n--- stream ---")

    for token in llm.stream(
        messages,
        GenConfig(
            max_new_tokens=128,
            temperature=0.0,
        ),
    ):
        print(
            token,
            end="",
            flush=True,
        )

    print()
