## Installation

The installation is only tested for Windows!

Install uv first: https://docs.astral.sh/uv/getting-started/installation/ — works on macOS, Linux, and Windows.

```bash
git clone <your-repo-url>
cd Resume_Rag
uv venv --python 3.12
```

Activate the venv:

### Windows
```powershell
.venv\Scripts\Activate.ps1
```
> If blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate again.

### Linux & macOS
```bash
source .venv/bin/activate
```

---

Install PyTorch for your hardware (pick one) — selector: https://pytorch.org/get-started/locally/

| Hardware              | Command |
| --------------------- | ------- |
| CPU (any)             | `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA (CUDA)         | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Apple Silicon         | `uv pip install torch` |
| AMD (Linux, ROCm)     | `uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |

Install the rest (from `pyproject.toml` + `uv.lock`):
```bash
uv sync --inexact
```
> `--inexact` keeps the torch you just installed — it lives outside the lockfile, and a bare `uv sync` would remove it.

Optional (GGUF / quantized models). Use the prebuilt index — a bare `pip install llama-cpp-python` compiles from source:
```bash
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Run:
```bash
python local_llm.py qwen-1.5b
```

## Model access

The default models (**Qwen2.5**) are openly downloadable — no account needed. Some models are *gated* and require a free Hugging Face account plus a one-time license accept — most notably **Gemma**. To run a Gemma entry:

1. Create a free account at <https://huggingface.co>, open the model page (e.g. `google/gemma-2-2b-it`), and accept the license (approval is usually instant).
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Authenticate in your activated environment:
```bash
huggingface-cli login    # paste the token  (newer CLI: hf auth login)
```

Skip this and try a gated model, and you'll get a `403 GatedRepoError` — expected. Use a Qwen model, or authenticate.