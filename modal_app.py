"""Modal entry points (GPU runs, same convention as kotoba-lang/dllm-qwen38).

    modal run modal_app.py::data                                   # build the corpus into the volume once
    modal run modal_app.py::encoder --epochs 2                     # backbone A: ModernBERT-large
    modal run modal_app.py::dllm --epochs 1                        # backbone B: LLaDA-MoE-7B-A1B (LoRA)
    modal run modal_app.py::encoder --limit 64 --test-limit 30     # smoke

Everything lands in volume `typed-decisions-cache` (`/cache/hf` HF cache, `/cache/data` corpus,
`/cache/runs/<name>-<ts>/report.json`) and is echoed back. Single H100 per run.
"""

from __future__ import annotations

import json
import os
import time

import modal

app = modal.App("typed-decisions")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "transformers>=4.56,<5", "datasets", "safetensors", "accelerate", "peft>=0.13", "numpy", "sentencepiece", "tiktoken", "protobuf",
                 "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": "/root"})
    .add_local_dir("src/typed_decisions", remote_path="/root/typed_decisions")
)
vol = modal.Volume.from_name("typed-decisions-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("hf-token")]
DATA = "/cache/data"


def _run_dir(name: str) -> str:
    d = f"/cache/runs/{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(d, exist_ok=True)
    return d


def _gpu() -> str:
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


@app.function(image=image, timeout=3600, volumes={"/cache": vol}, secrets=secrets)
def data_remote(n_train: int, n_val: int, n_test: int) -> dict:
    from typed_decisions import data
    counts = data.build(DATA, n_train, n_val, n_test)
    vol.commit()
    return counts


@app.function(image=image, gpu="H100", timeout=6 * 3600, volumes={"/cache": vol}, secrets=secrets)
def encoder_remote(argv: list[str]) -> dict:
    from typed_decisions import train_encoder
    d = _run_dir("encoder")
    t0 = time.time()
    rep = train_encoder.main(["--data", DATA, "--out", d, "--device", "cuda"] + argv)
    rep.update({"wall_total_s": round(time.time() - t0, 1), "run_dir": d, "gpu": _gpu()})
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


@app.function(image=image, gpu="H100", timeout=6 * 3600, volumes={"/cache": vol}, secrets=secrets)
def dllm_remote(argv: list[str]) -> dict:
    from typed_decisions import train_dllm
    d = _run_dir("dllm")
    t0 = time.time()
    rep = train_dllm.main(["--data", DATA, "--out", d, "--device", "cuda"] + argv)
    rep.update({"wall_total_s": round(time.time() - t0, 1), "run_dir": d, "gpu": _gpu()})
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1)
    vol.commit()
    return rep


def _save(rep: dict, name: str):
    os.makedirs("reports", exist_ok=True)
    p = f"reports/{name}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(p, "w") as f:
        json.dump(rep, f, indent=1)
    print("saved", p)


@app.local_entrypoint()
def data(n_train: int = 6000, n_val: int = 400, n_test: int = 1000):
    print(json.dumps(data_remote.remote(n_train, n_val, n_test), indent=1))


@app.local_entrypoint()
def encoder(model: str = "answerdotai/ModernBERT-large", epochs: float = 2.0, batch: int = 16, lr: float = 3e-5,
            limit: int = 0, test_limit: int = 0, brier_weight: float = 1.0, name: str = "encoder", head_lr: float = 1e-3, pool: str = "span", no_amp: bool = False, attn: str = "sdpa", reference_compile: str = "auto", max_len: int = 8192, bench_n: str = "1,10,100", max_state: int = 512):
    argv = ["--model", model, "--epochs", str(epochs), "--batch", str(batch), "--lr", str(lr), "--brier-weight", str(brier_weight), "--head-lr", str(head_lr), "--pool", pool, "--attn", attn, "--reference-compile", reference_compile, "--max-len", str(max_len), "--bench-n", bench_n, "--max-state", str(max_state)]
    if no_amp:
        argv += ["--no-amp"]
    if limit:
        argv += ["--limit", str(limit)]
    if test_limit:
        argv += ["--test-limit", str(test_limit)]
    rep = encoder_remote.remote(argv)
    _save(rep, name)


@app.local_entrypoint()
def dllm(model: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct", epochs: float = 1.0, batch: int = 4, grad_accum: int = 2,
         lr: float = 1e-4, limit: int = 0, test_limit: int = 0, lora_r: int = 16, eval_batch: int = 8, name: str = "dllm",
         skip_zero_shot: bool = False, eval_steps: str = "1,2,4", no_lora: bool = False, dtype: str = "bfloat16", max_state: int = 512):
    argv = ["--model", model, "--epochs", str(epochs), "--batch", str(batch), "--grad-accum", str(grad_accum), "--lr", str(lr),
            "--lora-r", str(lora_r), "--eval-batch", str(eval_batch), "--eval-steps", eval_steps, "--dtype", dtype, "--max-state", str(max_state)]
    if no_lora:
        argv += ["--no-lora"]
    if limit:
        argv += ["--limit", str(limit)]
    if test_limit:
        argv += ["--test-limit", str(test_limit)]
    if skip_zero_shot:
        argv += ["--skip-zero-shot"]
    rep = dllm_remote.remote(argv)
    _save(rep, name)
