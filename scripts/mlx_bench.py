#!/usr/bin/env python3
"""Benchmark an MLX 4-bit model with mlx_lm at several prompt lengths.

    .venv/bin/python scripts/mlx_bench.py --model mlx-community/<repo> --out OUT_DIR

For each prompt length a synthetic prompt of exactly N tokens is built from
the model's own tokenizer (natural text, repeated and truncated to N ids),
then `mlx_lm.stream_generate` runs `--gen` tokens, `--reps` times, after one
warm-up. Per repetition the script records prompt tokens/s, generation
tokens/s, time-to-first-token (wall clock from call to first yielded token),
peak Metal memory (`mx.get_peak_memory()`), and the number of tokens actually
generated (the EOS check is disabled so generation runs to --gen unless the
model has no controllable EOS set).

Output: OUT_DIR/manifest.json plus one OUT_DIR/p<N>.json per prompt length,
schema "edge-inference-bench/mlx/v1", read by scripts/summarize.py.
Resumable: existing p<N>.json files are skipped. Failures are recorded as
p<N>.failed.json and the run continues.

Options: --prompts 512,1024,... --gen 128 --reps 3 --smoke (p=128, gen=16, 1 rep)
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback

SCHEMA = "edge-inference-bench/mlx/v1"
DEFAULT_PROMPTS = "512,1024,2048,4096,8192,16384"
FILLER = (
    "The quick brown fox jumps over the lazy dog while the river keeps flowing "
    "toward the sea, and the old clock on the wall measures the afternoon in "
    "slow, even ticks. Numbers like 17, 42 and 1024 appear in the margins of "
    "the notebook next to sketches of bridges, kites and small wooden boats. "
)


def sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def sw_vers(flag):
    try:
        return subprocess.check_output(["sw_vers", flag], text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def host_fingerprint(mx):
    chip = sysctl("machdep.cpu.brand_string") or platform.processor()
    mem = int(sysctl("hw.memsize") or 0)
    macos = sw_vers("-productVersion")
    slug = "%s-%dgb-macos%s" % (
        "".join(c if c.isalnum() else "-" for c in chip.lower()).strip("-"),
        mem // (1024 ** 3),
        macos,
    )
    while "--" in slug:
        slug = slug.replace("--", "-")
    info = {}
    try:
        info = dict(mx.device_info())
    except Exception:  # noqa: BLE001
        pass
    return {
        "chip": chip,
        "model_identifier": sysctl("hw.model"),
        "memory_bytes": mem,
        "cores_total": int(sysctl("hw.ncpu") or 0),
        "cores_performance": int(sysctl("hw.perflevel0.physicalcpu") or 0),
        "cores_efficiency": int(sysctl("hw.perflevel1.physicalcpu") or 0),
        "macos": macos,
        "macos_build": sw_vers("-buildVersion"),
        "host_slug": slug,
        "mlx_device_info": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()},
    }


def build_prompt_ids(tokenizer, n_tokens):
    """Return exactly n_tokens token ids of natural-looking text."""
    reps = max(1, n_tokens // 16)
    ids = []
    text = FILLER * reps
    while len(ids) < n_tokens:
        ids = tokenizer.encode(text, add_special_tokens=False)
        text += FILLER * reps
    return ids[:n_tokens]


def disable_eos(tokenizer):
    """Best effort: make stream_generate run to max_tokens instead of stopping at EOS."""
    try:
        tokenizer.eos_token_ids = set()
        return True
    except Exception:  # noqa: BLE001
        try:
            tokenizer._eos_token_ids = set()  # noqa: SLF001
            return True
        except Exception:  # noqa: BLE001
            return False


def run_once(mx, stream_generate, model, tokenizer, prompt_ids, gen):
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    ttft = None
    last = None
    for resp in stream_generate(model, tokenizer, prompt_ids, max_tokens=gen):
        if ttft is None:
            ttft = time.perf_counter() - t0
        last = resp
    wall = time.perf_counter() - t0
    if last is None:
        raise RuntimeError("stream_generate yielded nothing")
    return {
        "prompt_tokens": int(last.prompt_tokens),
        "prompt_tps": float(last.prompt_tps),
        "generation_tokens": int(last.generation_tokens),
        "generation_tps": float(last.generation_tps),
        "ttft_s": float(ttft),
        "wall_s": float(wall),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "finish_reason": last.finish_reason,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="MLX repo id, e.g. mlx-community/<name>-4bit, or a local path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="p=128, gen=16, 1 rep")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    prompts = [int(p) for p in args.prompts.split(",") if p.strip()]
    gen, reps = args.gen, args.reps
    if args.smoke:
        prompts, gen, reps = [128], 16, 1

    import mlx.core as mx  # noqa: PLC0415  (import after argparse so --help works without mlx)
    import mlx_lm  # noqa: PLC0415
    from mlx_lm import load, stream_generate  # noqa: PLC0415

    os.makedirs(args.out, exist_ok=True)
    print("loading %s" % args.model, flush=True)
    model, tokenizer = load(args.model)
    eos_disabled = disable_eos(tokenizer)

    revision = ""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        snap = snapshot_download(args.model, local_files_only=True)
        revision = os.path.basename(snap)
    except Exception:  # noqa: BLE001
        pass

    manifest = {
        "schema": "edge-inference-bench/manifest/v1",
        "kind": "mlx",
        "engine": "mlx_lm",
        "engine_versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "python": platform.python_version()},
        "model_repo": args.model,
        "model_revision": revision,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {"prompts": " ".join(str(p) for p in prompts), "gen": gen, "repetitions": reps},
        "eos_disabled": eos_disabled,
        "smoke": bool(args.smoke),
        "host": host_fingerprint(mx),
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path):
        print("manifest.json exists in %s; keeping it" % args.out, flush=True)
    else:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # warm-up (compiles kernels, touches weights); not recorded
    warm = build_prompt_ids(tokenizer, 32)
    run_once(mx, stream_generate, model, tokenizer, warm, 8)

    for n in prompts:
        name = "p%d" % n
        final = os.path.join(args.out, name + ".json")
        failed = os.path.join(args.out, name + ".failed.json")
        if os.path.exists(final):
            print("[%s] exists, skipping" % name, flush=True)
            continue
        if os.path.exists(failed) and not args.retry_failed:
            continue
        print("[%s] gen=%d reps=%d" % (name, gen, reps), flush=True)
        try:
            ids = build_prompt_ids(tokenizer, n)
            samples = [run_once(mx, stream_generate, model, tokenizer, ids, gen) for _ in range(reps)]
            record = {
                "schema": SCHEMA,
                "engine": "mlx_lm",
                "model_repo": args.model,
                "n_prompt": n,
                "n_gen": gen,
                "repetitions": reps,
                "samples": samples,
            }
            with open(final, "w") as fh:
                json.dump(record, fh, indent=2, sort_keys=True)
                fh.write("\n")
            if os.path.exists(failed):
                os.remove(failed)
            s = samples[0]
            print("  pp %.1f t/s, tg %.1f t/s, ttft %.3fs, peak %.0f MiB" % (
                s["prompt_tps"], s["generation_tps"], s["ttft_s"], s["peak_memory_bytes"] / 2 ** 20), flush=True)
        except Exception as exc:  # noqa: BLE001
            with open(failed, "w") as fh:
                json.dump({"config": name, "error": repr(exc), "traceback": traceback.format_exc()}, fh, indent=2)
                fh.write("\n")
            print("  failed: %r (recorded %s)" % (exc, os.path.basename(failed)), flush=True)
    print("mlx_bench complete -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
