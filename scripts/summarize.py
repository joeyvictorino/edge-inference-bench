#!/usr/bin/env python3
"""Aggregate raw benchmark JSON into summary tables.

    python3 scripts/summarize.py RESULTS_DIR [--no-readme] [--readme PATH]

RESULTS_DIR is either one host directory (results/<host>/, containing one
sub-directory per model id with sweep/, context/ and mlx/ inside) or the
results/ root (every host directory below it is summarised).

For every host the script writes

    results/<host>/summary.md     human-readable tables
    results/<host>/summary.json   the same numbers, machine-readable

and rewrites the block between <!-- RESULTS:BEGIN --> and <!-- RESULTS:END -->
in README.md. Output is deterministic: identical inputs give identical bytes.

Statistics: per configuration and metric, the median and inter-quartile range
(Q3 - Q1, inclusive quantiles) of the per-repetition tokens/s samples
(`samples_ts` from llama-bench; per-sample fields from mlx_bench.py).
Only completed raw files are used; *.failed.json files are counted, never read
for numbers. Runs whose manifest says "smoke": true are reported but flagged.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"
PLACEHOLDER = (
    "_Results pending._ No sweep has been run on this machine yet. This block is "
    "rewritten by `scripts/summarize.py` from the raw JSON under `results/`; it "
    "is never edited by hand."
)


# ---------------------------------------------------------------- statistics
def med_iqr(values):
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    median = statistics.median(vals)
    if len(vals) < 2:
        return {"median": median, "iqr": 0.0, "q1": median, "q3": median, "n": 1}
    q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    return {"median": median, "iqr": q3 - q1, "q1": q1, "q3": q3, "n": len(vals)}


def fmt(stat, digits=1):
    if not stat:
        return "n/a"
    if stat["n"] > 1:
        return "%.*f (IQR %.*f)" % (digits, stat["median"], digits, stat["iqr"])
    return "%.*f (n=1)" % (digits, stat["median"])


def fmt_plain(stat, digits=1):
    if not stat:
        return "n/a"
    return "%.*f" % (digits, stat["median"])


# ---------------------------------------------------------------- loading
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_manifest(d):
    p = os.path.join(d, "manifest.json")
    return load_json(p) if os.path.exists(p) else {}


def parse_config_name(name):
    m = re.match(r"^t(\d+)_b(\d+)_ub(\d+)_fa([01])_kv(.+)$", name)
    if not m:
        return None
    return {"threads": int(m.group(1)), "batch": int(m.group(2)), "ubatch": int(m.group(3)),
            "flash_attn": int(m.group(4)), "kv_cache": m.group(5)}


def llama_tests(records):
    """Normalise a llama-bench JSON array into {"pp512": stat, "tg128": stat, ...}."""
    out = {}
    for r in records:
        n_p, n_g = int(r.get("n_prompt", 0)), int(r.get("n_gen", 0))
        if n_p and not n_g:
            key = "pp%d" % n_p
        elif n_g and not n_p:
            key = "tg%d" % n_g
        else:
            key = "pp%d+tg%d" % (n_p, n_g)
        samples = r.get("samples_ts") or [r.get("avg_ts")]
        out[key] = med_iqr(samples)
    return out


def load_sweep(d):
    manifest = read_manifest(d)
    configs = {}
    failed = sorted(os.path.basename(p)[: -len(".failed.json")] for p in glob.glob(os.path.join(d, "*.failed.json")))
    interrupted = sorted(os.path.basename(p)[: -len(".interrupted.json")] for p in glob.glob(os.path.join(d, "*.interrupted.json")))
    excluded = {}
    below_idle = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        base = os.path.basename(path)
        if base == "manifest.json" or base.endswith((".failed.json", ".interrupted.json", ".env.json")):
            continue
        name = base[:-5]
        cfg = parse_config_name(name)
        if cfg is None:
            continue
        # Numbers are published only for runs made under controlled
        # conditions: mains power, no sleep during the run, no timeout. A
        # missing conditions record (results from before it existed) is not
        # trusted either.
        why = uncontrolled_reason(os.path.join(d, name + ".env.json"))
        if why:
            excluded[name] = why
            continue
        records = load_json(path)
        if not isinstance(records, list) or not records:
            continue
        if not idle_threshold_met(os.path.join(d, name + ".env.json")):
            below_idle.append(name)
        configs[name] = {"config": cfg, "tests": llama_tests(records),
                         "model_type": records[0].get("model_type", ""),
                         "model_size": records[0].get("model_size")}
    return {"manifest": manifest, "configs": configs, "failed": failed,
            "interrupted": interrupted, "excluded": excluded, "below_idle": below_idle}


def idle_threshold_met(env_path):
    """False only when a conditions record says the run started below the idle
    threshold. That can only happen on a GitHub runner (BENCH_HOST=gha-macos);
    on the operator's machine such a run is refused before it starts. Records
    from before the field existed count as met, as they were refused otherwise."""
    if not os.path.exists(env_path):
        return True
    env = load_json(env_path)
    return not (isinstance(env, dict) and env.get("quiet_threshold_met") is False)


def uncontrolled_reason(env_path):
    """Why a configuration's conditions disqualify it, or '' if it is clean."""
    if not os.path.exists(env_path):
        return "no conditions record"
    env = load_json(env_path)
    if not isinstance(env, dict):
        return "unreadable conditions record"
    for key in ("power_before", "power_after"):
        if (env.get(key) or {}).get("source") != "AC":
            return "not on mains power"
    if env.get("slept_during_run"):
        return "machine slept during the run"
    if env.get("timed_out"):
        return "timed out"
    return ""


def load_context(d):
    manifest = read_manifest(d)
    points = {}
    failed = sorted(os.path.basename(p)[: -len(".failed.json")] for p in glob.glob(os.path.join(d, "*.failed.json")))
    for path in sorted(glob.glob(os.path.join(d, "p*.json"))):
        base = os.path.basename(path)
        if base.endswith(".failed.json"):
            continue
        m = re.match(r"^p(\d+)\.json$", base)
        if not m:
            continue
        records = load_json(path)
        if not isinstance(records, list) or not records:
            continue
        tests = llama_tests(records)
        pp = next((v for k, v in tests.items() if k.startswith("pp")), None)
        tg_key = next((k for k in tests if k.startswith("tg")), None)
        points[int(m.group(1))] = {"pp": pp, "tg": tests.get(tg_key) if tg_key else None,
                                   "tg_len": int(tg_key[2:]) if tg_key else None}
    excluded_stage = uncontrolled_reason(os.path.join(d, "stage-conditions.json")) if points else ""
    if excluded_stage:
        points = {}
    below_idle = bool(points) and not idle_threshold_met(os.path.join(d, "stage-conditions.json"))
    return {"manifest": manifest, "points": points, "failed": failed, "excluded_stage": excluded_stage,
            "below_idle_threshold": below_idle}


def load_mlx(d):
    manifest = read_manifest(d)
    points = {}
    failed = sorted(os.path.basename(p)[: -len(".failed.json")] for p in glob.glob(os.path.join(d, "*.failed.json")))
    for path in sorted(glob.glob(os.path.join(d, "p*.json"))):
        base = os.path.basename(path)
        if base.endswith(".failed.json"):
            continue
        rec = load_json(path)
        if not isinstance(rec, dict) or not rec.get("samples"):
            continue
        s = rec["samples"]
        points[int(rec["n_prompt"])] = {
            "pp": med_iqr([x["prompt_tps"] for x in s]),
            "tg": med_iqr([x["generation_tps"] for x in s]),
            "ttft_s": med_iqr([x["ttft_s"] for x in s]),
            "peak_memory_bytes": med_iqr([x["peak_memory_bytes"] for x in s]),
            "n_gen": int(rec.get("n_gen", 0)),
            "generation_tokens": med_iqr([x.get("generation_tokens", rec.get("n_gen", 0)) for x in s]),
        }
    excluded_stage = uncontrolled_reason(os.path.join(d, "stage-conditions.json")) if points else ""
    if excluded_stage:
        points = {}
    below_idle = bool(points) and not idle_threshold_met(os.path.join(d, "stage-conditions.json"))
    return {"manifest": manifest, "points": points, "failed": failed, "excluded_stage": excluded_stage,
            "below_idle_threshold": below_idle}


def load_host(host_dir):
    models = {}
    for mdir in sorted(p for p in glob.glob(os.path.join(host_dir, "*")) if os.path.isdir(p)):
        mid = os.path.basename(mdir)
        entry = {}
        if os.path.isdir(os.path.join(mdir, "sweep")):
            entry["sweep"] = load_sweep(os.path.join(mdir, "sweep"))
        if os.path.isdir(os.path.join(mdir, "context")):
            entry["context"] = load_context(os.path.join(mdir, "context"))
        if os.path.isdir(os.path.join(mdir, "mlx")):
            entry["mlx"] = load_mlx(os.path.join(mdir, "mlx"))
        if entry:
            models[mid] = entry
    return models


def is_host_dir(d):
    return any(os.path.isdir(os.path.join(d, m, sub)) for m in os.listdir(d)
               for sub in ("sweep", "context", "mlx") if os.path.isdir(os.path.join(d, m)))


# ---------------------------------------------------------------- analysis
def tg_key_of(tests):
    keys = sorted((k for k in tests if k.startswith("tg")), key=lambda k: int(k[2:]))
    return keys[-1] if keys else None


def best_config(sweep):
    """Configuration with the highest median generation tokens/s (ties: name order)."""
    best = None
    for name in sorted(sweep["configs"]):
        tests = sweep["configs"][name]["tests"]
        k = tg_key_of(tests)
        if not k or not tests[k]:
            continue
        score = tests[k]["median"]
        if best is None or score > best[1]:
            best = (name, score, k)
    return best


def config_label(cfg):
    return "t=%d b=%d ub=%d fa=%s kv=%s" % (cfg["threads"], cfg["batch"], cfg["ubatch"],
                                            "on" if cfg["flash_attn"] else "off", cfg["kv_cache"])


def summarize_host(host_dir):
    models = load_host(host_dir)
    host_info, timestamps, smoke = {}, [], False
    for entry in models.values():
        for part in entry.values():
            man = part.get("manifest") or {}
            if man.get("host") and not host_info:
                host_info = man["host"]
            if man.get("timestamp"):
                timestamps.append(man["timestamp"])
            smoke = smoke or bool(man.get("smoke"))
    summary = {
        "schema": "edge-inference-bench/summary/v1",
        "host_dir": os.path.basename(os.path.normpath(host_dir)),
        "host": host_info,
        "data_timestamps": {"first": min(timestamps) if timestamps else None,
                            "last": max(timestamps) if timestamps else None},
        "smoke": smoke,
        "models": {},
    }
    summary["reproducibility"] = load_reproducibility(host_dir)
    for mid, entry in models.items():
        m = {}
        if "sweep" in entry:
            sw = entry["sweep"]
            best = best_config(sw)
            m["sweep"] = {
                "n_configs_ok": len(sw["configs"]),
                "n_configs_failed": len(sw["failed"]),
                "failed": sw["failed"],
                "n_configs_interrupted": len(sw["interrupted"]),
                "n_configs_excluded_conditions": len(sw["excluded"]),
                "excluded_conditions": sw["excluded"],
                "n_configs_below_idle_threshold": len(sw["below_idle"]),
                "llama_cpp_version": (sw["manifest"].get("host") or {}).get("llama_cpp_version"),
                "model_sha256": sw["manifest"].get("model_sha256"),
                "best": None,
                "configs": {name: {"config": c["config"], "tests": c["tests"]} for name, c in sorted(sw["configs"].items())},
            }
            if best:
                name, _, tgk = best
                m["sweep"]["best"] = {"name": name, "config": sw["configs"][name]["config"],
                                      "tg_key": tgk, "tests": sw["configs"][name]["tests"],
                                      "model_type": sw["configs"][name]["model_type"]}
        if "context" in entry:
            cx = entry["context"]
            m["context"] = {"failed": cx["failed"], "excluded_stage": cx["excluded_stage"],
                            "below_idle_threshold": cx["below_idle_threshold"],
                            "points": {str(k): v for k, v in sorted(cx["points"].items())}}
        if "mlx" in entry:
            ml = entry["mlx"]
            m["mlx"] = {"failed": ml["failed"], "excluded_stage": ml["excluded_stage"],
                        "below_idle_threshold": ml["below_idle_threshold"],
                        "model_repo": ml["manifest"].get("model_repo"),
                        "engine_versions": ml["manifest"].get("engine_versions"),
                        "points": {str(k): v for k, v in sorted(ml["points"].items())}}
        summary["models"][mid] = m
    return summary


def load_reproducibility(host_dir):
    """Replica comparisons written by compare_runs.py for this host, found at
    <results>/replicas/<host>/<replica>/compare.json. Only the per-model
    counts are carried into the summary; the full table stays beside the
    replica."""
    host = os.path.basename(os.path.normpath(host_dir))
    pattern = os.path.join(os.path.dirname(os.path.normpath(host_dir)), "replicas", host, "*", "compare.json")
    out = []
    for path in sorted(glob.glob(pattern)):
        cmp = load_json(path)
        out.append({"replica": os.path.basename(os.path.dirname(path)), "rule": cmp.get("rule"),
                    "by_model": cmp.get("by_model", {})})
    return out


# ---------------------------------------------------------------- rendering
def render_markdown(summary):
    L = []
    h = summary["host"] or {}
    L.append("# Results: %s" % summary["host_dir"])
    L.append("")
    if summary["smoke"]:
        L.append("> **Smoke-test data.** These files come from `SMOKE=1` runs (one tiny configuration, "
                 "one repetition). They prove the pipeline works and are not benchmark results.")
        L.append("")
    if h:
        runner = h.get("runner") or {}
        if h.get("host_kind") == "gha-macos":
            L.append("> **GitHub-hosted runner, not a physical machine.** These numbers come from a virtual "
                     "machine (`runs-on: %s`, image %s %s) on shared data-centre hardware. They describe that "
                     "runner class, not any laptop, and are not comparable with the other hosts in this "
                     "repository." % (runner.get("runs_on", "?"), runner.get("image_os", "?"),
                                      runner.get("image_version", "?")))
            L.append("")
        if h.get("cores_performance", 0) or h.get("cores_efficiency", 0):
            cores = "%d cores (%d performance + %d efficiency)" % (
                h.get("cores_total", 0), h.get("cores_performance", 0), h.get("cores_efficiency", 0))
        else:
            cores = "%d logical CPUs" % h.get("cores_total", 0)
        L.append("Host: %s, %d GB memory, %s, macOS %s (%s). "
                 "llama.cpp %s." % (h.get("chip", "?"), int(h.get("memory_bytes", 0)) // 2 ** 30,
                                    cores, h.get("macos", "?"), h.get("macos_build", "?"),
                                    h.get("llama_cpp_version", "?")))
        if runner.get("run_url"):
            L.append("Produced by GitHub Actions run %s." % runner["run_url"])
        if h.get("metal_note"):
            L.append("Metal note recorded by llama.cpp: `%s`." % h["metal_note"])
        ts = summary["data_timestamps"]
        if ts["first"]:
            L.append("Data collected between %s and %s (UTC)." % (ts["first"], ts["last"]))
        L.append("")
    for rep in summary.get("reproducibility") or []:
        for mid, m in sorted(rep["by_model"].items()):
            if m["metrics"] and m["within_published_iqr"] < m["metrics"]:
                L.append("> **Not reproduced (`%s`).** A second run on another runner (`%s`) matched %d of %d "
                         "metrics within the published inter-quartile range; median difference %.1f%%, largest "
                         "%.1f%%. Treat these numbers as one observation, not a benchmark result." % (
                             mid, rep["replica"], m["within_published_iqr"], m["metrics"],
                             m["median_abs_delta_pct"] or 0.0, m["max_abs_delta_pct"] or 0.0))
                L.append("")
    L.append("Values are the median of the per-repetition tokens/s samples with the inter-quartile range in "
             "parentheses; `(n=1)` marks a single repetition; `n/a` means no completed run.")
    L.append("")

    # --- best config per model
    L.append("## Best llama.cpp configuration per model")
    L.append("")
    L.append("Best = highest median generation tokens/s across the sweep grid. Prompt columns are the "
             "same configuration's prompt-processing throughput.")
    L.append("")
    pp_cols = sorted({int(k[2:]) for m in summary["models"].values()
                      for k in ((m.get("sweep") or {}).get("best") or {}).get("tests", {}) if k.startswith("pp")})
    header = ["Model", "Configuration", "Generation t/s"] + ["Prompt %d t/s" % p for p in pp_cols] + ["Configs ok/failed/excluded"]
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "|".join("---" for _ in header) + "|")
    for mid, m in summary["models"].items():
        sw = m.get("sweep")
        if not sw:
            continue
        b = sw["best"]
        if not b:
            row = [mid, "n/a", "n/a"] + ["n/a"] * len(pp_cols)
        else:
            row = ["`%s`" % mid, config_label(b["config"]), "%s (%s)" % (fmt(b["tests"][b["tg_key"]]), b["tg_key"])]
            row += [fmt(b["tests"].get("pp%d" % p)) for p in pp_cols]
        row.append("%d/%d/%d" % (sw["n_configs_ok"], sw["n_configs_failed"], sw["n_configs_excluded_conditions"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    busy = [(mid, m["sweep"]["n_configs_below_idle_threshold"], m["sweep"]["n_configs_ok"])
            for mid, m in summary["models"].items() if (m.get("sweep") or {}).get("n_configs_below_idle_threshold")]
    busy_ctx = [mid for mid, m in summary["models"].items() if (m.get("context") or {}).get("below_idle_threshold")]
    if busy or busy_ctx:
        parts = ["`%s`: %d of %d sweep configurations" % (mid, n, ok) for mid, n, ok in busy]
        parts += ["`%s`: the context-length stage" % mid for mid in busy_ctx]
        L.append("Started below the CPU-idle threshold (recorded, not refused, on a GitHub runner; the "
                 "`cpu_idle_before_pct` in each conditions file gives the level): " + "; ".join(parts) + ".")
        L.append("")

    # --- throughput vs context
    L.append("## Throughput versus prompt length (llama.cpp)")
    L.append("")
    ctx_models = [(mid, m["context"]) for mid, m in summary["models"].items() if m.get("context")]
    if ctx_models:
        sizes = sorted({int(p) for _, c in ctx_models for p in c["points"]})
        L.append("Prompt-processing tokens/s by prompt length (llama-bench defaults apart from `-p`; "
                 "see `context/manifest.json` for the exact grid).")
        L.append("")
        header = ["Prompt tokens"] + ["`%s`" % mid for mid, _ in ctx_models]
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join("---" for _ in header) + "|")
        for s in sizes:
            row = [str(s)] + [fmt((c["points"].get(str(s)) or {}).get("pp")) for _, c in ctx_models]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        tg_len = next((v.get("tg_len") for _, c in ctx_models for v in c["points"].values() if v.get("tg_len")), None)
        L.append("Generation tokens/s (%s tokens) measured in the same runs:" % (tg_len if tg_len else "n"))
        L.append("")
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join("---" for _ in header) + "|")
        for s in sizes:
            row = [str(s)] + [fmt((c["points"].get(str(s)) or {}).get("tg")) for _, c in ctx_models]
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        failed = [(mid, c["failed"]) for mid, c in ctx_models if c["failed"]]
        if failed:
            L.append("Failed points (see the `*.failed.json` files): " +
                     "; ".join("`%s`: %s" % (mid, ", ".join(f)) for mid, f in failed) + ".")
            L.append("")
    else:
        L.append("_No context-length runs found._")
        L.append("")

    # --- reproducibility
    if summary.get("reproducibility"):
        L.append("## Reproducibility")
        L.append("")
        L.append("Each replica is a second run of the same pipeline, compared with `scripts/compare_runs.py`. "
                 "Rule: a metric reproduces when the replica's median is within the published run's "
                 "inter-quartile range. Full tables: `results/replicas/%s/<replica>/compare.md`." % summary["host_dir"])
        L.append("")
        L.append("| Replica | Model | Metrics compared | Reproduced | Median abs. difference | Largest abs. difference |")
        L.append("|---|---|---|---|---|---|")
        for rep in summary["reproducibility"]:
            for mid, m in sorted(rep["by_model"].items()):
                L.append("| `%s` | `%s` | %d | %d | %s | %s |" % (
                    rep["replica"], mid, m["metrics"], m["within_published_iqr"],
                    "%.1f%%" % m["median_abs_delta_pct"] if m["median_abs_delta_pct"] is not None else "n/a",
                    "%.1f%%" % m["max_abs_delta_pct"] if m["max_abs_delta_pct"] is not None else "n/a"))
        L.append("")

    # --- llama.cpp vs MLX
    L.append("## llama.cpp versus MLX, same base model")
    L.append("")
    pairs = [(mid, m) for mid, m in summary["models"].items() if m.get("mlx") and m["mlx"]["points"]]
    if pairs:
        L.append("MLX rows come from `scripts/mlx_bench.py` (4-bit MLX conversion of the same base model, "
                 "generation length in the `MLX gen` column). llama.cpp prompt numbers are taken from the "
                 "context-length run at the same prompt length; llama.cpp generation numbers from the same run. "
                 "TTFT and peak memory are only measured for MLX.")
        L.append("")
        for mid, m in pairs:
            ml = m["mlx"]
            ev = ml.get("engine_versions") or {}
            L.append("### `%s` vs `%s`" % (mid, ml.get("model_repo") or "mlx"))
            L.append("")
            if ev:
                L.append("mlx %s, mlx_lm %s." % (ev.get("mlx", "?"), ev.get("mlx_lm", "?")))
                L.append("")
            ctx = (m.get("context") or {}).get("points", {})
            sizes = sorted({int(p) for p in ml["points"]} | {int(p) for p in ctx})
            header = ["Prompt tokens", "llama.cpp prompt t/s", "MLX prompt t/s", "llama.cpp gen t/s",
                      "MLX gen t/s", "MLX gen tokens", "MLX TTFT s", "MLX peak MiB"]
            L.append("| " + " | ".join(header) + " |")
            L.append("|" + "|".join("---" for _ in header) + "|")
            for s in sizes:
                c = ctx.get(str(s)) or {}
                x = ml["points"].get(str(s)) or {}
                pk = x.get("peak_memory_bytes")
                row = [str(s), fmt(c.get("pp")), fmt(x.get("pp")), fmt(c.get("tg")), fmt(x.get("tg")),
                       fmt_plain(x.get("generation_tokens"), 0) if x else "n/a",
                       fmt(x.get("ttft_s"), 3) if x else "n/a",
                       ("%.0f" % (pk["median"] / 2 ** 20)) if pk else "n/a"]
                L.append("| " + " | ".join(row) + " |")
            L.append("")
            if ml["failed"]:
                L.append("MLX failed points: " + ", ".join(ml["failed"]) + ".")
                L.append("")
    else:
        L.append("_No MLX runs found._")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def readme_block(summaries, results_root):
    if not summaries:
        return PLACEHOLDER + "\n"
    parts = []
    for s in summaries:
        rel = os.path.relpath(os.path.join(results_root, s["host_dir"], "summary.md"), os.path.dirname(results_root) or ".")
        parts.append("Generated by `scripts/summarize.py`; do not edit by hand. Full tables: [`%s`](%s)." % (rel, rel))
        parts.append("")
        body = render_markdown(s).split("\n", 1)[1]  # drop the H1
        body = re.sub(r"^## ", "### ", body, flags=re.M)
        body = re.sub(r"^### (`.*` vs)", "#### \\1", body, flags=re.M)
        parts.append(body.strip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def rewrite_readme(path, block):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN not in text or END not in text:
        raise SystemExit("README markers %s / %s not found in %s" % (BEGIN, END, path))
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    new = pre + BEGIN + "\n" + block + END + post
    if new != text:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)
        return True
    return False


def write_if_changed(path, content):
    old = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            old = fh.read()
    if old != content:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--no-readme", action="store_true", help="do not touch README.md")
    ap.add_argument("--readme", default=None, help="README path (default: <repo>/README.md)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.results_dir)
    if not os.path.isdir(root):
        raise SystemExit("not a directory: %s" % root)
    if is_host_dir(root):
        host_dirs, results_root = [root], os.path.dirname(root)
    else:
        host_dirs = sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p) and is_host_dir(p))
        results_root = root

    summaries = []
    for hd in host_dirs:
        s = summarize_host(hd)
        summaries.append(s)
        changed_md = write_if_changed(os.path.join(hd, "summary.md"), render_markdown(s))
        changed_js = write_if_changed(os.path.join(hd, "summary.json"), json.dumps(s, indent=2, sort_keys=True) + "\n")
        print("%s: %d model(s); summary.md %s, summary.json %s" % (
            os.path.basename(hd), len(s["models"]),
            "updated" if changed_md else "unchanged", "updated" if changed_js else "unchanged"))
    if not summaries:
        print("no host directories with results under %s" % root)

    if not args.no_readme:
        readme = args.readme or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
        # never let smoke data into the README
        publishable = [s for s in summaries if not s["smoke"] and s["host_dir"] != "smoke"]
        changed = rewrite_readme(readme, readme_block(publishable, results_root))
        print("README results block %s" % ("updated" if changed else "unchanged"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
