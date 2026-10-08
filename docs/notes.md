# Method notes and environment observations

Observations recorded while building and smoke-testing the pipeline on the
target machine. Nothing here is a benchmark result.

## Hardware and software captured from the system (2026-10-07)

| Item | Command | Value recorded |
|---|---|---|
| Chip | `sysctl -n machdep.cpu.brand_string` | `Apple A18 Pro` |
| Model identifier | `system_profiler SPHardwareDataType` | `Mac17,5` |
| Cores | `system_profiler SPHardwareDataType` | 6 (2 performance, 4 efficiency) |
| Memory | `system_profiler SPHardwareDataType` / `sysctl -n hw.memsize` | 8 GB (8589934592 bytes) |
| OS | `sw_vers` | macOS 27.0, build 26A428 |
| llama.cpp | `llama-server --version` (Homebrew) | `version: 0.5.0 (build 11146, commit 7fe450e19)`, `built with AppleClang 21.0.0.21000334 for Darwin arm64` |
| ggml | Homebrew keg path printed by `llama-bench` | `ggml 0.25.1` (`libggml-blas.so` loaded as BLAS backend; Metal backend `MTL`) |
| Python | `python3 --version` | Python 3.14.4 |
| mlx / mlx-lm | `.venv/bin/pip list` | mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 (transformers 5.19.0, numpy 2.5.3) |
| Shell | `bash --version` | GNU bash 3.2.57 (the macOS system bash; scripts avoid bash 4 features) |

## Metal tensor API notice

Every `llama-bench` invocation on this chip prints:

    ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices

llama.cpp's Metal backend therefore runs without the tensor API on this
hardware. The string is captured verbatim into every `manifest.json`
(`host.metal_note`) so that the numbers stay tied to that fact. Numbers from
this machine should not be read as representative of newer chips.

MLX reports the device as `applegpu_g17p` with
`max_recommended_working_set_size` = 5726633984 bytes (about 5.3 GiB) out of
8 GiB of unified memory (`mx.device_info()`). That working-set ceiling, not
the nominal 8 GB, bounds what fits on the GPU side.

## llama-bench flags in the installed build

Checked with `llama-bench --help` on the installed version:

- `-fa` accepts `on|off|auto` (not `0|1`). The scripts pass `on`/`off` and
  encode it as `fa1`/`fa0` in the result file names; the raw JSON carries
  `"flash_attn": 1/0` as llama-bench writes it.
- `-ctk`/`-ctv` set K and V cache types; the sweep sets both to the same
  value (`f16` or `q8_0`).
- `-t`, `-b`, `-ub`, `-p`, `-n`, `-r`, `-o json` behave as expected. With
  `-p 512,2048,8192 -n 128` one invocation emits four test records (three
  prompt-processing, one generation); the sweep keeps one such JSON array per
  configuration.
- `-r 3` yields three entries in `samples_ts`, which is what `summarize.py`
  takes the median and IQR over. Default `-ngl -1` offloads everything to
  Metal, so the thread count mostly affects the CPU-side work.

## Expected failures inside the grid

- A quantised V cache (`-ctv q8_0`) generally requires flash attention in
  llama.cpp. The `fa=off, kv=q8_0` cells of the grid are therefore expected to
  fail and are recorded as `*.failed.json`, counted in the "configs failed"
  column and never used for numbers. They are kept in the grid rather than
  skipped so the failure is observable data, not an assumption.
- The 8192-token prompt (and 16384 in the context curve) may exhaust memory
  for the 3B/4B models on 8 GB. Those points are recorded as failed and the
  run continues; the summary lists them.

## Smoke test (pipeline check only)

Run on 2026-10-07 with `SMOKE=1` (one configuration, `-p 128 -n 32 -r 1` for
the sweep, `-p 128 -n 16 -r 1` for the context script, `p=128, gen=16, 1 rep`
for MLX) against the 1.5B GGUF and its MLX 4-bit equivalent. All five stages
(`fetch.sh`, `sweep.sh`, `context_length.sh`, `mlx_bench.py`, `summarize.py`)
completed and `summarize.py` produced all three table types including the
llama.cpp-vs-MLX comparison. The output lives under `results/smoke/`, which
is gitignored; its numbers are intentionally not reproduced here.

MLX worked on this chip: `mlx_lm.load` and `stream_generate` ran on the GPU
(`Device(gpu, 0)`) with no errors. The deprecation warning
`mx.metal.device_info is deprecated ... Use mx.device_info instead` was seen
once while probing; the script uses `mx.device_info()`.

## Build-time issues encountered

- Bash 3.2 treats an empty array as unbound under `set -u`; optional
  argument arrays use the `${arr[@]+"${arr[@]}"}` idiom.
- `pip install mlx-lm` hit transient DNS failures for `files.pythonhosted.org`
  and resumed on retry; the install completed.
- `shellcheck` is not installed on the target machine, so shell linting runs
  only in CI.


## 2026-10-07: first sweep discarded

The first llama.cpp sweep (1.5B model, 96-configuration grid) was run on a laptop
that was on battery, fell to 5% charge, and slept or woke more than a thousand
times over the host's lifetime. One configuration sat suspended for 1 h 44 min
before it was killed by hand, and other CPU-heavy work ran on the machine during
the sweep. Those results were moved out of `results/` into a git-ignored
quarantine directory and are not published, summarized or quoted. The scripts
were then changed to refuse such conditions and to record them (see
"Measurement conditions" in the README). The sweep has to be redone on mains
power with nothing else running.
