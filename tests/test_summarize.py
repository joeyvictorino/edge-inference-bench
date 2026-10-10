"""Unit tests for scripts/summarize.py against the synthetic fixture in tests/fixtures."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import summarize  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "results")
FIXTURE_README = os.path.join(HERE, "fixtures", "README.fixture.md")


def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


class TempResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eib-test-")
        self.results = os.path.join(self.tmp, "results")
        shutil.copytree(FIXTURE, self.results)
        self.readme = os.path.join(self.tmp, "README.md")
        shutil.copy(FIXTURE_README, self.readme)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class MedIqrTests(unittest.TestCase):
    def test_three_samples(self):
        s = summarize.med_iqr([30, 10, 20])
        self.assertEqual(s["median"], 20)
        self.assertEqual(s["q1"], 15)
        self.assertEqual(s["q3"], 25)
        self.assertEqual(s["iqr"], 10)
        self.assertEqual(s["n"], 3)

    def test_single_sample_has_zero_iqr(self):
        s = summarize.med_iqr([42.0])
        self.assertEqual(s["median"], 42.0)
        self.assertEqual(s["iqr"], 0.0)
        self.assertEqual(s["n"], 1)

    def test_empty_is_none(self):
        self.assertIsNone(summarize.med_iqr([]))

    def test_fmt(self):
        self.assertEqual(summarize.fmt(summarize.med_iqr([10, 20, 30])), "20.0 (IQR 10.0)")
        self.assertEqual(summarize.fmt(summarize.med_iqr([7])), "7.0 (n=1)")
        self.assertEqual(summarize.fmt(None), "n/a")


class LoadingTests(TempResults):
    def test_sweep_loads_configs_and_counts_failures(self):
        sw = summarize.load_sweep(os.path.join(self.results, "testhost", "model-a", "sweep"))
        self.assertEqual(sorted(sw["configs"]), ["t4_b512_ub256_fa1_kvf16", "t6_b1024_ub512_fa1_kvq8_0"])
        self.assertEqual(sw["failed"], ["t2_b256_ub128_fa0_kvq8_0"])
        tests = sw["configs"]["t4_b512_ub256_fa1_kvf16"]["tests"]
        self.assertEqual(tests["pp512"]["median"], 110.0)
        self.assertEqual(tests["tg128"]["median"], 21.0)

    def test_best_config_is_highest_median_generation(self):
        sw = summarize.load_sweep(os.path.join(self.results, "testhost", "model-a", "sweep"))
        name, score, key = summarize.best_config(sw)
        self.assertEqual(name, "t6_b1024_ub512_fa1_kvq8_0")
        self.assertEqual(score, 30.0)  # median of [30, 25, 35], not the mean
        self.assertEqual(key, "tg128")

    def test_context_points(self):
        cx = summarize.load_context(os.path.join(self.results, "testhost", "model-a", "context"))
        self.assertEqual(sorted(cx["points"]), [512, 2048])
        self.assertEqual(cx["points"][2048]["pp"]["median"], 82.0)
        self.assertEqual(cx["points"][2048]["tg_len"], 64)
        self.assertEqual(cx["failed"], ["p16384"])

    def test_mlx_points(self):
        ml = summarize.load_mlx(os.path.join(self.results, "testhost", "model-a", "mlx"))
        self.assertEqual(ml["points"][512]["pp"]["median"], 210.0)
        self.assertEqual(ml["points"][512]["tg"]["iqr"], 1.0)
        self.assertEqual(ml["points"][2048]["peak_memory_bytes"]["median"], 1024 * 2 ** 20)

    def test_parse_config_name(self):
        self.assertEqual(summarize.parse_config_name("t2_b256_ub128_fa0_kvq8_0"),
                         {"threads": 2, "batch": 256, "ubatch": 128, "flash_attn": 0, "kv_cache": "q8_0"})
        self.assertIsNone(summarize.parse_config_name("manifest"))


class ConditionsTests(TempResults):
    """Only runs made under controlled conditions may be published."""

    def sweep_dir(self):
        return os.path.join(self.results, "testhost", "model-a", "sweep")

    def set_env(self, name, **changes):
        path = os.path.join(self.sweep_dir(), name + ".env.json")
        with open(path) as fh:
            env = json.load(fh)
        for key, value in changes.items():
            env[key] = value
        with open(path, "w") as fh:
            json.dump(env, fh)

    def load(self):
        return summarize.load_sweep(self.sweep_dir())

    def test_clean_fixture_has_nothing_excluded(self):
        sw = self.load()
        self.assertEqual(sw["excluded"], {})
        self.assertEqual(len(sw["configs"]), 2)

    def test_battery_before_or_after_is_excluded(self):
        self.set_env("t4_b512_ub256_fa1_kvf16", power_before={"source": "Battery", "battery_pct": 5})
        sw = self.load()
        self.assertEqual(sw["excluded"], {"t4_b512_ub256_fa1_kvf16": "not on mains power"})
        self.assertNotIn("t4_b512_ub256_fa1_kvf16", sw["configs"])
        self.set_env("t6_b1024_ub512_fa1_kvq8_0", power_after={"source": "Battery", "battery_pct": 90})
        self.assertEqual(self.load()["configs"], {})

    def test_below_idle_threshold_is_published_but_counted(self):
        self.set_env("t4_b512_ub256_fa1_kvf16", quiet_threshold_met=False, cpu_idle_before_pct=55.0)
        sw = self.load()
        self.assertIn("t4_b512_ub256_fa1_kvf16", sw["configs"])
        self.assertEqual(sw["below_idle"], ["t4_b512_ub256_fa1_kvf16"])
        summary = summarize.summarize_host(os.path.join(self.results, "testhost"))
        self.assertEqual(summary["models"]["model-a"]["sweep"]["n_configs_below_idle_threshold"], 1)
        md = summarize.render_markdown(summary)
        self.assertIn("Started below the CPU-idle threshold", md)
        self.assertIn("`model-a`: 1 of 2 sweep configurations", md)

    def test_runner_host_is_labelled(self):
        man_path = os.path.join(self.sweep_dir(), "manifest.json")
        with open(man_path) as fh:
            man = json.load(fh)
        man.setdefault("host", {}).update({"host_kind": "gha-macos", "cores_performance": 0, "cores_efficiency": 0,
                                           "cores_total": 3,
                                           "runner": {"virtualized": True, "runs_on": "macos-15", "image_os": "macos15",
                                                      "image_version": "1", "run_url": "https://example/run/1"}})
        with open(man_path, "w") as fh:
            json.dump(man, fh)
        summary = summarize.summarize_host(os.path.join(self.results, "testhost"))
        md = summarize.render_markdown(summary)
        self.assertIn("GitHub-hosted runner, not a physical machine", md)
        self.assertIn("3 logical CPUs", md)
        self.assertIn("https://example/run/1", md)

    def test_replica_comparison_is_rendered_with_a_warning(self):
        sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
        import compare_runs
        host = os.path.join(self.results, "testhost")
        rep = os.path.join(self.results, "replicas", "testhost", "run-1-replica2")
        shutil.copytree(host, rep)
        path = os.path.join(rep, "model-a", "sweep", "t4_b512_ub256_fa1_kvf16.json")
        with open(path) as fh:
            records = json.load(fh)
        for r in records:
            r["samples_ts"] = [x * 0.5 for x in r["samples_ts"]]
        with open(path, "w") as fh:
            json.dump(records, fh)
        with open(os.path.join(rep, "compare.json"), "w") as fh:
            json.dump(compare_runs.compare(host, rep), fh)
        summary = summarize.summarize_host(host)
        self.assertEqual([r["replica"] for r in summary["reproducibility"]], ["run-1-replica2"])
        md = summarize.render_markdown(summary)
        self.assertIn("Not reproduced (`model-a`)", md)
        self.assertNotIn("Not reproduced (`model-b`)", md)
        self.assertIn("## Reproducibility", md)
        # the replica directory itself is never summarised as a host
        self.assertFalse(summarize.is_host_dir(os.path.join(self.results, "replicas")))

    def test_sleep_and_timeout_are_excluded(self):
        self.set_env("t4_b512_ub256_fa1_kvf16", slept_during_run=True)
        self.set_env("t6_b1024_ub512_fa1_kvq8_0", timed_out=True)
        sw = self.load()
        self.assertEqual(sw["excluded"], {"t4_b512_ub256_fa1_kvf16": "machine slept during the run",
                                          "t6_b1024_ub512_fa1_kvq8_0": "timed out"})

    def test_missing_conditions_record_is_not_trusted(self):
        os.remove(os.path.join(self.sweep_dir(), "t4_b512_ub256_fa1_kvf16.env.json"))
        sw = self.load()
        self.assertEqual(sw["excluded"], {"t4_b512_ub256_fa1_kvf16": "no conditions record"})

    def test_interrupted_files_are_listed_never_read(self):
        with open(os.path.join(self.sweep_dir(), "t2_b256_ub128_fa1_kvf16.interrupted.json"), "w") as fh:
            fh.write("[]")
        sw = self.load()
        self.assertEqual(sw["interrupted"], ["t2_b256_ub128_fa1_kvf16"])
        self.assertNotIn("t2_b256_ub128_fa1_kvf16", sw["configs"])

    def test_excluded_count_reaches_the_summary_and_table(self):
        self.set_env("t4_b512_ub256_fa1_kvf16", slept_during_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            summarize.main([os.path.join(self.results, "testhost"), "--no-readme"])
        with open(os.path.join(self.results, "testhost", "summary.json")) as fh:
            summary = json.load(fh)
        sw = summary["models"]["model-a"]["sweep"]
        self.assertEqual(sw["n_configs_excluded_conditions"], 1)
        self.assertEqual(sw["excluded_conditions"], {"t4_b512_ub256_fa1_kvf16": "machine slept during the run"})
        with open(os.path.join(self.results, "testhost", "summary.md")) as fh:
            self.assertIn("ok/failed/excluded", fh.read())


class StageConditionsTests(TempResults):
    """The context-length and MLX stages are published or excluded as a whole."""

    def stage_dir(self, stage):
        return os.path.join(self.results, "testhost", "model-a", stage)

    def write_conditions(self, stage, **changes):
        path = os.path.join(self.stage_dir(stage), "stage-conditions.json")
        with open(path) as fh:
            rec = json.load(fh)
        rec.update(changes)
        with open(path, "w") as fh:
            json.dump(rec, fh)

    def test_clean_stages_are_published(self):
        self.assertTrue(summarize.load_context(self.stage_dir("context"))["points"])
        self.assertTrue(summarize.load_mlx(self.stage_dir("mlx"))["points"])

    def test_context_stage_on_battery_is_excluded_whole(self):
        self.write_conditions("context", power_before={"source": "Battery", "battery_pct": 5})
        cx = summarize.load_context(self.stage_dir("context"))
        self.assertEqual(cx["points"], {})
        self.assertEqual(cx["excluded_stage"], "not on mains power")

    def test_mlx_stage_that_slept_is_excluded_whole(self):
        self.write_conditions("mlx", slept_during_run=True)
        ml = summarize.load_mlx(self.stage_dir("mlx"))
        self.assertEqual(ml["points"], {})
        self.assertEqual(ml["excluded_stage"], "machine slept during the run")

    def test_stage_without_a_conditions_record_is_excluded(self):
        os.remove(os.path.join(self.stage_dir("context"), "stage-conditions.json"))
        cx = summarize.load_context(self.stage_dir("context"))
        self.assertEqual(cx["points"], {})
        self.assertEqual(cx["excluded_stage"], "no conditions record")

    def test_exclusion_is_visible_in_summary_json(self):
        self.write_conditions("mlx", slept_during_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            summarize.main([os.path.join(self.results, "testhost"), "--no-readme"])
        with open(os.path.join(self.results, "testhost", "summary.json")) as fh:
            summary = json.load(fh)
        self.assertEqual(summary["models"]["model-a"]["mlx"]["excluded_stage"], "machine slept during the run")
        self.assertEqual(summary["models"]["model-a"]["mlx"]["points"], {})


class RenderTests(TempResults):
    def test_markdown_tables(self):
        s = summarize.summarize_host(os.path.join(self.results, "testhost"))
        md = summarize.render_markdown(s)
        self.assertIn("# Results: testhost", md)
        self.assertIn("| `model-a` | t=6 b=1024 ub=512 fa=on kv=q8_0 | 30.0 (IQR 5.0) (tg128) | 131.0 (IQR 1.0) | 102.0 (IQR 1.0) | 2/1/0 |", md)
        self.assertIn("| `model-b` | t=2 b=256 ub=128 fa=off kv=f16 | 10.0 (n=1) (tg128) | 50.0 (n=1) | n/a | 1/0/0 |", md)
        self.assertIn("| 2048 | 82.0 (IQR 2.0) |", md)
        self.assertIn("### `model-a` vs `example/model-a-4bit`", md)
        self.assertIn("| 512 | 100.0 (IQR 0.0) | 210.0 (IQR 10.0) | 20.0 (IQR 0.0) | 41.0 (IQR 1.0) | 128 | 2.438 (IQR 0.116) | 1024 |", md)
        self.assertIn("Failed points", md)
        self.assertNotIn("Smoke-test data", md)
        self.assertIn("synthetic metal note", md)

    def test_smoke_flag_in_markdown(self):
        s = summarize.summarize_host(os.path.join(self.results, "smoke"))
        self.assertTrue(s["smoke"])
        self.assertIn("Smoke-test data", summarize.render_markdown(s))


class MainTests(TempResults):
    def run_main(self, target, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize.main([target, "--readme", self.readme, *extra])

    def test_writes_summaries_and_readme_from_root(self):
        self.run_main(self.results)
        host = os.path.join(self.results, "testhost")
        self.assertTrue(os.path.exists(os.path.join(host, "summary.md")))
        with open(os.path.join(host, "summary.json")) as fh:
            js = json.load(fh)
        self.assertEqual(js["models"]["model-a"]["sweep"]["best"]["name"], "t6_b1024_ub512_fa1_kvq8_0")
        self.assertEqual(js["models"]["model-a"]["sweep"]["n_configs_failed"], 1)
        with open(self.readme) as fh:
            readme = fh.read()
        self.assertIn("Text before the block must survive.", readme)
        self.assertIn("Text after the block must survive too.", readme)
        self.assertIn("### Best llama.cpp configuration per model", readme)
        self.assertIn("`model-a`", readme)
        self.assertNotIn("placeholder that summarize.py replaces", readme)
        # smoke host is summarised on disk but never reaches the README
        self.assertTrue(os.path.exists(os.path.join(self.results, "smoke", "summary.md")))
        self.assertNotIn("Smoke-test data", readme)
        self.assertNotIn("Results: smoke", readme)

    def test_single_host_dir_argument(self):
        self.run_main(os.path.join(self.results, "testhost"))
        self.assertTrue(os.path.exists(os.path.join(self.results, "testhost", "summary.json")))
        self.assertFalse(os.path.exists(os.path.join(self.results, "smoke", "summary.json")))

    def test_placeholder_when_nothing_publishable(self):
        shutil.rmtree(os.path.join(self.results, "testhost"))
        self.run_main(self.results)
        with open(self.readme) as fh:
            readme = fh.read()
        self.assertIn(summarize.PLACEHOLDER, readme)
        self.assertNotIn("Smoke-test data", readme)

    def test_no_readme_flag(self):
        before = read_bytes(self.readme)
        self.run_main(self.results, "--no-readme")
        self.assertEqual(before, read_bytes(self.readme))

    def test_deterministic(self):
        self.run_main(self.results)
        paths = [os.path.join(self.results, "testhost", "summary.md"),
                 os.path.join(self.results, "testhost", "summary.json"), self.readme]
        first = [read_bytes(p) for p in paths]
        self.run_main(self.results)
        second = [read_bytes(p) for p in paths]
        self.assertEqual(first, second)

    def test_missing_markers_is_an_error(self):
        with open(self.readme, "w") as fh:
            fh.write("no markers here\n")
        with self.assertRaises(SystemExit):
            self.run_main(self.results)


if __name__ == "__main__":
    unittest.main()
