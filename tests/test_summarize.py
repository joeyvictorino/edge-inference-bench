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


class RenderTests(TempResults):
    def test_markdown_tables(self):
        s = summarize.summarize_host(os.path.join(self.results, "testhost"))
        md = summarize.render_markdown(s)
        self.assertIn("# Results: testhost", md)
        self.assertIn("| `model-a` | t=6 b=1024 ub=512 fa=on kv=q8_0 | 30.0 (IQR 5.0) (tg128) | 131.0 (IQR 1.0) | 102.0 (IQR 1.0) | 2/1 |", md)
        self.assertIn("| `model-b` | t=2 b=256 ub=128 fa=off kv=f16 | 10.0 (n=1) (tg128) | 50.0 (n=1) | n/a | 1/0 |", md)
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
