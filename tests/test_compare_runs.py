"""compare_runs.py applies the README's reproducibility rule: a replica
reproduces a metric when its median is within the published IQR."""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import compare_runs  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "results", "testhost")
CONFIG = os.path.join("model-a", "sweep", "t4_b512_ub256_fa1_kvf16.json")


class CompareRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eib-cmp-")
        self.a = os.path.join(self.tmp, "a", "testhost")
        self.b = os.path.join(self.tmp, "b", "testhost")
        shutil.copytree(FIXTURE, self.a)
        shutil.copytree(FIXTURE, self.b)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def scale_replica(self, factor):
        path = os.path.join(self.b, CONFIG)
        with open(path) as fh:
            records = json.load(fh)
        for r in records:
            r["samples_ts"] = [x * factor for x in r["samples_ts"]]
        with open(path, "w") as fh:
            json.dump(records, fh)

    def row(self, cmp, test):
        return next(r for r in cmp["rows"] if r["item"] == "t4_b512_ub256_fa1_kvf16" and r["test"] == test)

    def test_identical_runs_reproduce(self):
        cmp = compare_runs.compare(self.a, self.b)
        for m in cmp["by_model"].values():
            self.assertEqual(m["metrics"], m["within_published_iqr"])

    def test_a_shift_beyond_the_published_iqr_does_not_reproduce(self):
        self.scale_replica(1.5)
        cmp = compare_runs.compare(self.a, self.b)
        r = self.row(cmp, "pp512")
        self.assertFalse(r["within_published_iqr"])
        self.assertAlmostEqual(r["delta_pct"], 50.0)
        self.assertLess(cmp["by_model"]["model-a"]["within_published_iqr"], cmp["by_model"]["model-a"]["metrics"])

    def test_render_is_deterministic(self):
        self.assertEqual(compare_runs.render(compare_runs.compare(self.a, self.b)),
                         compare_runs.render(compare_runs.compare(self.a, self.b)))


if __name__ == "__main__":
    unittest.main()
