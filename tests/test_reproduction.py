"""Regression checks for data conversion and the manuscript's reported results."""

import unittest

import numpy as np

from upward_bias.analysis import build_results, boundary_distances
from upward_bias.calibration import COMPARISONS, pool_bundle, pooled_results
from upward_bias.data import Dataset
from upward_bias.validation import validate_data


class DataTests(unittest.TestCase):
    def test_cross_table_consistency_and_record_counts(self):
        summary = validate_data()
        self.assertEqual(summary["events"], 9)
        self.assertEqual(summary["individual_self_report_rows"], 1885)
        self.assertEqual(summary["aggregate_papercopilot_reports"], 582)
        self.assertEqual(summary["total_reports_represented"], 2467)
        self.assertEqual(summary["iclr_population_rows"], 19084)
        self.assertEqual(summary["iclr_score_bearing_population_rows"], 18783)

    def test_missing_decisions_and_aggregate_sources_remain_distinct(self):
        data = Dataset()
        self.assertFalse(any(r["platform"] == "papercopilot" for r in data.reports()))
        self.assertTrue(any(r["decision"] == "" for r in data.reports()))
        for event in ("www_2025", "kdd_2025"):
            self.assertEqual({r["platform"] for r in data.reports(event)}, {"pooled"})
        for event in ("arr_2024_02", "arr_2024_06", "arr_2025_02", "iclr_2024", "iclr_2025"):
            for platform in ("zhihu", "reddit"):
                expected = [r["review_score_raw"] for r in data.reports(event, platform)
                            if r["decision"] == "" and r["review_score_raw"] is not None]
                np.testing.assert_array_equal(data.sample(event, platform, group="nd"), expected)


class ManuscriptTests(unittest.TestCase):
    def test_pooled_cdf_cohorts_and_inclusive_endpoints(self):
        expected = [(16677, 1285), (16348, 418), (18783, 391)]
        for panel, counts in zip(build_results()["panels"], expected):
            self.assertEqual((panel["population"]["n"], panel["sample"]["n"]), counts)
            for group in ("population", "sample"):
                self.assertEqual(sum(panel[group]["counts"]), panel[group]["n"])
                self.assertEqual(panel[group]["cdf"][-1], 1)
            self.assertLess(panel["p_value"], .001)

    def test_full_report_calibration_matches_figure3(self):
        rows = pooled_results()["comparisons"]
        expected = [(.5311, .1722), (.4510, .0712), (.5177, .0828)]
        for row, (before, after) in zip(rows, expected):
            self.assertEqual(round(row["wasserstein"]["bias_agnostic"], 4), before)
            self.assertEqual(round(row["wasserstein"]["bias_aware"], 4), after)
        self.assertEqual(round(rows[0]["f1"]["bias_agnostic"], 4), .7706)
        self.assertEqual(round(rows[0]["f1"]["bias_aware"], 4), .8440)
        self.assertEqual((rows[0]["withdrawals"]["bias_agnostic"],
                          rows[0]["withdrawals"]["bias_aware"]), (20, 0))

    def test_restricted_calibration_matches_figure_s4(self):
        rows = pooled_results(nd_only=True)["comparisons"]
        expected = [(.3702, .2226), (.4048, .0905), (.4306, .0733)]
        for row, (before, after) in zip(rows, expected):
            self.assertEqual(round(row["wasserstein"]["bias_agnostic"], 4), before)
            self.assertEqual(round(row["wasserstein"]["bias_aware"], 4), after)
        self.assertEqual(round(rows[0]["f1"]["bias_aware"], 4), .8479)
        for restricted, counts in ((False, [(158, 233), (197, 261), (245, 197)]),
                                   (True, [(118, 154), (161, 190), (165, 161)])):
            for comparison, expected_n in zip(COMPARISONS, counts):
                bundle = pool_bundle(comparison, nd_only=restricted)
                self.assertEqual((len(bundle.target_sample), len(bundle.reference_sample)), expected_n)

    def test_boundary_concentration_matches_figure2(self):
        for family, population, sample in (("ARR", 28.3, 47.5), ("ICLR", 25.1, 41.9)):
            result = boundary_distances(family)
            self.assertEqual(round(100 * result["population_proportions"][0], 1), population)
            self.assertEqual(round(100 * result["sample_proportions"][0], 1), sample)
            self.assertEqual(sum(result["population_counts"]), result["population_n"])
            self.assertEqual(sum(result["sample_counts"]), result["sample_n"])


if __name__ == "__main__":
    unittest.main()
