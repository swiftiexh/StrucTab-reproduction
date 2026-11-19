import random
import unittest

import numpy as np
import torch
import yaml

from reproduction.common.data_policy import assert_data_usage, is_data_usage_allowed
from reproduction.common.ids import make_sample_id
from reproduction.common.paths import PROJECT_ROOT, project_path
from reproduction.common.protocol import FINAL_HTML_TAG, MERGE_TAG, ROW_COL_TAG, format_sequential_target
from reproduction.common.reproducibility import DEFAULT_SEED, seed_everything


class ConfigContractTests(unittest.TestCase):
    def test_base_config_matches_code_contracts(self):
        with project_path("reproduction/configs/base.yaml").open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(config["project"]["seed"], DEFAULT_SEED)
        self.assertEqual(config["protocol"]["row_col_tag"], ROW_COL_TAG)
        self.assertEqual(config["protocol"]["merge_tag"], MERGE_TAG)
        self.assertEqual(config["protocol"]["final_html_tag"], FINAL_HTML_TAG)
        self.assertNotIn("pretrain", config["data_policy"]["tableverse"]["allowed_uses"])


class ReproducibilityTests(unittest.TestCase):
    def test_default_seed_is_42_and_reproducible(self):
        self.assertEqual(DEFAULT_SEED, 42)
        seed_everything()
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        seed_everything()
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)


class PathTests(unittest.TestCase):
    def test_project_paths_are_root_relative(self):
        self.assertEqual(project_path("assets"), (PROJECT_ROOT / "assets").resolve())
        self.assertTrue(project_path("StrucTab").is_dir())


class SampleIdTests(unittest.TestCase):
    def test_id_is_stable(self):
        args = ("tableverse", 12, "images/a.jpg", "<table></table>")
        self.assertEqual(make_sample_id(*args), make_sample_id(*args))

    def test_same_image_with_different_reference_has_different_id(self):
        first = make_sample_id("tableverse", 12, "images/a.jpg", "<table><td>A</td></table>")
        second = make_sample_id("tableverse", 13, "images/a.jpg", "<table><td>B</td></table>")
        self.assertNotEqual(first, second)

    def test_path_separators_are_normalized(self):
        unix_id = make_sample_id("tableverse", 1, "images/a.jpg", "ref")
        windows_id = make_sample_id("tableverse", 1, r"images\a.jpg", "ref")
        self.assertEqual(unix_id, windows_id)


class DataPolicyTests(unittest.TestCase):
    def test_expected_uses(self):
        self.assertTrue(is_data_usage_allowed("synthtabnet", "pretrain"))
        self.assertTrue(is_data_usage_allowed("pubtables", "sft"))
        self.assertTrue(is_data_usage_allowed("tableverse", "evaluation"))

    def test_tableverse_training_is_rejected(self):
        for usage in ("pretrain", "sft", "rl"):
            with self.subTest(usage=usage):
                with self.assertRaises(ValueError):
                    assert_data_usage("tableverse", usage)


class ProtocolTests(unittest.TestCase):
    def test_target_uses_official_markers_in_order(self):
        target = format_sequential_target(2, 3, "未发现合并单元格。", "<table></table>")
        positions = [target.index(tag) for tag in (ROW_COL_TAG, MERGE_TAG, FINAL_HTML_TAG)]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("<STRUCTURE>", target)
        self.assertNotIn("<HTML>", target)


if __name__ == "__main__":
    unittest.main()
