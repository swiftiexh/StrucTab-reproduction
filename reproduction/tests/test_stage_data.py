import importlib.util
import unittest

from reproduction.common.paths import project_path
from reproduction.common.protocol import (
    FINAL_HTML_TAG,
    MERGE_TAG,
    ROW_COL_TAG,
    parse_sequential_output,
)
from reproduction.data.schema import NormalizedTableSample
from reproduction.data.training_examples import (
    STAGE1_TASKS,
    TrainingExample,
    build_stage1_examples,
    build_stage2_example,
    validate_stage2_semantics,
)


def base_sample(html=None):
    return NormalizedTableSample(
        sample_id="synthtabnet-12345678",
        source="synthtabnet",
        split="train",
        image_path="assets/images/a.png",
        html=html
        or (
            '<table><tr><th rowspan="2">A</th><td colspan="2">B</td></tr>'
            "<tr><td>C</td><td>D</td></tr></table>"
        ),
        metadata={"source_key": "train:1"},
    )


class Stage1BuilderTests(unittest.TestCase):
    def test_one_base_produces_three_balanced_independent_tasks(self):
        examples = build_stage1_examples(base_sample())
        self.assertEqual(tuple(example.task_type for example in examples), STAGE1_TASKS)
        self.assertEqual(len({example.sample_id for example in examples}), 3)
        self.assertTrue(all(example.base_sample_id == "synthtabnet-12345678" for example in examples))

    def test_all_three_answers_come_from_the_same_structure(self):
        examples = {example.task_type: example for example in build_stage1_examples(base_sample())}
        self.assertEqual(examples["row_col"].target, "这是一个2行3列的表格。")
        self.assertEqual(
            examples["merge"].target,
            "<merge>(1,1),(2,1)</merge><merge>(1,2),(1,3)</merge>",
        )
        self.assertEqual(examples["html"].target, base_sample().html)

    def test_no_merge_task_uses_official_sentence(self):
        html = "<table><tr><td>A</td></tr></table>"
        examples = {example.task_type: example for example in build_stage1_examples(base_sample(html))}
        self.assertEqual(examples["merge"].target, "未发现合并单元格。")


class SequentialProtocolTests(unittest.TestCase):
    def test_stage2_target_round_trips_without_extra_legacy_tags(self):
        example = build_stage2_example(base_sample())
        parsed = parse_sequential_output(example.target)
        self.assertTrue(parsed["is_valid"], parsed["errors"])
        self.assertEqual((parsed["row_count"], parsed["col_count"]), (2, 3))
        self.assertEqual(parsed["html"], base_sample().html)
        self.assertNotIn("<STRUCTURE>", example.target)
        self.assertNotIn("<HTML>", example.target)

    def test_official_reward_parser_accepts_generated_target(self):
        path = project_path("StrucTab/code/Uni_TabRL/reward/utils/parsing.py")
        spec = importlib.util.spec_from_file_location("official_stage_parser_test", path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = build_stage2_example(base_sample()).target
        self.assertTrue(module.has_all_sections(target))
        sections = module.extract_sections(target)
        self.assertEqual(sections["origin_tab"], base_sample().html)
        self.assertTrue(all(sections.values()))

    def test_parser_rejects_duplicate_missing_or_reordered_sections(self):
        target = build_stage2_example(base_sample()).target
        invalid = (
            target + ROW_COL_TAG,
            target.replace(MERGE_TAG, ""),
            target.replace(ROW_COL_TAG, "TEMP").replace(MERGE_TAG, ROW_COL_TAG).replace("TEMP", MERGE_TAG),
        )
        for value in invalid:
            with self.subTest(value=value[:30]):
                self.assertFalse(parse_sequential_output(value)["is_valid"])

    def test_semantic_validator_detects_wrong_dimensions_and_merge(self):
        target = build_stage2_example(base_sample()).target
        wrong_dimensions = target.replace("这是一个2行3列的表格。", "这是一个3行3列的表格。")
        wrong_merge = target.replace(
            "<merge>(1,1),(2,1)</merge>", "<merge>(1,1),(1,2)</merge>"
        )
        self.assertIn("semantic_row_col_mismatch", validate_stage2_semantics(wrong_dimensions)["errors"])
        self.assertIn("semantic_merge_mismatch", validate_stage2_semantics(wrong_merge)["errors"])

    def test_training_example_schema_rejects_stage_task_mismatch(self):
        example = build_stage2_example(base_sample())
        values = example.to_dict()
        values["task_type"] = "html"
        with self.assertRaisesRegex(ValueError, "invalid stage/task"):
            TrainingExample.from_dict(values)

    def test_section_markers_appear_in_official_order(self):
        target = build_stage2_example(base_sample()).target
        self.assertLess(target.index(ROW_COL_TAG), target.index(MERGE_TAG))
        self.assertLess(target.index(MERGE_TAG), target.index(FINAL_HTML_TAG))


if __name__ == "__main__":
    unittest.main()
