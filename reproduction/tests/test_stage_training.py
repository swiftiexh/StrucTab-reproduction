import json
import tempfile
import unittest
from pathlib import Path

from reproduction.data.training_examples import TrainingExample
from reproduction.train.build_stage12_soak import select_base_ids
from reproduction.train.run_stage12_soak import memory_drift
from reproduction.train.stage_trainer import file_sha256, load_stage_examples


class StageTrainerDataTests(unittest.TestCase):
    def test_loader_rejects_wrong_stage(self):
        example = TrainingExample(
            sample_id="stage2-0000000000000001",
            base_sample_id="base-0000000000000001",
            source="test",
            split="train",
            image_path="images/x.png",
            stage="stage2",
            task_type="sequential",
            prompt="prompt",
            target="target",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps(example.to_dict()) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stage2 example in stage1 run"):
                load_stage_examples(path, "stage1")

    def test_sha256_is_streamed_and_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"
            path.write_bytes(b"abc")
            self.assertEqual(
                file_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class SoakSelectionTests(unittest.TestCase):
    def test_selection_covers_strata_then_fills_longest(self):
        rows = []
        for source in ("synthtabnet", "fintabnet"):
            for complexity in ("complex", "merged", "simple"):
                rows.append(
                    {
                        "sample_id": f"{source}-{complexity}",
                        "source": source,
                        "complexity": complexity,
                        "split": "train",
                        "measurements": {"sequential": {"sequence_tokens": 100}},
                    }
                )
        rows.extend(
            {
                "sample_id": f"extra-{index}",
                "source": "synthtabnet",
                "complexity": "complex",
                "split": "train",
                "measurements": {"sequential": {"sequence_tokens": 200 + index}},
            }
            for index in range(3)
        )
        selected = select_base_ids(rows, 8)
        self.assertEqual(len(selected), 8)
        selected_rows = [row for row in rows if row["sample_id"] in selected]
        self.assertEqual(
            {(row["source"], row["complexity"]) for row in selected_rows},
            {
                (source, complexity)
                for source in ("synthtabnet", "fintabnet")
                for complexity in ("complex", "merged", "simple")
            },
        )

    def test_memory_drift_uses_post_cleanup_values(self):
        result = {
            "train": {
                "cleanup_allocated_first_gib": 6.0,
                "cleanup_allocated_last_gib": 6.125,
            }
        }
        self.assertEqual(memory_drift(result), 0.125)


if __name__ == "__main__":
    unittest.main()
