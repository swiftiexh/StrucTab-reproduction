import json
import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.train.hunyuan_feasibility import (
    build_training_messages,
    is_cuda_oom,
    load_examples,
    mask_prompt_labels,
    parameter_counts,
)
from reproduction.data.training_examples import TrainingExample
from reproduction.train.run_feasibility_guarded import result_path_for


class BatchConstructionTests(unittest.TestCase):
    def test_training_messages_append_plain_assistant_target(self):
        prompt, full = build_training_messages(Path("/tmp/image.png"), "PROMPT", "TARGET")
        self.assertEqual(full[:-1], prompt)
        self.assertEqual(full[-1], {"role": "assistant", "content": "TARGET"})

    def test_only_assistant_suffix_is_supervised(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention = torch.tensor([[1, 1, 1, 1, 0]])
        labels = mask_prompt_labels(input_ids, attention, prompt_length=3)
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 4, -100]])

    def test_prompt_must_leave_supervised_tokens(self):
        ids = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "leave at least one"):
            mask_prompt_labels(ids, torch.ones_like(ids), prompt_length=2)


class ProbeSafetyTests(unittest.TestCase):
    def test_guard_resolves_named_result_path(self):
        path = result_path_for(
            [
                "--strategy",
                "full",
                "--run-name=stress",
                "--result-dir",
                "/tmp/probe-results",
            ]
        )
        self.assertEqual(path, Path("/tmp/probe-results/stress.json"))

    def test_load_examples_can_select_one_exact_sample(self):
        examples = [
            TrainingExample(
                sample_id=f"sample-{index}",
                base_sample_id=f"base-{index}",
                source="test",
                split="train",
                image_path=f"images/{index}.png",
                stage="stage1",
                task_type="html",
                prompt="prompt",
                target="target",
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text(
                "".join(json.dumps(example.to_dict()) + "\n" for example in examples),
                encoding="utf-8",
            )
            selected = load_examples(path, max_samples=1, sample_id="sample-1")
        self.assertEqual([example.sample_id for example in selected], ["sample-1"])

    def test_load_examples_rejects_missing_sample_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched 0"):
                load_examples(path, max_samples=1, sample_id="missing")

    def test_cuda_oom_classification(self):
        self.assertTrue(is_cuda_oom(torch.OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(is_cuda_oom(RuntimeError("CUDA error: memory allocation failed")))
        self.assertFalse(is_cuda_oom(RuntimeError("ordinary failure")))

    def test_parameter_counts_separate_frozen_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        counts = parameter_counts(model)
        self.assertEqual(counts["total"], 13)
        self.assertEqual(counts["trainable"], 4)


if __name__ == "__main__":
    unittest.main()
