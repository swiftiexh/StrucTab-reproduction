"""Single-image HunyuanOCR baseline inference.

This entry point intentionally uses the model's native Hugging Face processor
and chat template. It does not use StrucTab's Qwen-style ``local_vllm`` prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, HunYuanVLForConditionalGeneration

from reproduction.common.paths import project_path
from reproduction.common.reproducibility import DEFAULT_SEED, seed_everything


DEFAULT_MODEL_PATH = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_PROMPT = "把图中的表格解析为HTML。"
DEFAULT_OUTPUT_PATH = project_path("reproduction/results/baseline/single_sample.json")
DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"


def extract_table(text: str) -> str:
    """Extract the first complete HTML table while preserving the raw output."""
    match = re.search(r"<table\b[^>]*>[\s\S]*?</table>", text or "", flags=re.IGNORECASE)
    return match.group(0).strip() if match else ""


def diagnose_generation(text: str, output_tokens: int, max_new_tokens: int) -> dict:
    """Return conservative, inspectable truncation and repetition signals."""
    table_open_count = len(re.findall(r"<table\b[^>]*>", text or "", flags=re.IGNORECASE))
    table_close_count = len(re.findall(r"</table>", text or "", flags=re.IGNORECASE))
    hit_max_new_tokens = output_tokens >= max_new_tokens
    return {
        "hit_max_new_tokens": hit_max_new_tokens,
        "table_open_count": table_open_count,
        "table_close_count": table_close_count,
        "suspected_truncation": hit_max_new_tokens or table_open_count > table_close_count,
        "suspected_repetition": table_open_count > 1 or table_close_count > 1,
    }


def build_messages(image_path: Path, prompt: str) -> list[dict]:
    """Build the official HunyuanOCR multimodal chat message."""
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HunyuanOCR on one table image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager"),
        default=DEFAULT_ATTENTION_IMPLEMENTATION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class HunyuanBaselineEngine:
    """One loaded HunyuanOCR model reusable across baseline samples."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype_name: str,
        seed: int,
        attention_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION,
    ):
        self.model_path = model_path.resolve()
        self.device = device
        self.dtype_name = dtype_name
        self.seed = seed
        self.attention_implementation = attention_implementation
        if attention_implementation not in {"sdpa", "eager"}:
            raise ValueError(f"unsupported attention implementation: {attention_implementation}")
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"model directory not found: {self.model_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        seed_everything(seed)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            backend="pil",
        )
        config = AutoConfig.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        # The checkpoint config has -1; tokenizer/generation_config use 120002.
        config.pad_token_id = self.pad_token_id

        load_started = time.perf_counter()
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=config,
            local_files_only=True,
            trust_remote_code=True,
            dtype=_dtype_from_name(dtype_name),
            attn_implementation=attention_implementation,
            device_map={"": device},
            low_cpu_mem_usage=True,
        ).eval()
        self.model.generation_config.pad_token_id = self.pad_token_id
        self.load_seconds = time.perf_counter() - load_started

    def infer(self, image_path: Path, prompt: str, max_new_tokens: int) -> dict:
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        image = Image.open(image_path).convert("RGB")
        image_size = list(image.size)
        messages = build_messages(image_path, prompt)
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat_prompt], images=image, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        input_ids = inputs.get("input_ids", inputs.get("inputs"))
        input_token_count = int(input_ids.shape[-1]) if input_ids is not None else None

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            model_memory_gib = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        else:
            model_memory_gib = None

        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.03,
                pad_token_id=self.pad_token_id,
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started

        if input_ids is not None:
            generated_only = generated_ids[:, input_ids.shape[-1] :]
        else:
            generated_only = generated_ids
        raw_output = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        peak_generation_memory_gib = (
            torch.cuda.max_memory_allocated() / 1024**3
            if self.device.startswith("cuda")
            else None
        )
        output_token_count = int(generated_only.shape[-1])
        pred_html = extract_table(raw_output)
        return {
            "image_size": image_size,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "generation_seconds": round(generation_seconds, 4),
            "model_memory_gib": round(model_memory_gib, 4) if model_memory_gib is not None else None,
            "peak_generation_memory_gib": (
                round(peak_generation_memory_gib, 4)
                if peak_generation_memory_gib is not None
                else None
            ),
            "raw_output": raw_output,
            "pred_html": pred_html,
            "has_complete_table": bool(pred_html),
            "diagnostics": diagnose_generation(raw_output, output_token_count, max_new_tokens),
        }


def run_inference(args: argparse.Namespace) -> dict:
    image_path = args.image.resolve()
    output_path = args.output.resolve()
    engine = HunyuanBaselineEngine(
        args.model, args.device, args.dtype, args.seed, args.attn_implementation
    )
    inference_result = engine.infer(image_path, args.prompt, args.max_new_tokens)

    result = {
        "sample_id": args.sample_id,
        "image_path": str(image_path),
        "model_path": str(engine.model_path),
        "prompt": args.prompt,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "generation": {
            "do_sample": False,
            "repetition_penalty": 1.03,
            "max_new_tokens": args.max_new_tokens,
            "attention_implementation": args.attn_implementation,
        },
        "load_seconds": round(engine.load_seconds, 4),
        **inference_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    result["output_path"] = str(output_path)
    return result


def main() -> None:
    result = run_inference(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
