"""Minimal, failure-recording HunyuanOCR training feasibility probe."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    HunYuanVLForConditionalGeneration,
)

from reproduction.common.paths import project_path
from reproduction.common.reproducibility import DEFAULT_SEED, seed_everything
from reproduction.data.training_examples import TrainingExample
from reproduction.inference.hunyuan_baseline import _dtype_from_name, build_messages


DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_DATA = project_path("reproduction/data/training/feasibility_probe.jsonl")
DEFAULT_RESULT_DIR = project_path("reproduction/results/feasibility")
DEFAULT_CHECKPOINT_DIR = project_path("reproduction/checkpoints/feasibility")
STRATEGIES = ("full", "checkpoint", "freeze_vision", "lora", "qlora")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def build_training_messages(image_path: Path, prompt: str, target: str) -> tuple[list[dict], list[dict]]:
    prompt_messages = build_messages(image_path, prompt)
    full_messages = [*prompt_messages, {"role": "assistant", "content": target}]
    return prompt_messages, full_messages


def mask_prompt_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("feasibility probe currently requires batch size 1")
    if not 0 < prompt_length < input_ids.shape[1]:
        raise ValueError(
            f"prompt length must leave at least one supervised token: {prompt_length}/{input_ids.shape[1]}"
        )
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    labels[attention_mask == 0] = -100
    return labels


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or (
        "cuda" in text and ("out of memory" in text or "memory allocation" in text)
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parameter_counts(model) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def memory_metrics() -> dict[str, float]:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 4),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 4),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
    }


def load_examples(
    path: Path, max_samples: int, sample_id: str | None = None
) -> list[TrainingExample]:
    examples = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                example = TrainingExample.from_dict(json.loads(line))
                if sample_id is None or example.sample_id == sample_id:
                    examples.append(example)
            if sample_id is None and len(examples) == max_samples:
                break
    if sample_id is not None and len(examples) != 1:
        raise ValueError(
            f"sample-id must match exactly one example: {sample_id!r} matched {len(examples)}"
        )
    if not examples:
        raise ValueError("probe dataset is empty")
    return examples


def _base_model_kwargs(args, config) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "device_map": {"": args.device},
        "low_cpu_mem_usage": True,
    }
    if args.strategy == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_dtype_from_name(args.dtype),
        )
    else:
        kwargs["dtype"] = _dtype_from_name(args.dtype)
    return kwargs


def load_base_model(args, model_path: Path):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    config.pad_token_id = args.pad_token_id
    config.use_cache = False
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        model_path, **_base_model_kwargs(args, config)
    )
    model.generation_config.pad_token_id = args.pad_token_id
    model.config.use_cache = False
    return model


def load_and_configure_model(args, model_path: Path):
    model = load_base_model(args, model_path)

    if args.strategy in {"checkpoint", "freeze_vision", "lora", "qlora"}:
        model.gradient_checkpointing_enable()
    if args.strategy == "freeze_vision":
        for parameter in model.model.vision_tower.parameters():
            parameter.requires_grad = False
    if args.strategy in {"lora", "qlora"}:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("LoRA/QLoRA requires `python -m pip install peft`") from exc
        if args.strategy == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGETS),
        )
        model = get_peft_model(model, lora_config)
    return model


def prepare_example(processor, example: TrainingExample, device: str, max_sequence_tokens: int):
    image_path = project_path(example.image_path)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    prompt_messages, full_messages = build_training_messages(
        image_path, example.prompt, example.target
    )
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=False
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_inputs = processor(
        text=[prompt_text], images=image, padding=False, return_tensors="pt"
    )
    full_inputs = processor(text=[full_text], images=image, padding=False, return_tensors="pt")
    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    sequence_length = int(full_inputs["input_ids"].shape[1])
    if not torch.equal(
        full_inputs["input_ids"][:, :prompt_length], prompt_inputs["input_ids"]
    ):
        raise ValueError("full training sequence does not preserve the prompt token prefix")
    if sequence_length > max_sequence_tokens:
        raise ValueError(
            f"sequence_too_long: {sequence_length}>{max_sequence_tokens}; truncation is forbidden"
        )
    labels = mask_prompt_labels(
        full_inputs["input_ids"], full_inputs["attention_mask"], prompt_length
    )
    full_inputs["labels"] = labels
    return full_inputs.to(device), {
        "sample_id": example.sample_id,
        "task_type": example.task_type,
        "image_size": list(image.size),
        "prompt_tokens": prompt_length,
        "sequence_tokens": sequence_length,
        "supervised_tokens": int((labels != -100).sum().item()),
    }


def _save_and_reload(args, model, processor, first_example: TrainingExample, checkpoint_path: Path) -> dict:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    adapter = args.strategy in {"lora", "qlora"}
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if adapter:
        from peft import PeftModel

        base = load_base_model(args, args.model.resolve())
        reloaded = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=False)
    else:
        config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True, trust_remote_code=True)
        config.pad_token_id = args.pad_token_id
        reloaded = HunYuanVLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            config=config,
            local_files_only=True,
            trust_remote_code=True,
            dtype=_dtype_from_name(args.dtype),
            attn_implementation=args.attn_implementation,
            device_map={"": args.device},
            low_cpu_mem_usage=True,
        )
    reloaded.eval()
    image_path = project_path(first_example.image_path)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    messages = build_messages(image_path, first_example.prompt)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=image, padding=False, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        generated = reloaded.generate(
            **inputs, max_new_tokens=args.reload_new_tokens, do_sample=False, pad_token_id=args.pad_token_id
        )
    generated_only = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(
        generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {"checkpoint_path": str(checkpoint_path), "reload_success": True, "reload_output": text}


def run(args) -> dict:
    run_name = args.run_name or args.strategy
    result_path = args.result_dir.resolve() / f"{run_name}.json"
    checkpoint_path = args.checkpoint_dir.resolve() / run_name
    result = {
        "status": "running",
        "run_name": run_name,
        "strategy": args.strategy,
        "config": {
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "device": args.device,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "max_samples": args.max_samples,
            "sample_id": args.sample_id,
            "max_steps": args.max_steps,
            "max_sequence_tokens": args.max_sequence_tokens,
            "learning_rate": args.learning_rate,
            "checkpoint_reload": args.checkpoint_reload,
            "seed": args.seed,
        },
        "steps": [],
    }
    atomic_json(result_path, result)
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Phase 6 probe requires a CUDA device")
        seed_everything(args.seed)
        examples = load_examples(args.data.resolve(), args.max_samples, args.sample_id)
        processor = AutoProcessor.from_pretrained(
            args.model.resolve(), local_files_only=True, trust_remote_code=True, backend="pil"
        )
        args.pad_token_id = processor.tokenizer.pad_token_id
        load_started = time.perf_counter()
        model = load_and_configure_model(args, args.model.resolve()).train()
        result["model_load_seconds"] = round(time.perf_counter() - load_started, 4)
        result["parameters"] = parameter_counts(model)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("strategy has no trainable parameters")
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
        torch.cuda.reset_peak_memory_stats()
        result["memory_after_load"] = memory_metrics()
        atomic_json(result_path, result)

        for step_index in range(args.max_steps):
            example = examples[step_index % len(examples)]
            result["active_step"] = {
                "step": step_index + 1,
                "sample_id": example.sample_id,
                "task_type": example.task_type,
            }
            atomic_json(result_path, result)
            batch, batch_info = prepare_example(
                processor, example, args.device, args.max_sequence_tokens
            )
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            with torch.autocast(
                device_type="cuda", dtype=_dtype_from_name(args.dtype), enabled=args.dtype != "float32"
            ):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {loss}")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            step = {
                "step": step_index + 1,
                **batch_info,
                "loss": float(loss.detach().float().item()),
                "seconds": round(time.perf_counter() - started, 4),
                "memory": memory_metrics(),
            }
            result["steps"].append(step)
            # Do not retain the previous step's graph/logits while evaluating the
            # next variable-length sample. In particular, full-vocabulary logits
            # for long HTML targets occupy hundreds of MiB.
            optimizer.zero_grad(set_to_none=True)
            del batch, outputs, loss
            step["memory_after_cleanup"] = memory_metrics()
            result.pop("active_step", None)
            atomic_json(result_path, result)

        final_allocated = result["steps"][-1]["memory"]["allocated_gib"]
        loaded_allocated = result["memory_after_load"]["allocated_gib"]
        result["allocated_memory_growth_gib"] = round(
            final_allocated - loaded_allocated, 4
        )
        result["losses_finite"] = all(math.isfinite(step["loss"]) for step in result["steps"])
        if args.checkpoint_reload:
            del optimizer
            result["checkpoint"] = _save_and_reload(
                args, model, processor, examples[0], checkpoint_path
            )
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "oom" if is_cuda_oom(exc) else "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["completed_at_unix"] = time.time()
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument(
        "--sample-id",
        default=None,
        help="run repeatedly on one exact probe sample instead of taking the first max-samples",
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--checkpoint-reload", action="store_true")
    parser.add_argument("--reload-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.max_samples < 1 or args.max_steps < 1 or args.max_sequence_tokens < 1:
        parser.error("max-samples, max-steps and max-sequence-tokens must be positive")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name may contain only letters, digits, dot, underscore and hyphen")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}, ensure_ascii=False, indent=2))
    if result["status"] == "oom":
        raise SystemExit(2)
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
