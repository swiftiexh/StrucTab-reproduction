"""Single-epoch full-parameter trainer shared by budgeted Stage 1 and Stage 2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor

from reproduction.common.paths import project_path
from reproduction.common.reproducibility import DEFAULT_SEED, seed_everything
from reproduction.data.training_examples import TrainingExample
from reproduction.inference.hunyuan_baseline import _dtype_from_name, build_messages
from reproduction.train.hunyuan_feasibility import (
    atomic_json,
    is_cuda_oom,
    load_base_model,
    memory_metrics,
    parameter_counts,
    prepare_example,
)


DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_BUDGET_DIR = project_path("reproduction/data/training/budgeted")
DEFAULT_RESULT_DIR = project_path("reproduction/results/training")
DEFAULT_CHECKPOINT_DIR = project_path("reproduction/checkpoints")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_examples(path: Path, stage: str) -> list[TrainingExample]:
    examples = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                example = TrainingExample.from_dict(json.loads(line))
                if example.stage != stage:
                    raise ValueError(
                        f"{path} contains {example.stage} example in {stage} run: {example.sample_id}"
                    )
                examples.append(example)
    if not examples:
        raise ValueError(f"empty training data: {path}")
    return examples


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(value, ensure_ascii=False) + "\n")
        output_file.flush()


def _model_args(args, model_path: Path):
    return SimpleNamespace(
        strategy="checkpoint",
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attention_implementation,
        pad_token_id=args.pad_token_id,
        model=model_path,
    )


def load_training_model(args, model_path: Path):
    model = load_base_model(_model_args(args, model_path), model_path)
    model.gradient_checkpointing_enable()
    return model


def save_checkpoint(model, processor, checkpoint_path: Path) -> float:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint_path}; choose a new run-name"
        )
    started = time.perf_counter()
    temporary = Path(
        tempfile.mkdtemp(prefix=checkpoint_path.name + ".tmp-", dir=checkpoint_path.parent)
    )
    model.save_pretrained(temporary, safe_serialization=True)
    processor.save_pretrained(temporary)
    temporary.rename(checkpoint_path)
    return round(time.perf_counter() - started, 4)


def reload_and_generate(args, checkpoint_path: Path, example: TrainingExample) -> dict:
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        checkpoint_path, local_files_only=True, trust_remote_code=True, backend="pil"
    )
    args.pad_token_id = processor.tokenizer.pad_token_id
    model = load_base_model(_model_args(args, checkpoint_path), checkpoint_path).eval()
    from PIL import Image

    image_path = project_path(example.image_path)
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    messages = build_messages(image_path, example.prompt)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=image, padding=False, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.reload_new_tokens,
            do_sample=False,
            pad_token_id=args.pad_token_id,
        )
    generated = generated[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {
        "success": True,
        "seconds": round(time.perf_counter() - started, 4),
        "sample_id": example.sample_id,
        "output": output,
    }


def _step(
    model,
    processor,
    optimizer,
    scaler,
    example,
    args,
    training: bool,
) -> dict:
    batch, info = prepare_example(processor, example, args.device, args.max_sequence_tokens)
    if training:
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        with torch.autocast(
            device_type="cuda",
            dtype=_dtype_from_name(args.dtype),
            enabled=args.dtype != "float32",
        ):
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        if training:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    torch.cuda.synchronize()
    record = {
        **info,
        "loss": float(loss.detach().float().item()),
        "seconds": round(time.perf_counter() - started, 4),
        "memory": memory_metrics(),
    }
    if training:
        optimizer.zero_grad(set_to_none=True)
    del batch, outputs, loss
    record["memory_after_cleanup"] = memory_metrics()
    return record


def run(args) -> dict:
    result_path = args.result_dir.resolve() / f"{args.run_name}.json"
    steps_path = args.result_dir.resolve() / f"{args.run_name}_steps.jsonl"
    checkpoint_path = args.checkpoint_dir.resolve() / args.run_name
    if result_path.exists() or steps_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"run-name already has outputs: {args.run_name}")
    train_path = args.train_data.resolve()
    validation_path = args.validation_data.resolve() if args.validation_data else None
    result = {
        "status": "running",
        "run_name": args.run_name,
        "stage": args.stage,
        "config": {
            "model": str(args.model.resolve()),
            "processor": str(args.processor.resolve()),
            "train_data": str(train_path),
            "validation_data": str(validation_path) if validation_path else None,
            "train_data_sha256": file_sha256(train_path),
            "epochs": 1,
            "learning_rate": args.learning_rate,
            "batch_size": 1,
            "full_parameter": True,
            "gradient_checkpointing": True,
            "dtype": args.dtype,
            "attention_implementation": args.attention_implementation,
            "max_sequence_tokens": args.max_sequence_tokens,
            "seed": args.seed,
        },
    }
    atomic_json(result_path, result)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage trainer requires CUDA")
        seed_everything(args.seed)
        train_examples = load_stage_examples(train_path, args.stage)
        validation_examples = (
            load_stage_examples(validation_path, args.stage) if validation_path else []
        )
        random.Random(args.seed).shuffle(train_examples)
        if args.max_train_steps:
            train_examples = train_examples[: args.max_train_steps]
        if args.validation_steps >= 0:
            validation_examples = validation_examples[: args.validation_steps]
        processor = AutoProcessor.from_pretrained(
            args.processor.resolve(), local_files_only=True, trust_remote_code=True, backend="pil"
        )
        args.pad_token_id = processor.tokenizer.pad_token_id
        loaded = time.perf_counter()
        model = load_training_model(args, args.model.resolve()).train()
        result["model_load_seconds"] = round(time.perf_counter() - loaded, 4)
        result["parameters"] = parameter_counts(model)
        if result["parameters"]["trainable"] != result["parameters"]["total"]:
            raise RuntimeError("formal pretraining must keep every parameter trainable")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
        torch.cuda.reset_peak_memory_stats()
        result["memory_after_load"] = memory_metrics()
        atomic_json(result_path, result)

        train_losses = []
        train_seconds = []
        cleanup_allocated = []
        for index, example in enumerate(train_examples, 1):
            result["active_step"] = {"step": index, "sample_id": example.sample_id}
            atomic_json(result_path, result)
            record = _step(model, processor, optimizer, scaler, example, args, training=True)
            record.update({"split": "train", "step": index})
            append_jsonl(steps_path, record)
            train_losses.append(record["loss"])
            train_seconds.append(record["seconds"])
            cleanup_allocated.append(record["memory_after_cleanup"]["allocated_gib"])
            result.pop("active_step", None)
            result["completed_train_steps"] = index
            if index == 1 or index % args.log_every == 0 or index == len(train_examples):
                result["latest_loss"] = record["loss"]
                atomic_json(result_path, result)
                print(
                    f"[{args.stage} train {index}/{len(train_examples)}] "
                    f"loss={record['loss']:.6f} seconds={record['seconds']:.3f} "
                    f"eta_hours={sum(train_seconds)/len(train_seconds)*(len(train_examples)-index)/3600:.3f}",
                    flush=True,
                )

        model.eval()
        validation_losses = []
        for index, example in enumerate(validation_examples, 1):
            record = _step(model, processor, optimizer, scaler, example, args, training=False)
            record.update({"split": "validation", "step": index})
            append_jsonl(steps_path, record)
            validation_losses.append(record["loss"])
        result["train"] = {
            "steps": len(train_losses),
            "loss_mean": sum(train_losses) / len(train_losses),
            "loss_min": min(train_losses),
            "loss_max": max(train_losses),
            "seconds": sum(train_seconds),
            "mean_step_seconds": sum(train_seconds) / len(train_seconds),
            "cleanup_allocated_first_gib": cleanup_allocated[0],
            "cleanup_allocated_last_gib": cleanup_allocated[-1],
        }
        result["validation"] = {
            "steps": len(validation_losses),
            "loss_mean": (
                sum(validation_losses) / len(validation_losses) if validation_losses else None
            ),
        }
        result["losses_finite"] = all(
            math.isfinite(value) for value in [*train_losses, *validation_losses]
        )
        del optimizer
        result["checkpoint_save_seconds"] = save_checkpoint(model, processor, checkpoint_path)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        result["reload"] = reload_and_generate(
            args, checkpoint_path, validation_examples[0] if validation_examples else train_examples[0]
        )
        result["checkpoint_path"] = str(checkpoint_path)
        result["status"] = "ok"
    except Exception as exc:
        if "active_step" in result:
            result["failed_active_step"] = result["active_step"]
        result["status"] = "oom" if is_cuda_oom(exc) else "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result.pop("active_step", None)
    result["completed_at_unix"] = time.time()
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--processor", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--validation-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--max-sequence-tokens", type=int, default=1536)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reload-new-tokens", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.train_data is None:
        args.train_data = DEFAULT_BUDGET_DIR / f"{args.stage}_train.jsonl"
    if args.validation_data is None:
        args.validation_data = DEFAULT_BUDGET_DIR / f"{args.stage}_validation.jsonl"
    if args.max_train_steps < 0 or args.validation_steps < -1:
        parser.error("invalid step limit")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 2 if result["status"] == "oom" else 1)


if __name__ == "__main__":
    main()
