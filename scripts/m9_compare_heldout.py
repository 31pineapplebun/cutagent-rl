"""Compare frozen M6 initialization and an M9 GRPO adapter on identical holdout states."""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, cast

from cutagent_training.m9_contracts import M9EnvironmentInput, M9EnvironmentLabel
from cutagent_training.m9_environment import (
    ControlledAgentEnvironment,
    parsed_action_dict,
)

torch = importlib.import_module("torch")
peft_module = importlib.import_module("peft")
transformers_module = importlib.import_module("transformers")


def _read(path: Path, model: type[Any]) -> list[Any]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return ordered[index]


def _run_policy(
    *,
    policy_name: str,
    model_path: Path,
    adapter_path: Path | None,
    public_rows: list[M9EnvironmentInput],
    labels: dict[str, M9EnvironmentLabel],
    output: Path,
    max_new_tokens: int,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor = transformers_module.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=False,
        padding_side="left",
        truncation_side="left",
    )
    model = transformers_module.AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map="cuda:0",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = peft_module.PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    public_predictions: list[dict[str, object]] = []
    private_scores: list[dict[str, object]] = []
    latencies: list[float] = []
    lengths: list[int] = []
    started = time.perf_counter()
    for index, public_input in enumerate(public_rows):
        messages = [{"role": "user", "content": public_input.prompt}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt_text], return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        torch.manual_seed(20260824 + index)
        begin = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        latency_ms = (time.perf_counter() - begin) * 1000
        input_length = inputs["input_ids"].shape[1]
        raw = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0]
        environment = ControlledAgentEnvironment(labels)
        environment.reset(public_input)
        observation, breakdown, exact, valid = environment.step(raw)
        latencies.append(latency_ms)
        lengths.append(len(raw))
        public_predictions.append(
            {
                "sample_id": public_input.sample_id,
                "environment_snapshot_sha256": public_input.environment_snapshot_sha256,
                "raw_completion": raw,
                "parsed_action": parsed_action_dict(raw),
                "public_observation": observation,
                "latency_ms": round(latency_ms, 3),
            }
        )
        private_scores.append(
            {
                "sample_id": public_input.sample_id,
                "exact_action_match": exact,
                "structured_valid": valid,
                "total_environment_reward": breakdown.total_environment_reward,
                "reward_breakdown": breakdown.model_dump(mode="json"),
            }
        )
    wall = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "public_predictions.json", public_predictions)
    _write(output / "private_scores.json", private_scores)
    metrics = {
        "schema_version": "1.0",
        "policy": policy_name,
        "case_count": len(public_rows),
        "curriculum_stages": sorted({item.curriculum_stage for item in public_rows}),
        "objective_success": sum(bool(item["exact_action_match"]) for item in private_scores)
        / len(private_scores),
        "structured_validity": sum(bool(item["structured_valid"]) for item in private_scores)
        / len(private_scores),
        "mean_environment_reward": sum(
            cast(float, item["total_environment_reward"]) for item in private_scores
        )
        / len(private_scores),
        "mean_completion_length": statistics.mean(lengths),
        "mean_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "wall_time_seconds": wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "protected_access_count": 0,
    }
    _write(output / "metrics.json", metrics)
    del model, processor
    torch.cuda.empty_cache()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/m9/dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m9/heldout_comparison"))
    parser.add_argument("--maximum-cases", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--curriculum-stage", default="single_decision", choices=("single_decision",)
    )
    args = parser.parse_args()
    root = args.dataset.resolve(strict=True)
    public = cast(
        list[M9EnvironmentInput],
        _read(root / "public" / "holdout.jsonl", M9EnvironmentInput),
    )
    labels_list = cast(
        list[M9EnvironmentLabel],
        _read(root / "private" / "holdout_labels.jsonl", M9EnvironmentLabel),
    )
    labels = {item.sample_id: item for item in labels_list}
    selected = [item for item in public if item.curriculum_stage == args.curriculum_stage]
    selected = selected[: args.maximum_cases]
    if not selected:
        raise ValueError("no matching M9 holdout cases")
    model_path = args.model.resolve(strict=True)
    adapter_path = args.adapter.resolve(strict=True)
    output = args.output.resolve()
    initialization = _run_policy(
        policy_name="frozen_M6_SFT",
        model_path=model_path,
        adapter_path=None,
        public_rows=selected,
        labels=labels,
        output=output / "initialization",
        max_new_tokens=args.max_new_tokens,
    )
    grpo = _run_policy(
        policy_name="M9_GRPO",
        model_path=model_path,
        adapter_path=adapter_path,
        public_rows=selected,
        labels=labels,
        output=output / "grpo",
        max_new_tokens=args.max_new_tokens,
    )
    comparison = {
        "schema_version": "1.0",
        "split": "train_source_disjoint_holdout",
        "same_case_count": len(selected),
        "initialization": initialization,
        "grpo": grpo,
        "objective_success_delta": cast(float, grpo["objective_success"])
        - cast(float, initialization["objective_success"]),
        "structured_validity_delta": cast(float, grpo["structured_validity"])
        - cast(float, initialization["structured_validity"]),
        "protected_access_count": 0,
    }
    _write(output / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
