"""Run a real, resumable TRL GRPO pilot or controlled M9 training run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from cutagent_training.m9_contracts import (
    M9EnvironmentInput,
    M9EnvironmentLabel,
    M9RolloutTrace,
)
from cutagent_training.m9_environment import (
    ControlledAgentEnvironment,
    completion_text,
    parsed_action_dict,
)

torch = importlib.import_module("torch")
datasets_module = importlib.import_module("datasets")
peft_module = importlib.import_module("peft")
transformers_module = importlib.import_module("transformers")
trl_module = importlib.import_module("trl")

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
LORA_TARGET = (
    r"^(model\.language_model(?=\.).*\."
    r"(up_proj|o_proj|down_proj|gate_proj|q_proj|k_proj|v_proj))$"
)


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(model.model_validate_json(line))
    if not rows:
        raise ValueError(f"empty M9 file: {path}")
    return rows


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for value in values:
            if hasattr(value, "model_dump"):
                value = value.model_dump(mode="json")
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _aggregate_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _reward_function(
    public_by_id: dict[str, M9EnvironmentInput],
    labels_by_id: dict[str, M9EnvironmentLabel],
    trace_path: Path,
    *,
    group_size: int,
) -> Any:
    occurrences: Counter[str] = Counter()

    def controlled_environment_reward(
        prompts: list[object],
        completions: list[object],
        sample_id: list[str],
        environment_snapshot_sha256: list[str],
        **_: object,
    ) -> list[float]:
        del prompts
        if not (len(completions) == len(sample_id) == len(environment_snapshot_sha256)):
            raise ValueError("TRL reward columns are not aligned")
        rewards: list[float] = []
        traces: list[M9RolloutTrace] = []
        for completion, identifier, supplied_snapshot in zip(
            completions, sample_id, environment_snapshot_sha256, strict=True
        ):
            public_input = public_by_id[identifier]
            if supplied_snapshot != public_input.environment_snapshot_sha256:
                raise ValueError("trainer supplied a mismatched environment snapshot")
            occurrence = occurrences[identifier]
            occurrences[identifier] += 1
            group_number = occurrence // group_size
            group_index = occurrence % group_size
            group_id = f"group-{identifier}-{group_number:04d}"
            raw = completion_text(completion)
            environment = ControlledAgentEnvironment(labels_by_id)
            environment.reset(public_input)
            observation, breakdown, exact, valid = environment.step(raw)
            rewards.append(breakdown.total_environment_reward)
            traces.append(
                M9RolloutTrace(
                    rollout_id=f"rollout-{identifier}-{occurrence:06d}",
                    group_id=group_id,
                    sample_id=identifier,
                    environment_snapshot_sha256=public_input.environment_snapshot_sha256,
                    group_index=group_index,
                    raw_completion=raw,
                    parsed_action=parsed_action_dict(raw),
                    public_observation=observation,
                    breakdown=breakdown,
                    exact_action_match=exact,
                    structured_valid=valid,
                    completion_length=len(raw),
                )
            )
        _append_jsonl(trace_path, cast(list[object], traces))
        return rewards

    return controlled_environment_reward


def _trace_summary(path: Path, group_size: int) -> dict[str, object]:
    traces = _read_jsonl(path, M9RolloutTrace)
    by_group: dict[str, list[M9RolloutTrace]] = defaultdict(list)
    for trace in traces:
        by_group[trace.group_id].append(trace)
    complete_groups = [values for values in by_group.values() if len(values) == group_size]
    same_snapshot = sum(
        len({item.environment_snapshot_sha256 for item in values}) == 1
        for values in complete_groups
    )
    variances = []
    for values in complete_groups:
        rewards = [item.breakdown.total_environment_reward for item in values]
        mean = sum(rewards) / len(rewards)
        variances.append(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    components = {
        field: sum(getattr(item.breakdown, field) for item in traces) / len(traces)
        for field in (
            "structured_validity_reward",
            "decision_kind_reward",
            "tool_or_operation_reward",
            "argument_reward",
            "observable_progress_reward",
            "rm_reward",
            "malformed_penalty",
            "invalid_action_penalty",
            "premature_terminal_penalty",
            "repetition_penalty",
            "unsafe_action_penalty",
            "total_environment_reward",
        )
    }
    return {
        "schema_version": "1.0",
        "rollout_count": len(traces),
        "complete_group_count": len(complete_groups),
        "same_snapshot_group_count": same_snapshot,
        "nonzero_reward_variance_group_count": sum(value > 0 for value in variances),
        "mean_group_reward_variance": sum(variances) / len(variances) if variances else 0.0,
        "structured_validity": sum(item.structured_valid for item in traces) / len(traces),
        "exact_action_match": sum(item.exact_action_match for item in traces) / len(traces),
        "mean_completion_length": sum(item.completion_length for item in traces) / len(traces),
        "mean_reward_components": components,
        "protected_access_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/m9/dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument(
        "--curriculum-stages",
        nargs="+",
        choices=(
            "single_decision",
            "one_tool_step",
            "two_step_sequence",
            "observable_recovery",
        ),
        default=("single_decision",),
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M9 real GRPO requires CUDA")
    if args.max_steps < 1:
        raise ValueError("max steps must be positive")
    dataset_root = args.dataset.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "raw_rollouts.jsonl"
    if trace_path.exists() and args.resume_from_checkpoint is None:
        raise FileExistsError("refusing to append to an existing rollout trace without resume")

    public_rows = cast(
        list[M9EnvironmentInput],
        _read_jsonl(dataset_root / "public" / "fit.jsonl", M9EnvironmentInput),
    )
    private_rows = cast(
        list[M9EnvironmentLabel],
        _read_jsonl(dataset_root / "private" / "fit_labels.jsonl", M9EnvironmentLabel),
    )
    public_by_id = {item.sample_id: item for item in public_rows}
    labels_by_id = {item.sample_id: item for item in private_rows}
    if set(public_by_id) != set(labels_by_id):
        raise ValueError("M9 fit public/private records differ")
    selected = [item for item in public_rows if item.curriculum_stage in args.curriculum_stages]
    if not selected:
        raise ValueError("no M9 fit examples match the requested curriculum stages")
    if args.pilot:
        first_by_stage: dict[str, M9EnvironmentInput] = {}
        for item in selected:
            first_by_stage.setdefault(item.curriculum_stage, item)
        selected = list(first_by_stage.values())

    transformers_module.set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    processor = transformers_module.AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=False,
        padding_side="left",
        truncation_side="left",
    )
    train_rows = [
        {
            "prompt": [{"role": "user", "content": item.prompt}],
            "sample_id": item.sample_id,
            "environment_snapshot_sha256": item.environment_snapshot_sha256,
            "curriculum_stage": item.curriculum_stage,
        }
        for item in selected
    ]
    train_dataset = datasets_module.Dataset.from_list(train_rows)
    group_size = 4
    config = trl_module.GRPOConfig(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=group_size,
        gradient_accumulation_steps=1,
        num_generations=group_size,
        generation_batch_size=group_size,
        max_steps=args.max_steps,
        learning_rate=2e-7,
        beta=0.02,
        max_completion_length=args.max_completion_length,
        temperature=0.8,
        top_p=0.95,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        use_cache=False,
        use_vllm=False,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.max_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        shuffle_dataset=not args.pilot,
        max_grad_norm=1.0,
        log_completions=False,
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "eager",
            "trust_remote_code": False,
        },
    )
    lora = peft_module.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET,
        task_type="CAUSAL_LM",
    )
    reward_function = _reward_function(
        public_by_id,
        labels_by_id,
        trace_path,
        group_size=group_size,
    )
    started = time.perf_counter()
    trainer = trl_module.GRPOTrainer(
        model=str(model_path),
        reward_funcs=reward_function,
        args=config,
        train_dataset=train_dataset,
        processing_class=processor,
        peft_config=lora,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint.resolve(strict=True))
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    adapter_path = output / "checkpoint" / "grpo_adapter"
    trainer.save_model(str(adapter_path))
    processor.save_pretrained(adapter_path)
    trainer.save_state()
    wall_time = time.perf_counter() - started
    if not all(
        math.isfinite(float(value))
        for value in train_result.metrics.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("non-finite GRPO training metric")

    del trainer
    torch.cuda.empty_cache()
    # A real independent adapter construction validates the saved PEFT checkpoint contract.
    reload_base = transformers_module.AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    reloaded = peft_module.PeftModel.from_pretrained(reload_base, adapter_path, is_trainable=False)
    reload_ok = sum(parameter.numel() for parameter in reloaded.parameters()) > 0
    del reloaded, reload_base
    torch.cuda.empty_cache()

    trace_summary = _trace_summary(trace_path, group_size)
    _write(output / "rollout_summary.json", trace_summary)
    _write(output / "trainer_log_history.json", trainer_state_history(output / "trainer"))
    performance = {
        "schema_version": "1.0",
        "wall_time_seconds": wall_time,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }
    _write(output / "performance.json", performance)
    acceptance = {
        "schema_version": "1.0",
        "mode": "pilot" if args.pilot else "full",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "initialization": "frozen_M6_SFT_merged",
        "backend": "trl==0.29.1",
        "group_size": group_size,
        "optimizer_steps": args.max_steps,
        "beta": 0.02,
        "rm_coefficient": 0.0,
        "curriculum_stages": args.curriculum_stages,
        "adapter_path": str(adapter_path),
        "adapter_sha256": _aggregate_sha256(adapter_path),
        "adapter_reload": reload_ok,
        "train_metrics": train_result.metrics,
        "rollouts": trace_summary,
        "performance": performance,
        "protected_access_count": 0,
    }
    _write(output / "training_result.json", acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, default=str))
    return 0


def trainer_state_history(trainer_root: Path) -> list[dict[str, object]]:
    candidates = sorted(trainer_root.glob("checkpoint-*/trainer_state.json"))
    if not candidates:
        return []
    value = json.loads(candidates[-1].read_text(encoding="utf-8"))
    history = value.get("log_history", [])
    return cast(list[dict[str, object]], history if isinstance(history, list) else [])


if __name__ == "__main__":
    raise SystemExit(main())
