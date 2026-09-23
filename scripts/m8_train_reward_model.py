"""Train and evaluate M8 RM v1 with frozen Qwen3-VL features and two real heads."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from cutagent_training.m8_contracts import (
    FAILURE_CLASSES,
    M8RewardCheckpointManifest,
    M8RMInput,
    M8RMLabel,
)
from cutagent_training.m8_reward import (
    M8LinearRewardModel,
    RMTrainingBatch,
    canonical_metrics_json,
    evaluate_reward_model,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
MODEL_LICENSE: Literal["apache-2.0"] = "apache-2.0"


def _read_rows(path: Path, model: Any) -> tuple[Any, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _ordered(
    inputs: tuple[M8RMInput, ...], labels: tuple[M8RMLabel, ...]
) -> tuple[tuple[M8RMInput, ...], tuple[M8RMLabel, ...]]:
    by_id = {item.input_id: item for item in labels}
    if len(by_id) != len(labels) or set(by_id) != {item.input_id for item in inputs}:
        raise ValueError("M8 RM input/label files are not one-to-one")
    return inputs, tuple(by_id[item.input_id] for item in inputs)


def _seed(torch: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _feature_cache_key(inputs: tuple[M8RMInput, ...], max_length: int) -> str:
    payload = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": max_length,
        "input_ids": [item.input_id for item in inputs],
        "input_payloads": [item.model_dump(mode="json") for item in inputs],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _extract_features(
    inputs: tuple[M8RMInput, ...],
    *,
    model_path: Path,
    cache_path: Path,
    max_length: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cache_key = _feature_cache_key(inputs, max_length)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            stored_key = str(payload["cache_key"].item())
            if stored_key == cache_key:
                return (
                    payload["candidate_a"].astype(np.float64),
                    payload["candidate_b"].astype(np.float64),
                    payload["lengths"].astype(np.float64),
                    {
                        "cache_hit": True,
                        "embedding_latency_ms": 0,
                        "peak_allocated_bytes": 0,
                        "peak_reserved_bytes": 0,
                        "embedding_count": len(inputs) * 2,
                    },
                )
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    _seed(torch, seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter_ns()
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.eval()
    texts_a = [item.render_candidate("a") for item in inputs]
    texts_b = [item.render_candidate("b") for item in inputs]
    texts = texts_a + texts_b
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = output.hidden_states[-1].float()
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            chunks.append(cast(Any, pooled.cpu().numpy()))
    features = np.concatenate(chunks, axis=0).astype(np.float64)
    candidate_a, candidate_b = np.split(features, 2)
    lengths = np.asarray([len(item) for item in texts], dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_key=np.asarray(cache_key),
        candidate_a=candidate_a.astype(np.float32),
        candidate_b=candidate_b.astype(np.float32),
        lengths=lengths,
    )
    performance = {
        "cache_hit": False,
        "embedding_latency_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "embedding_count": len(texts),
    }
    del model
    torch.cuda.empty_cache()
    return candidate_a, candidate_b, lengths, performance


def _batch(
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    labels: tuple[M8RMLabel, ...],
) -> RMTrainingBatch:
    class_index = {name: index for index, name in enumerate(FAILURE_CLASSES)}
    return RMTrainingBatch(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        preferred_a=np.asarray(
            [item.preferred_candidate == "a" for item in labels], dtype=np.bool_
        ),
        failure_a=np.asarray([class_index[item.failure_class_a] for item in labels]),
        failure_b=np.asarray([class_index[item.failure_class_b] for item in labels]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/m8/dataset"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m8/training/full"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--lambda-rank", type=float, default=1.0)
    parser.add_argument("--lambda-failure", type=float, default=1.0)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    fit_inputs, fit_labels = _ordered(
        cast(tuple[M8RMInput, ...], _read_rows(args.dataset_root / "inputs/fit.jsonl", M8RMInput)),
        cast(tuple[M8RMLabel, ...], _read_rows(args.dataset_root / "labels/fit.jsonl", M8RMLabel)),
    )
    holdout_inputs, holdout_labels = _ordered(
        cast(
            tuple[M8RMInput, ...],
            _read_rows(args.dataset_root / "inputs/holdout.jsonl", M8RMInput),
        ),
        cast(
            tuple[M8RMLabel, ...],
            _read_rows(args.dataset_root / "labels/holdout.jsonl", M8RMLabel),
        ),
    )
    if args.pilot:
        fit_inputs, fit_labels = fit_inputs[:8], fit_labels[:8]
        holdout_inputs, holdout_labels = holdout_inputs[:8], holdout_labels[:8]
        args.steps = min(args.steps, 4)
    root = args.output_root
    fit_a, fit_b, _fit_lengths, fit_performance = _extract_features(
        fit_inputs,
        model_path=args.model_path,
        cache_path=root / "cache" / "fit_features.npz",
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    holdout_a, holdout_b, holdout_lengths, holdout_performance = _extract_features(
        holdout_inputs,
        model_path=args.model_path,
        cache_path=root / "cache" / "holdout_features.npz",
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    fit_batch = _batch(fit_a, fit_b, fit_labels)
    holdout_batch = _batch(holdout_a, holdout_b, holdout_labels)
    model = M8LinearRewardModel(fit_a.shape[1], len(FAILURE_CLASSES), seed=args.seed)
    initial_fit_losses = model.losses(fit_batch)
    history = model.fit(
        fit_batch,
        steps=args.steps,
        learning_rate=args.learning_rate,
        lambda_rank=args.lambda_rank,
        lambda_failure=args.lambda_failure,
        weight_decay=1e-4,
        seed=args.seed,
        mini_batch_size=32,
    )
    final_fit_losses = model.losses(fit_batch)
    weights = root / "checkpoint" / "reward_heads.npz"
    weights_sha256 = model.save(weights)
    reloaded = M8LinearRewardModel.load(weights)
    reload_difference = float(np.max(np.abs(model.scores(holdout_a) - reloaded.scores(holdout_a))))
    if reload_difference != 0.0:
        raise RuntimeError("M8 reward checkpoint reload changed deterministic scores")
    manifest_data = json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = M8RewardCheckpointManifest(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        license=MODEL_LICENSE,
        encoder_mode="frozen_structured_text_mean_pool",
        hidden_size=model.hidden_size,
        failure_classes=FAILURE_CLASSES,
        lambda_rank=args.lambda_rank,
        lambda_failure=args.lambda_failure,
        optimizer="numpy-adam",
        optimizer_steps=args.steps,
        seed=args.seed,
        max_length=args.max_length,
        dataset_inputs_sha256=manifest_data["public_inputs_sha256"],
        dataset_labels_sha256=manifest_data["private_labels_sha256"],
        weights_sha256=weights_sha256,
    )
    checkpoint_path = root / "checkpoint" / "manifest.json"
    checkpoint_path.write_text(checkpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
    metrics = evaluate_reward_model(
        reloaded,
        holdout_batch,
        candidate_lengths=holdout_lengths,
        failure_class_names=cast(tuple[str, ...], FAILURE_CLASSES),
        preference_reasons=tuple(
            item.failure_class_b if item.preferred_candidate == "a" else item.failure_class_a
            for item in holdout_labels
        ),
    )
    metrics.update(
        {
            "schema_version": "1.0",
            "split": "train_source_disjoint_holdout",
            "initial_fit_rank_loss": initial_fit_losses[0],
            "initial_fit_failure_loss": initial_fit_losses[1],
            "final_fit_rank_loss": final_fit_losses[0],
            "final_fit_failure_loss": final_fit_losses[1],
            "reload_max_absolute_score_difference": reload_difference,
            "protected_access_count": 0,
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "metrics.json").write_text(canonical_metrics_json(metrics), encoding="utf-8")
    performance = {
        "schema_version": "1.0",
        "fit_embedding": fit_performance,
        "holdout_embedding": holdout_performance,
        "wall_time_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "weights_size_bytes": weights.stat().st_size,
        "model_snapshot_path": str(args.model_path),
    }
    (root / "performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({"checkpoint": checkpoint.model_dump(mode="json"), "metrics": metrics}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
