"""Run validation-only policy/rule/RM/hybrid candidate selection for M8."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
from cutagent_training.m8_contracts import (
    FAILURE_CLASSES,
    M8AgentCandidate,
    M8AgentCandidateInput,
    M8AgentCandidateLabel,
)
from cutagent_training.m8_reward import M8LinearRewardModel

from cutagent.schemas.agent import POLICY_DECISION_ADAPTER, ToolDecision


def _rows(path: Path, model: Any) -> tuple[Any, ...]:
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _public_rule_score(case: M8AgentCandidateInput, candidate: M8AgentCandidate) -> float:
    try:
        decision = POLICY_DECISION_ADAPTER.validate_python(candidate.decision)
    except ValueError:
        return -4.0
    if not isinstance(decision, ToolDecision):
        return -2.0
    if decision.tool_call.tool_name != "search_video":
        return -1.0
    arguments = decision.tool_call.arguments.model_dump(mode="json")
    query_terms = {item for item in str(arguments.get("query", "")).casefold().split() if item}
    task = cast(dict[str, Any], case.observable_context["task"])
    instruction_terms = {
        item for item in str(task["instruction"]).casefold().split() if len(item) >= 4
    }
    overlap = len(query_terms & instruction_terms) / max(len(instruction_terms), 1)
    return 2.0 + overlap


def _embed(
    cases: tuple[M8AgentCandidateInput, ...],
    *,
    model_path: Path,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, list[tuple[str, str]], dict[str, int]]:
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    torch.cuda.reset_peak_memory_stats()
    processor = transformers.AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    encoder = transformers.AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True,
    )
    encoder.eval()
    identities: list[tuple[str, str]] = []
    texts: list[str] = []
    for case in cases:
        for candidate in case.candidates:
            identities.append((case.case_id, candidate.candidate_id))
            texts.append(case.render_candidate(candidate))
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
            output = encoder(
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
    features = np.concatenate(chunks).astype(np.float64)
    performance = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "candidate_count": len(features),
    }
    del encoder
    torch.cuda.empty_cache()
    return features, identities, performance


def _outcome(selected: bool, baseline: bool) -> str:
    if selected and not baseline:
        return "corrected"
    if baseline and not selected:
        return "harmed"
    return "no_effect"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m8/agent_experiment"))
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cases = cast(
        tuple[M8AgentCandidateInput, ...],
        _rows(args.candidate_root / "public" / "inputs.jsonl", M8AgentCandidateInput),
    )
    started = time.perf_counter_ns()
    features, identities, gpu = _embed(
        cases,
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    reward_model = M8LinearRewardModel.load(args.checkpoint)
    scores = reward_model.scores(features)
    failure_logits = reward_model.failure_logits(features)
    predicted_failures = np.argmax(failure_logits, axis=1)
    feature_by_id = {
        identity: (float(scores[index]), FAILURE_CLASSES[int(predicted_failures[index])])
        for index, identity in enumerate(identities)
    }
    public_rows: list[dict[str, Any]] = []
    for case in cases:
        candidates = list(case.candidates)
        rm_scores = np.asarray(
            [feature_by_id[(case.case_id, item.candidate_id)][0] for item in candidates]
        )
        normalized_rm = (rm_scores - np.mean(rm_scores)) / max(float(np.std(rm_scores)), 1e-8)
        rule_scores = np.asarray([_public_rule_score(case, item) for item in candidates])
        policy = next(
            item.candidate_id for item in candidates if item.candidate_id.endswith("-policy")
        )
        rule = candidates[int(np.argmax(rule_scores))].candidate_id
        rm = candidates[int(np.argmax(rm_scores))].candidate_id
        hybrid = candidates[int(np.argmax(rule_scores + 0.25 * normalized_rm))].candidate_id
        public_rows.append(
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "selections": {
                    "policy_top1": policy,
                    "rule_only": rule,
                    "rm": rm,
                    "hybrid": hybrid,
                },
                "candidate_scores": [
                    {
                        "candidate_id": item.candidate_id,
                        "rule_score": float(rule_scores[index]),
                        "rm_score": float(rm_scores[index]),
                        "predicted_failure": feature_by_id[(case.case_id, item.candidate_id)][1],
                    }
                    for index, item in enumerate(candidates)
                ],
            }
        )
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / "public_scores.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in public_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    labels = cast(
        tuple[M8AgentCandidateLabel, ...],
        _rows(args.candidate_root / "private" / "labels.jsonl", M8AgentCandidateLabel),
    )
    label_by_case = {item.case_id: set(item.acceptable_candidate_ids) for item in labels}
    methods = ("policy_top1", "rule_only", "rm", "hybrid")
    successes: dict[str, list[bool]] = {method: [] for method in methods}
    outcome_counts: dict[str, Counter[str]] = {"rm": Counter(), "hybrid": Counter()}
    private_rows: list[dict[str, Any]] = []
    for row in public_rows:
        accepted = label_by_case[row["case_id"]]
        selections = cast(dict[str, str], row["selections"])
        correctness = {method: selections[method] in accepted for method in methods}
        for method in methods:
            successes[method].append(correctness[method])
        for method in ("rm", "hybrid"):
            outcome_counts[method][_outcome(correctness[method], correctness["policy_top1"])] += 1
        private_rows.append(
            {
                "case_id": row["case_id"],
                "correctness": correctness,
                "rm_vs_policy": _outcome(correctness["rm"], correctness["policy_top1"]),
                "hybrid_vs_policy": _outcome(correctness["hybrid"], correctness["policy_top1"]),
            }
        )
    private_root = root / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    with (private_root / "evaluations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    wall_time_ms = (time.perf_counter_ns() - started) // 1_000_000
    summary = {
        "schema_version": "1.0",
        "split": "validation",
        "case_count": len(cases),
        "candidate_count": len(identities),
        "selection_accuracy": {
            method: sum(values) / len(values) for method, values in successes.items()
        },
        "rm_outcomes_vs_policy": {
            key: outcome_counts["rm"][key] for key in ("corrected", "no_effect", "harmed")
        },
        "hybrid_outcomes_vs_policy": {
            key: outcome_counts["hybrid"][key] for key in ("corrected", "no_effect", "harmed")
        },
        "added_rm_latency_ms_per_case": wall_time_ms / len(cases),
        "wall_time_ms": wall_time_ms,
        "gpu": gpu,
        "runtime_gold_access": 0,
        "protected_access_count": 0,
        "task_level_effect": "not_claimed_initial_decision_only",
    }
    (root / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
