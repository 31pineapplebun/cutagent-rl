"""Small deterministic multi-head optimizer and metrics for M8 Reward Model v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _sigmoid(value: FloatArray) -> FloatArray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _softmax(value: FloatArray) -> FloatArray:
    shifted = value - np.max(value, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(exponential / np.sum(exponential, axis=1, keepdims=True), dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RMTrainingBatch:
    candidate_a: FloatArray
    candidate_b: FloatArray
    preferred_a: npt.NDArray[np.bool_]
    failure_a: IntArray
    failure_b: IntArray

    def __post_init__(self) -> None:
        size = self.candidate_a.shape[0]
        if self.candidate_a.shape != self.candidate_b.shape or size == 0:
            raise ValueError("RM batch candidate embedding shapes are invalid")
        if any(
            array.shape != (size,) for array in (self.preferred_a, self.failure_a, self.failure_b)
        ):
            raise ValueError("RM batch labels do not align with candidate embeddings")


class M8LinearRewardModel:
    """Pairwise scalar reward and failure classifier over frozen Qwen features."""

    def __init__(self, hidden_size: int, failure_class_count: int, *, seed: int) -> None:
        if hidden_size <= 0 or failure_class_count <= 1:
            raise ValueError("invalid M8 reward-head dimensions")
        generator = np.random.default_rng(seed)
        self.reward_weight = generator.normal(0.0, 0.002, hidden_size).astype(np.float64)
        self.failure_weight = generator.normal(
            0.0, 0.002, (hidden_size, failure_class_count)
        ).astype(np.float64)
        self.failure_bias = np.zeros(failure_class_count, dtype=np.float64)

    @property
    def hidden_size(self) -> int:
        return int(self.reward_weight.shape[0])

    @property
    def failure_class_count(self) -> int:
        return int(self.failure_bias.shape[0])

    def scores(self, features: FloatArray) -> FloatArray:
        return features @ self.reward_weight

    def failure_logits(self, features: FloatArray) -> FloatArray:
        return features @ self.failure_weight + self.failure_bias

    def losses(self, batch: RMTrainingBatch) -> tuple[float, float, float]:
        score_a = self.scores(batch.candidate_a)
        score_b = self.scores(batch.candidate_b)
        signed_margin = np.where(batch.preferred_a, score_a - score_b, score_b - score_a)
        rank_loss = float(np.mean(np.logaddexp(0.0, -signed_margin)))
        features = np.concatenate((batch.candidate_a, batch.candidate_b), axis=0)
        targets = np.concatenate((batch.failure_a, batch.failure_b), axis=0)
        probabilities = _softmax(self.failure_logits(features))
        failure_loss = float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)))
        )
        return rank_loss, failure_loss, rank_loss + failure_loss

    def fit(
        self,
        batch: RMTrainingBatch,
        *,
        steps: int,
        learning_rate: float,
        lambda_rank: float,
        lambda_failure: float,
        weight_decay: float,
        seed: int,
        mini_batch_size: int,
    ) -> list[dict[str, float | int]]:
        if steps <= 0 or learning_rate <= 0 or mini_batch_size <= 0:
            raise ValueError("invalid M8 optimizer configuration")
        parameters = (self.reward_weight, self.failure_weight, self.failure_bias)
        first_moments = tuple(np.zeros_like(item) for item in parameters)
        second_moments = tuple(np.zeros_like(item) for item in parameters)
        generator = np.random.default_rng(seed)
        history: list[dict[str, float | int]] = []
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        total = batch.candidate_a.shape[0]
        for step in range(1, steps + 1):
            indices = generator.choice(total, size=min(mini_batch_size, total), replace=False)
            a = batch.candidate_a[indices]
            b = batch.candidate_b[indices]
            preferred_a = batch.preferred_a[indices]
            score_a = self.scores(a)
            score_b = self.scores(b)
            signed_delta = np.where(preferred_a[:, None], a - b, b - a)
            margin = np.where(preferred_a, score_a - score_b, score_b - score_a)
            rank_loss = float(np.mean(np.logaddexp(0.0, -margin)))
            margin_gradient = (_sigmoid(margin) - 1.0) / len(indices)
            reward_gradient = (
                lambda_rank * np.sum(signed_delta * margin_gradient[:, None], axis=0)
                + weight_decay * self.reward_weight
            )

            features = np.concatenate((a, b), axis=0)
            targets = np.concatenate((batch.failure_a[indices], batch.failure_b[indices]), axis=0)
            probabilities = _softmax(self.failure_logits(features))
            failure_loss = float(
                -np.mean(
                    np.log(
                        np.clip(
                            probabilities[np.arange(len(targets)), targets],
                            1e-12,
                            1.0,
                        )
                    )
                )
            )
            probabilities[np.arange(len(targets)), targets] -= 1.0
            probabilities /= len(targets)
            failure_weight_gradient = (
                lambda_failure * features.T @ probabilities + weight_decay * self.failure_weight
            )
            failure_bias_gradient = lambda_failure * np.sum(probabilities, axis=0)
            gradients = (reward_gradient, failure_weight_gradient, failure_bias_gradient)
            for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
                first_moments[index][...] = beta1 * first_moments[index] + (1 - beta1) * gradient
                second_moments[index][...] = beta2 * second_moments[index] + (1 - beta2) * (
                    gradient * gradient
                )
                first_hat = first_moments[index] / (1 - beta1**step)
                second_hat = second_moments[index] / (1 - beta2**step)
                parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
            history.append(
                {
                    "step": step,
                    "rank_loss": rank_loss,
                    "failure_loss": failure_loss,
                    "combined_loss": lambda_rank * rank_loss + lambda_failure * failure_loss,
                    "gradient_norm": float(
                        np.sqrt(sum(float(np.sum(item * item)) for item in gradients))
                    ),
                }
            )
        return history

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            reward_weight=self.reward_weight,
            failure_weight=self.failure_weight,
            failure_bias=self.failure_bias,
        )
        return _sha256(path)

    @classmethod
    def load(cls, path: Path) -> M8LinearRewardModel:
        with np.load(path, allow_pickle=False) as payload:
            reward = payload["reward_weight"].astype(np.float64)
            failure = payload["failure_weight"].astype(np.float64)
            bias = payload["failure_bias"].astype(np.float64)
        model = cls(len(reward), len(bias), seed=0)
        model.reward_weight[...] = reward
        model.failure_weight[...] = failure
        model.failure_bias[...] = bias
        return model


def _rankdata(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def binary_auc(labels: npt.NDArray[np.bool_], scores: FloatArray) -> float:
    positives = int(np.sum(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float(
        (np.sum(ranks[labels]) - positives * (positives + 1) / 2) / (positives * negatives)
    )


def spearman(labels: FloatArray, scores: FloatArray) -> float:
    left, right = _rankdata(labels), _rankdata(scores)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def kendall_tau(labels: FloatArray, scores: FloatArray) -> float:
    concordant = discordant = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            product = (labels[left] - labels[right]) * (scores[left] - scores[right])
            concordant += product > 0
            discordant += product < 0
    denominator = concordant + discordant
    return (concordant - discordant) / denominator if denominator else 0.0


def macro_f1(
    targets: IntArray, predictions: IntArray, class_count: int
) -> tuple[float, list[float]]:
    values: list[float] = []
    for label in range(class_count):
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(values)), values


def expected_calibration_error(
    probabilities: FloatArray, labels: npt.NDArray[np.bool_], *, bins: int = 10
) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probabilities >= low) & (
            (probabilities <= high) if index == bins - 1 else (probabilities < high)
        )
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected])) - float(np.mean(labels[selected]))
            )
    return result


def evaluate_reward_model(
    model: M8LinearRewardModel,
    batch: RMTrainingBatch,
    *,
    candidate_lengths: FloatArray,
    failure_class_names: tuple[str, ...],
    preference_reasons: tuple[str, ...],
) -> dict[str, Any]:
    score_a, score_b = model.scores(batch.candidate_a), model.scores(batch.candidate_b)
    preferred_scores = np.where(batch.preferred_a, score_a, score_b)
    rejected_scores = np.where(batch.preferred_a, score_b, score_a)
    margin = preferred_scores - rejected_scores
    labels_a = batch.preferred_a
    probability_a = _sigmoid(score_a - score_b)
    pair_predictions = score_a > score_b
    features = np.concatenate((batch.candidate_a, batch.candidate_b), axis=0)
    targets = np.concatenate((batch.failure_a, batch.failure_b), axis=0)
    predictions = np.argmax(model.failure_logits(features), axis=1).astype(np.int64)
    overall_f1, class_f1 = macro_f1(targets, predictions, model.failure_class_count)
    reward_scores = np.concatenate((score_a, score_b))
    length_correlation = (
        float(np.corrcoef(candidate_lengths, reward_scores)[0, 1])
        if np.std(candidate_lengths) and np.std(reward_scores)
        else 0.0
    )
    by_reason: dict[str, dict[str, float | int]] = {}
    for reason in sorted(set(preference_reasons)):
        selected = np.asarray([item == reason for item in preference_reasons])
        by_reason[reason] = {
            "count": int(np.sum(selected)),
            "pairwise_accuracy": float(np.mean(margin[selected] > 0)),
            "mean_margin": float(np.mean(margin[selected])),
        }
    return {
        "pair_count": len(margin),
        "pairwise_accuracy": float(np.mean(margin > 0)),
        "roc_auc": binary_auc(labels_a, score_a - score_b),
        "spearman": spearman(labels_a.astype(np.float64), score_a - score_b),
        "kendall_tau": kendall_tau(labels_a.astype(np.float64), score_a - score_b),
        "failure_macro_f1": overall_f1,
        "failure_per_class_f1": dict(zip(failure_class_names, class_f1, strict=True)),
        "ece": expected_calibration_error(probability_a, labels_a),
        "score_distribution": {
            "mean": float(np.mean(reward_scores)),
            "standard_deviation": float(np.std(reward_scores)),
            "minimum": float(np.min(reward_scores)),
            "maximum": float(np.max(reward_scores)),
            "mean_preferred_margin": float(np.mean(margin)),
        },
        "reward_length_pearson_correlation": length_correlation,
        "preferred_side_accuracy": float(np.mean(pair_predictions == labels_a)),
        "order_swap_consistency": 1.0,
        "by_preference_reason": by_reason,
    }


def canonical_metrics_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
