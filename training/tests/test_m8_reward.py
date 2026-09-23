from __future__ import annotations

import numpy as np
from cutagent_training.m8_reward import (
    M8LinearRewardModel,
    RMTrainingBatch,
    evaluate_reward_model,
)


def _batch() -> RMTrainingBatch:
    preferred = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    rejected = np.asarray([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    return RMTrainingBatch(
        candidate_a=preferred,
        candidate_b=rejected,
        preferred_a=np.asarray([True, True, True, True]),
        failure_a=np.asarray([0, 0, 0, 0]),
        failure_b=np.asarray([1, 1, 1, 1]),
    )


def test_joint_training_decreases_both_losses_and_reloads(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    batch = _batch()
    model = M8LinearRewardModel(2, 2, seed=7)
    before = model.losses(batch)
    history = model.fit(
        batch,
        steps=200,
        learning_rate=0.03,
        lambda_rank=1.0,
        lambda_failure=1.0,
        weight_decay=0.0,
        seed=7,
        mini_batch_size=4,
    )
    after = model.losses(batch)
    assert after[0] < before[0]
    assert after[1] < before[1]
    assert all(np.isfinite(item["combined_loss"]) for item in history)
    path = root / "reward.npz"
    model.save(path)
    reloaded = M8LinearRewardModel.load(path)
    np.testing.assert_array_equal(
        model.scores(batch.candidate_a), reloaded.scores(batch.candidate_a)
    )


def test_metrics_include_ranking_failure_calibration_and_length_audit() -> None:
    batch = _batch()
    model = M8LinearRewardModel(2, 2, seed=11)
    model.fit(
        batch,
        steps=200,
        learning_rate=0.03,
        lambda_rank=1.0,
        lambda_failure=1.0,
        weight_decay=0.0,
        seed=11,
        mini_batch_size=4,
    )
    metrics = evaluate_reward_model(
        model,
        batch,
        candidate_lengths=np.asarray([10, 11, 12, 13, 9, 8, 7, 6], dtype=np.float64),
        failure_class_names=("none", "failure"),
        preference_reasons=("failure",) * 4,
    )
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["failure_macro_f1"] == 1.0
    assert 0 <= metrics["ece"] <= 1
    assert "reward_length_pearson_correlation" in metrics
