"""Train-only data access must reject every non-train private split."""

import pytest
from cutagent_evaluation.schemas import DatasetSplit, TaskAnnotation
from cutagent_evaluation.split_access import (
    InMemoryAnnotationBackend,
    InMemorySplitRegistry,
    SplitAccessError,
    SplitRegistryEntry,
    TrainAnnotationStore,
)


def annotation(task_id: str, split: DatasetSplit) -> TaskAnnotation:
    return TaskAnnotation(
        annotation_id=f"annotation-{task_id}",
        task_id=task_id,
        source_group_id=f"source-{task_id}",
        proposed_split=split,
        annotator_ids=("annotator-1",),
        ground_truth={"private": True},
    )


def build_store() -> TrainAnnotationStore:
    splits = (
        DatasetSplit.TRAIN,
        DatasetSplit.DEV,
        DatasetSplit.VALIDATION,
        DatasetSplit.LOCKED_TEST,
        DatasetSplit.ADVERSARIAL_TEST,
    )
    annotations = {split.value: annotation(split.value, split) for split in splits}
    registry = InMemorySplitRegistry(
        SplitRegistryEntry(
            task_id=split.value,
            source_group_id=f"source-{split.value}",
            split=split,
        )
        for split in splits
    )
    return TrainAnnotationStore(registry, InMemoryAnnotationBackend(annotations))


def test_train_store_can_only_list_and_read_train() -> None:
    store = build_store()
    assert store.list_task_ids() == ("train",)
    assert store.get("train").proposed_split is DatasetSplit.TRAIN


@pytest.mark.parametrize(
    "task_id",
    ["dev", "validation", "locked_test", "adversarial_test"],
)
def test_train_store_rejects_non_train_private_labels(task_id: str) -> None:
    store = build_store()
    with pytest.raises(SplitAccessError, match="access denied"):
        store.get(task_id)
    assert store.audit_log[-1].task_id == task_id
    assert store.audit_log[-1].allowed is False


def test_split_registry_rejects_source_group_leakage() -> None:
    with pytest.raises(ValueError, match="crosses dataset splits"):
        InMemorySplitRegistry(
            (
                SplitRegistryEntry("task-a", "shared-source", DatasetSplit.TRAIN),
                SplitRegistryEntry("task-b", "shared-source", DatasetSplit.LOCKED_TEST),
            )
        )
