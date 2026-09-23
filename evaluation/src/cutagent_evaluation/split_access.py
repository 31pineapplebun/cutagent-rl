"""Split registry and train-only annotation capability."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cutagent_evaluation.schemas import DatasetSplit, TaskAnnotation


class SplitAccessError(PermissionError):
    """Raised when a train-only reader requests a non-train private label."""


@dataclass(frozen=True)
class SplitRegistryEntry:
    task_id: str
    source_group_id: str
    split: DatasetSplit


class SplitRegistry(Protocol):
    def split_for(self, task_id: str) -> DatasetSplit: ...

    def train_task_ids(self) -> tuple[str, ...]: ...


class AnnotationBackend(Protocol):
    def get_private_annotation(self, task_id: str) -> TaskAnnotation: ...


class InMemorySplitRegistry:
    """Small M0 registry with source-group leakage validation."""

    def __init__(self, entries: Iterable[SplitRegistryEntry]) -> None:
        task_splits: dict[str, DatasetSplit] = {}
        source_splits: dict[str, DatasetSplit] = {}
        for entry in entries:
            if entry.task_id in task_splits:
                raise ValueError(f"duplicate split entry for {entry.task_id!r}")
            previous_split = source_splits.get(entry.source_group_id)
            if previous_split is not None and previous_split is not entry.split:
                raise ValueError(f"source group {entry.source_group_id!r} crosses dataset splits")
            task_splits[entry.task_id] = entry.split
            source_splits[entry.source_group_id] = entry.split
        self._task_splits = task_splits

    def split_for(self, task_id: str) -> DatasetSplit:
        try:
            return self._task_splits[task_id]
        except KeyError as exc:
            raise KeyError(f"task {task_id!r} is absent from the split registry") from exc

    def train_task_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                task_id
                for task_id, split in self._task_splits.items()
                if split is DatasetSplit.TRAIN
            )
        )


class InMemoryAnnotationBackend:
    """Test/bootstrap backend; it is never exposed by TrainAnnotationStore."""

    def __init__(self, annotations: Mapping[str, TaskAnnotation]) -> None:
        self._annotations = dict(annotations)

    def get_private_annotation(self, task_id: str) -> TaskAnnotation:
        try:
            return self._annotations[task_id]
        except KeyError as exc:
            raise KeyError(f"annotation {task_id!r} does not exist") from exc


@dataclass(frozen=True)
class SplitAccessRecord:
    task_id: str
    split: DatasetSplit | None
    allowed: bool
    attempted_at: datetime


class TrainAnnotationStore:
    """Capability-limited view that returns private labels for train only."""

    def __init__(self, registry: SplitRegistry, backend: AnnotationBackend) -> None:
        self._registry = registry
        self._backend = backend
        self._audit: list[SplitAccessRecord] = []

    @property
    def audit_log(self) -> tuple[SplitAccessRecord, ...]:
        return tuple(self._audit)

    def list_task_ids(self) -> tuple[str, ...]:
        return self._registry.train_task_ids()

    def get(self, task_id: str) -> TaskAnnotation:
        try:
            split = self._registry.split_for(task_id)
        except KeyError:
            self._audit.append(
                SplitAccessRecord(
                    task_id=task_id,
                    split=None,
                    allowed=False,
                    attempted_at=datetime.now(UTC),
                )
            )
            raise

        allowed = split is DatasetSplit.TRAIN
        self._audit.append(
            SplitAccessRecord(
                task_id=task_id,
                split=split,
                allowed=allowed,
                attempted_at=datetime.now(UTC),
            )
        )
        if not allowed:
            raise SplitAccessError(
                f"train annotation access denied for {task_id!r} in split {split.value!r}"
            )
        annotation = self._backend.get_private_annotation(task_id)
        if annotation.proposed_split is not DatasetSplit.TRAIN:
            raise SplitAccessError("annotation split disagrees with the train registry")
        return annotation
