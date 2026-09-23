"""Shared M0 test fixtures."""

from pathlib import Path

import pytest

from cutagent.core.artifacts import ArtifactRef
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import DurationConstraint, TaskInput


@pytest.fixture
def artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id="video-001",
        uri=Path("internal/video.mp4").resolve().as_uri(),
        sha256="a" * 64,
        media_type="video/mp4",
        size_bytes=123,
    )


@pytest.fixture
def task_input(artifact_ref: ArtifactRef) -> TaskInput:
    return TaskInput(
        task_id="task-001",
        video_ref=artifact_ref,
        instruction="Create a short observable-only test output.",
        user_constraints=(DurationConstraint(min_ms=1_000, max_ms=2_000),),
    )


@pytest.fixture
def initial_state(task_input: TaskInput) -> AgentState:
    return AgentState.initial(
        task_input,
        ExecutionBudget(max_steps=4, max_tool_calls=2, max_wall_time_ms=1_000),
    )
