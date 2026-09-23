"""Public/private and host-path isolation for the M3A runtime package."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from cutagent.agent.policy_context import PolicyContextBuilder, PolicyViewSerializer
from cutagent.agent.state_reducer import StateReducer
from cutagent.schemas.state import AgentState, ExecutionBudget
from cutagent.schemas.task_input import TaskInput
from cutagent.tools.events import tool_record_to_event
from tests.tool_fixtures import build_tool_runtime, generate_tool_media


def test_runtime_tools_do_not_import_evaluator_private_modules() -> None:
    tool_root = Path("src/cutagent/tools")
    source = "\n".join(path.read_text(encoding="utf-8") for path in tool_root.glob("*.py"))
    assert "cutagent_evaluation" not in source
    assert "BenchmarkGold" not in source
    assert "TaskAnnotation" not in source


def test_policy_view_of_tool_output_contains_no_paths_hashes_or_private_gold(
    tmp_path: Path,
) -> None:
    source_path = generate_tool_media(tmp_path / "source.mp4", duration_seconds=1)
    runtime = build_tool_runtime(tmp_path / "runtime", source_path)
    record = runtime.registry.execute(
        {
            "tool_name": "trim_video",
            "tool_call_id": "leak-trim",
            "arguments": {
                "input_artifact_id": runtime.source.artifact_id,
                "time_range": {"start_ms": 0, "end_ms": 500},
            },
        },
        runtime.context,
    )
    initial = AgentState.initial(
        TaskInput(
            task_id="leak-task",
            video_ref=runtime.source,
            instruction="check the public tool boundary",
        ),
        ExecutionBudget(max_steps=2, max_tool_calls=2, max_wall_time_ms=30_000),
    )
    event = tool_record_to_event(
        record,
        initial,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    policy_json = PolicyViewSerializer.to_json(
        PolicyContextBuilder().build(StateReducer.apply(initial, event))
    ).casefold()
    for forbidden in (
        "benchmarkgold",
        "source_group_id",
        '"split"',
        '"uri"',
        "sha256",
        "filesystem_path",
        "evaluator_metadata",
        str(tmp_path).casefold(),
    ):
        assert forbidden not in policy_json

    trace_payload = json.loads(
        next((tmp_path / "runtime" / "traces").rglob("*.json")).read_text(encoding="utf-8")
    )
    serialized_trace = json.dumps(trace_payload).casefold()
    assert str(tmp_path).casefold() not in serialized_trace
    assert "password" not in serialized_trace
