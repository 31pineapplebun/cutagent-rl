"""Policy, failure, trace, and frozen-retrieval tests for ToolRegistry."""

import inspect
import subprocess
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from cutagent.retrieval.adaptive import AdaptiveHybridRetriever
from cutagent.retrieval.fusion import WeightedRankFusion
from cutagent.retrieval.protocols import MultimodalRetriever
from cutagent.retrieval.retrievers import DenseTextRetriever, SparseRetriever, VisualRetriever
from cutagent.schemas.base import SchemaModel
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalChannel
from cutagent.schemas.tools import SearchVideoArgs, ToolExecutionContext, ToolSpec
from cutagent.tools.artifacts import ArtifactStore
from cutagent.tools.errors import ToolTimeout
from cutagent.tools.executor import FFmpegExecutor
from cutagent.tools.protocols import ToolResult
from cutagent.tools.readonly import SearchVideoTool
from cutagent.tools.registry import ToolRegistry
from cutagent.tools.trace import ToolTraceRecorder
from tests.retrieval_fixtures import build_fake_memory


class _TimeoutTool:
    arguments_type = SearchVideoArgs
    spec = ToolSpec(
        name="search_video",
        version="timeout-fixture-v1",
        description="controlled timeout fixture",
        capabilities=("retrieval.read",),
        argument_schema=cast(dict[str, JsonValue], SearchVideoArgs.model_json_schema()),
        deterministic=True,
        produces_artifact=False,
    )

    def execute(
        self,
        arguments: SchemaModel,
        context: ToolExecutionContext,
        *,
        tool_call_id: str,
    ) -> ToolResult:
        del arguments, context, tool_call_id
        raise ToolTimeout("controlled timeout")


def _registry(root: Path) -> tuple[ToolRegistry, ToolExecutionContext]:
    registry = ToolRegistry(
        artifact_store=ArtifactStore(root / "store"),
        trace_recorder=ToolTraceRecorder(root / "traces"),
    )
    context = ToolExecutionContext(
        execution_id="registry-test",
        allowed_output_root_id="unit",
        allowed_artifact_ids=(),
        allowed_capabilities=("retrieval.read",),
    )
    return registry, context


def test_unknown_and_malformed_tool_calls_return_structured_failures(tmp_path: Path) -> None:
    registry, context = _registry(tmp_path)
    unknown = registry.execute(
        {"tool_name": "arbitrary_shell", "tool_call_id": "bad-001", "arguments": {}},
        context,
    )
    assert unknown.observation.status == "invalid"
    assert unknown.observation.error_code == "unknown_tool"

    registry.register(_TimeoutTool())
    malformed = registry.execute(
        {
            "tool_name": "search_video",
            "tool_call_id": "bad-002",
            "arguments": {"query": "safe", "raw_command": "whoami"},
        },
        context,
    )
    assert malformed.observation.status == "invalid"
    assert malformed.observation.error_code == "invalid_call"


def test_capability_denial_and_timeout_are_structured(tmp_path: Path) -> None:
    registry, context = _registry(tmp_path)
    registry.register(_TimeoutTool())
    denied = registry.execute(
        {
            "tool_name": "search_video",
            "tool_call_id": "denied-001",
            "arguments": {"query": "fixture"},
        },
        context.model_copy(update={"allowed_capabilities": ()}),
    )
    assert denied.observation.error_code == "capability_denied"
    timed_out = registry.execute(
        {
            "tool_name": "search_video",
            "tool_call_id": "timeout-001",
            "arguments": {"query": "fixture"},
        },
        context,
    )
    assert timed_out.observation.status == "timeout"
    assert timed_out.observation.error_code == "timeout"


def test_executor_has_no_shell_true_path() -> None:
    source = inspect.getsource(FFmpegExecutor.run)
    assert "shell=False" in source
    assert "shell=True" not in source
    assert "os.system" not in source
    assert "Popen" not in source


def test_executor_passes_an_argv_vector_and_shell_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "ffmpeg version fixture\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    FFmpegExecutor("ffmpeg-fixture")
    assert captured["command"] == ["ffmpeg-fixture", "-version"]
    assert captured["shell"] is False


def test_search_video_wraps_exact_frozen_m2b_policy(tmp_path: Path) -> None:
    retrieval_root = tmp_path / "retrieval"
    retrieval_root.mkdir()
    memory, text, visual = build_fake_memory(retrieval_root)
    components: dict[RetrievalChannel, MultimodalRetriever] = {
        "bm25_transcript": SparseRetriever("transcript"),
        "bm25_ocr": SparseRetriever("ocr"),
        "bm25_structured": SparseRetriever("structured_semantic"),
        "bm25_combined": SparseRetriever("combined"),
        "dense_text": DenseTextRetriever(text),
        "visual": VisualRetriever(visual),
    }
    retriever = AdaptiveHybridRetriever(
        config=AdaptiveRetrievalConfig(native_video_enabled=False),
        fusion=WeightedRankFusion(components),
        strategy="query_aware",
        evidence_reranking=True,
        native_video=False,
    )
    registry, context = _registry(tmp_path / "tools")
    registry.register(SearchVideoTool(retriever=retriever, memory=memory))
    result = registry.execute(
        {
            "tool_name": "search_video",
            "tool_call_id": "search-001",
            "arguments": {"query": "red square moving right", "top_k": 2},
        },
        context.model_copy(update={"allowed_capabilities": ("retrieval.read",)}),
    )
    assert result.observation.status == "success"
    plan = cast(dict[str, object], result.observation.details["retrieval_plan"])
    assert plan["native_video_policy"] == "disabled"
    assert set(cast(list[str], plan["enabled_channels"])) >= {
        "dense_text",
        "visual",
    }
    assert "rerank_trace" in result.observation.details
