"""Evaluator-private executed oracle used only to create M6 train decisions.

The oracle may read train Gold to choose an action, but the policy request and
persisted model messages remain public-only.  This module is deliberately not
importable from the deployed ``cutagent`` package.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, cast

from cutagent.agent.m4b_protocols import M4BPolicyModelRequest, M4BPolicyModelResult
from cutagent.perception.artifacts import write_json_artifact
from cutagent.schemas.agent import (
    CannotCompleteDecision,
    FinishDecision,
    PlanGraph,
    PolicyDecision,
    ToolDecision,
)
from cutagent.schemas.event import PlanNode
from cutagent.schemas.m4b_agent import (
    M4BPolicyInferenceStats,
    RecoveryDecision,
    RetryCurrentNode,
)
from cutagent.schemas.media import TimeRange
from cutagent.schemas.tools import ToolName
from cutagent_evaluation.m5a_schemas import (
    CutAgentBenchGold,
    ResolutionGoldConstraint,
    SpeedGoldConstraint,
    SubtitleGoldConstraint,
)


class M6ExecutedOraclePolicy:
    """Produce deterministic train-only decisions for real ToolRegistry execution."""

    version = "m6-executed-oracle-policy-v1"

    def __init__(self, gold_by_task: dict[str, CutAgentBenchGold]) -> None:
        self._gold_by_task = dict(gold_by_task)

    @property
    def backend_version(self) -> str:
        return self.version

    @staticmethod
    def _context(request: M4BPolicyModelRequest) -> dict[str, Any]:
        value = json.loads(request.serialized_context)
        if not isinstance(value, dict):
            raise ValueError("oracle received a non-object policy context")
        return cast(dict[str, Any], value)

    @staticmethod
    def _task_id(context: dict[str, Any]) -> str:
        task = context.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str):
            raise ValueError("oracle context omitted public task_id")
        return cast(str, task["task_id"])

    @staticmethod
    def _sequence(gold: CutAgentBenchGold) -> tuple[ToolName, ...]:
        if gold.acceptable_tool_sequences:
            return gold.acceptable_tool_sequences[0]
        return gold.required_tools

    @classmethod
    def _plan(cls, gold: CutAgentBenchGold) -> PlanGraph:
        sequence = cls._sequence(gold)
        nodes: list[PlanNode] = []
        for index, tool_name in enumerate(sequence):
            node_id = f"oracle-{index + 1:02d}-{tool_name.replace('_', '-')}"
            dependencies = () if index == 0 else (nodes[-1].node_id,)
            nodes.append(
                PlanNode(
                    node_id=node_id,
                    subgoal=f"Execute the typed {tool_name} operation required by the task",
                    dependencies=dependencies,
                    status="ready" if index == 0 else "pending",
                    expected_evidence=(f"observable {tool_name} success",),
                    preferred_capability=None,
                    completion_criteria=(f"typed {tool_name} call passes online validation",),
                )
            )
        return PlanGraph(revision=1, nodes=tuple(nodes))

    @staticmethod
    def _active_index(context: dict[str, Any]) -> int | None:
        raw = context.get("plan_steps")
        if not isinstance(raw, list):
            return None
        for index, node in enumerate(raw):
            if isinstance(node, dict) and node.get("status") in {"ready", "running"}:
                return index
        return None

    @staticmethod
    def _artifact_ids(context: dict[str, Any], tool_name: str) -> list[str]:
        observations = context.get("recent_observations")
        if not isinstance(observations, list):
            return []
        values: list[str] = []
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("tool_name") != tool_name:
                continue
            artifacts = observation.get("artifact_ids")
            if isinstance(artifacts, list):
                values.extend(item for item in artifacts if isinstance(item, str))
        return values

    @staticmethod
    def _current_artifact(context: dict[str, Any]) -> str:
        working = context.get("working_artifacts")
        if not isinstance(working, dict):
            raise ValueError("oracle context omitted working artifacts")
        current = working.get("current_working_media")
        if not isinstance(current, dict) or not isinstance(current.get("artifact_id"), str):
            raise ValueError("oracle context omitted current working artifact")
        return cast(str, current["artifact_id"])

    @staticmethod
    def _original_artifact(context: dict[str, Any]) -> str:
        working = context.get("working_artifacts")
        if not isinstance(working, dict) or not isinstance(
            working.get("original_input_artifact_id"), str
        ):
            raise ValueError("oracle context omitted original input artifact")
        return cast(str, working["original_input_artifact_id"])

    @staticmethod
    def _instruction(context: dict[str, Any]) -> str:
        task = context.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("instruction"), str):
            raise ValueError("oracle context omitted public instruction")
        return cast(str, task["instruction"])

    @classmethod
    def _arguments(
        cls,
        *,
        tool_name: ToolName,
        sequence_index: int,
        sequence: tuple[ToolName, ...],
        gold: CutAgentBenchGold,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        current = cls._current_artifact(context)
        if tool_name == "search_video":
            return {
                "query": cls._instruction(context),
                "top_k": 5,
                "video_id": cls._original_artifact(context),
                "required_evidence_types": [],
            }
        if tool_name == "trim_video":
            occurrence = sum(item == "trim_video" for item in sequence[: sequence_index + 1]) - 1
            ranges = gold.acceptable_time_ranges
            selected = ranges[min(occurrence, len(ranges) - 1)]
            return {
                "input_artifact_id": cls._original_artifact(context),
                "time_range": selected.model_dump(mode="json"),
            }
        if tool_name == "concat_videos":
            trim_ids = cls._artifact_ids(context, "trim_video")
            if len(trim_ids) < 2:
                raise ValueError("oracle concat requires two observable trim artifacts")
            return {"input_artifact_ids": trim_ids[-2:]}
        if tool_name == "change_speed":
            factor = next(
                (
                    item.speed_factor
                    for item in gold.objective_constraints
                    if isinstance(item, SpeedGoldConstraint)
                ),
                2.0,
            )
            return {"input_artifact_id": current, "speed_factor": factor}
        if tool_name == "add_subtitles":
            subtitle = next(
                item
                for item in gold.objective_constraints
                if isinstance(item, SubtitleGoldConstraint)
            )
            return {
                "input_artifact_id": current,
                "cues": [
                    {
                        "cue_id": "oracle-subtitle-001",
                        "time_range": subtitle.time_range.model_dump(mode="json"),
                        "text": subtitle.text,
                    }
                ],
                "style": {
                    "font_size": 24,
                    "alignment": "bottom",
                    "text_color": "white",
                    "outline_color": "black",
                },
            }
        if tool_name == "reframe_video":
            resolution = next(
                item
                for item in gold.objective_constraints
                if isinstance(item, ResolutionGoldConstraint)
            )
            return {
                "input_artifact_id": current,
                "width": resolution.width,
                "height": resolution.height,
                "fit": "crop",
            }
        if tool_name == "normalize_audio":
            return {
                "input_artifact_id": current,
                "target_lufs": -16.0,
                "loudness_range": 11.0,
                "true_peak_db": -1.5,
            }
        if tool_name == "inspect_media":
            return {"input_artifact_id": current}
        if tool_name == "validate_media":
            return {
                "input_artifact_id": current,
                "require_audio": True,
                "decode_entire_video": True,
            }
        raise ValueError(f"unsupported oracle tool {tool_name}")

    @classmethod
    def _decision(
        cls,
        gold: CutAgentBenchGold,
        context: dict[str, Any],
    ) -> PolicyDecision:
        sequence = cls._sequence(gold)
        active_index = cls._active_index(context)
        if active_index is None:
            if gold.expected_terminal_behavior == "CANNOT_COMPLETE":
                return CannotCompleteDecision(
                    reason="The requested evidence is absent after observable search.",
                    missing_evidence_or_capability=("requested public evidence",),
                    attempted_action_ids=(),
                )
            current = cls._current_artifact(context)
            return FinishDecision(
                output_artifact_id=current,
                completion_summary="All planned typed operations and validation completed.",
                evidence_ids=(cls._original_artifact(context),),
            )
        tool_name = sequence[active_index]
        call_id = f"oracle-call-{active_index + 1:02d}-{tool_name.replace('_', '-')}"
        return ToolDecision.model_validate(
            {
                "decision_type": "tool",
                "tool_call": {
                    "tool_name": tool_name,
                    "tool_call_id": call_id,
                    "arguments": cls._arguments(
                        tool_name=tool_name,
                        sequence_index=active_index,
                        sequence=sequence,
                        gold=gold,
                        context=context,
                    ),
                },
                "rationale": f"Execute the active typed {tool_name} plan node.",
                "expected_observation": f"A structured {tool_name} ToolObservation.",
                "success_condition": f"Online validation accepts {tool_name}.",
            }
        )

    @staticmethod
    def _raw_artifact(
        request: M4BPolicyModelRequest,
        payload: dict[str, Any],
    ) -> Any:
        return write_json_artifact(
            request.output_directory / "executed_oracle_output.json",
            payload,
            artifact_prefix="m6-executed-oracle",
        )

    def infer(self, request: M4BPolicyModelRequest) -> M4BPolicyModelResult:
        context = self._context(request)
        task_id = self._task_id(context)
        gold = self._gold_by_task[task_id]
        plan: PlanGraph | None = None
        decision: PolicyDecision | RecoveryDecision | None = None
        if request.operation == "plan":
            plan = self._plan(gold)
            raw = plan.model_dump(mode="json")
        elif request.operation == "recover":
            active = context.get("step_outcome")
            node_id = active.get("active_plan_node") if isinstance(active, dict) else None
            if not isinstance(node_id, str):
                raise ValueError("observable recovery has no active failed node")
            decision = RetryCurrentNode(
                node_id=node_id,
                reason="Retry once after the observable injected tool failure.",
            )
            raw = decision.model_dump(mode="json")
        elif request.operation == "decide":
            decision = self._decision(gold, context)
            raw = decision.model_dump(mode="json")
        else:
            raise ValueError("M6 oracle does not use the legacy full replan contract")
        artifact = self._raw_artifact(request, raw)
        return M4BPolicyModelResult(
            operation=request.operation,
            decision=decision,
            proposed_plan=plan,
            raw_output_artifact=artifact,
            stats=M4BPolicyInferenceStats(
                operation=request.operation,
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                repair_count=0,
            ),
        )


def expected_trim_ranges(gold: CutAgentBenchGold) -> tuple[TimeRange, ...]:
    """Expose a narrow helper for executable-dataset audits."""

    return gold.acceptable_time_ranges


def tool_counts(gold_by_task: dict[str, CutAgentBenchGold]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for gold in gold_by_task.values():
        for tool_name in M6ExecutedOraclePolicy._sequence(gold):
            counts[tool_name] += 1
    return dict(counts)
