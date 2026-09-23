"""Deterministic plan progression and constrained local recovery for M4B."""

from __future__ import annotations

from dataclasses import dataclass

from cutagent.agent.planning import update_readiness
from cutagent.schemas.agent import PlanGraph
from cutagent.schemas.event import PlanNode, PlanPatch, VerificationStatus
from cutagent.schemas.m4b_agent import (
    CannotRecover,
    InsertRecoveryNode,
    ModifyCurrentNode,
    RecoveryDecision,
    RetryCurrentNode,
    SkipBlockedNode,
)


@dataclass(frozen=True, slots=True)
class LocalRecoveryPatch:
    patch: PlanPatch
    affected_node_ids: tuple[str, ...]


def active_plan_node(graph: PlanGraph) -> PlanNode | None:
    """Return the one node the runtime may execute or recover locally."""

    ready = update_readiness(graph)
    for statuses in (("running",), ("failed", "blocked"), ("ready",)):
        selected = next((node for node in ready.nodes if node.status in statuses), None)
        if selected is not None:
            return selected
    return None


def progress_plan_node(
    graph: PlanGraph,
    *,
    node_id: str,
    verification_status: VerificationStatus,
) -> PlanPatch:
    """Move only the active node; readiness of its dependents is derived."""

    nodes = {node.node_id: node for node in graph.nodes}
    current = nodes.get(node_id)
    if current is None:
        raise ValueError("cannot progress an unknown plan node")
    if current.status == "succeeded":
        raise ValueError("cannot progress a succeeded plan node")
    next_status = {
        "passed": "succeeded",
        "failed": "failed",
        "inconclusive": "running",
    }[verification_status]
    progressed = tuple(
        node.model_copy(update={"status": next_status}) if node.node_id == node_id else node
        for node in graph.nodes
    )
    next_graph = update_readiness(PlanGraph(revision=graph.revision + 1, nodes=progressed))
    return next_graph.as_patch(
        reason=f"M4B deterministic progression after {verification_status} verification"
    )


def _require_local_target(graph: PlanGraph, node_id: str, active_node_id: str) -> PlanNode:
    if node_id != active_node_id:
        raise ValueError("recovery may target only the active failed/blocked node")
    node = next((item for item in graph.nodes if item.node_id == node_id), None)
    if node is None:
        raise ValueError("recovery referenced an unknown node")
    if node.status not in {"failed", "blocked"}:
        raise ValueError("recovery target must be failed or blocked")
    return node


def _assert_succeeded_unchanged(current: PlanGraph, candidate: PlanGraph) -> None:
    by_id = {node.node_id: node for node in candidate.nodes}
    for node in current.nodes:
        if node.status == "succeeded" and by_id.get(node.node_id) != node:
            raise ValueError("compact recovery cannot rewrite succeeded history")


def build_local_recovery_patch(
    graph: PlanGraph,
    decision: RecoveryDecision,
    *,
    active_node_id: str,
    recovery_index: int,
) -> LocalRecoveryPatch | None:
    """Translate one compact model decision into a runtime-owned safe PlanPatch."""

    if isinstance(decision, CannotRecover):
        return None
    nodes = list(graph.nodes)
    affected: tuple[str, ...]
    if isinstance(decision, RetryCurrentNode):
        target = _require_local_target(graph, decision.node_id, active_node_id)
        replacement = target.model_copy(
            update={
                "status": "ready",
                "preferred_capability": (
                    decision.preferred_capability
                    if decision.preferred_capability is not None
                    else target.preferred_capability
                ),
            }
        )
        nodes = [replacement if node.node_id == target.node_id else node for node in nodes]
        affected = (target.node_id,)
    elif isinstance(decision, ModifyCurrentNode):
        target = _require_local_target(graph, decision.node_id, active_node_id)
        replacement = target.model_copy(
            update={
                "subgoal": decision.revised_subgoal,
                "completion_criteria": decision.revised_completion_criteria,
                "preferred_capability": decision.preferred_capability,
                "status": "ready",
            }
        )
        nodes = [replacement if node.node_id == target.node_id else node for node in nodes]
        affected = (target.node_id,)
    elif isinstance(decision, InsertRecoveryNode):
        target = _require_local_target(graph, decision.affected_node_id, active_node_id)
        recovery_id = f"recovery-{target.node_id}-{recovery_index:02d}"
        if any(node.node_id == recovery_id for node in nodes):
            raise ValueError("deterministic recovery node identifier already exists")
        recovery_node = PlanNode(
            node_id=recovery_id,
            subgoal=decision.recovery_subgoal,
            dependencies=target.dependencies,
            status="pending",
            expected_evidence=decision.completion_criteria,
            preferred_capability=decision.preferred_capability,
            completion_criteria=decision.completion_criteria,
        )
        replacement = target.model_copy(
            update={
                "dependencies": (*target.dependencies, recovery_id),
                "status": "pending",
            }
        )
        nodes = [replacement if node.node_id == target.node_id else node for node in nodes]
        nodes.append(recovery_node)
        affected = (target.node_id, recovery_id)
    elif isinstance(decision, SkipBlockedNode):
        target = _require_local_target(graph, decision.node_id, active_node_id)
        dependents = [node for node in nodes if target.node_id in node.dependencies]
        if target.completion_criteria or any(
            node.status not in {"succeeded", "skipped"} for node in dependents
        ):
            raise ValueError("required or dependency-bearing node cannot be skipped")
        replacement = target.model_copy(update={"status": "skipped"})
        nodes = [replacement if node.node_id == target.node_id else node for node in nodes]
        affected = (target.node_id,)
    else:  # pragma: no cover - RecoveryDecision discriminator is exhaustive
        raise ValueError("unsupported recovery decision")

    next_graph = update_readiness(PlanGraph(revision=graph.revision + 1, nodes=tuple(nodes)))
    _assert_succeeded_unchanged(graph, next_graph)
    patch = next_graph.as_patch(reason=f"compact recovery: {decision.recovery_type}")
    return LocalRecoveryPatch(patch=patch, affected_node_ids=affected)
