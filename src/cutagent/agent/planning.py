"""Deterministic PlanGraph readiness and targeted-patch validation."""

from __future__ import annotations

from cutagent.schemas.agent import PlanGraph
from cutagent.schemas.event import PlanNode, PlanPatch


def update_readiness(graph: PlanGraph) -> PlanGraph:
    """Derive ready/blocked states without changing completed history."""

    status = {node.node_id: node.status for node in graph.nodes}
    nodes: list[PlanNode] = []
    for node in graph.nodes:
        if node.status not in {"pending", "ready", "blocked"}:
            nodes.append(node)
            continue
        dependencies = tuple(status[item] for item in node.dependencies)
        if any(item in {"failed", "blocked"} for item in dependencies):
            next_status = "blocked"
        elif all(item in {"succeeded", "skipped"} for item in dependencies):
            next_status = "ready"
        else:
            next_status = "pending"
        nodes.append(node.model_copy(update={"status": next_status}))
    return graph.model_copy(update={"nodes": tuple(nodes)})


def current_executable_node(graph: PlanGraph) -> PlanNode | None:
    ready = update_readiness(graph)
    return next(
        (node for node in ready.nodes if node.status in {"running", "ready"}),
        None,
    )


def patch_node_status(graph: PlanGraph, node_id: str, status: str, *, reason: str) -> PlanPatch:
    if status not in {"pending", "ready", "running", "succeeded", "failed", "blocked", "skipped"}:
        raise ValueError("unsupported plan node status")
    if node_id not in {node.node_id for node in graph.nodes}:
        raise ValueError("cannot patch an unknown plan node")
    nodes = tuple(
        node.model_copy(update={"status": status}) if node.node_id == node_id else node
        for node in graph.nodes
    )
    ready = update_readiness(PlanGraph(revision=graph.revision + 1, nodes=nodes))
    return ready.as_patch(reason=reason)


def validate_targeted_patch(
    current: PlanGraph,
    patch: PlanPatch,
    affected_nodes: tuple[str, ...],
) -> PlanGraph:
    if patch.revision != current.revision + 1:
        raise ValueError("replan patch revision must increment exactly once")
    affected = set(affected_nodes)
    if not affected:
        raise ValueError("replan patch must identify affected nodes")
    old = {node.node_id: node for node in current.nodes}
    new = {node.node_id: node for node in patch.steps}
    for node_id, prior in old.items():
        candidate = new.get(node_id)
        if candidate is None:
            if prior.status == "succeeded" or node_id not in affected:
                raise ValueError("replan cannot delete successful or unaffected nodes")
            continue
        if prior.status == "succeeded" and candidate != prior:
            raise ValueError("replan cannot rewrite successful plan history")
        if node_id not in affected and candidate != prior:
            raise ValueError("replan modified a node outside affected_plan_nodes")
    added = set(new) - set(old)
    if added - affected:
        raise ValueError("new replan nodes must be listed as affected")
    return update_readiness(PlanGraph(revision=patch.revision, nodes=patch.steps))
