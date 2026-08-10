"""Graph traversal, path finding, and visualization endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from fie_common.errors import NotFoundError
from marketmind.api.dependencies import (
    PERMISSION_READ_GRAPH,
    ContainerDep,
    require_permission,
)
from marketmind.api.schemas import (
    GraphEdgeView,
    GraphNodeView,
    GraphStatsView,
    GraphView,
    PathStep,
    PathView,
)
from marketmind.domain.relationships import RelationshipType

router = APIRouter(
    prefix="/api/v1/graph",
    tags=["graph"],
    dependencies=[Depends(require_permission(PERMISSION_READ_GRAPH))],
)

EntityId = Annotated[str, Path(min_length=1, max_length=64, description="Graph entity id")]


@router.get(
    "/visualization/{entity_id}",
    response_model=GraphView,
    summary="Neighbourhood as nodes and edges",
)
async def visualization(
    container: ContainerDep,
    entity_id: EntityId,
    depth: Annotated[int, Query(ge=1, le=3, description="Hops from the seed")] = 1,
    limit: Annotated[int | None, Query(ge=1, le=2000)] = None,
    relationship_types: Annotated[
        list[RelationshipType] | None, Query(description="Restrict traversal to these edge types")
    ] = None,
) -> GraphView:
    """A seed entity's neighbourhood, shaped for a force-directed renderer.

    Depth is capped at 3 and node count at a configured maximum. Both caps are
    load-bearing rather than defensive: financial graphs are dense enough that
    an unbounded three-hop expansion from a large company returns a result no
    browser can lay out and no reader can interpret.
    """
    node_cap = min(
        limit or container.settings.graph_visualization_max_nodes,
        container.settings.graph_visualization_max_nodes,
    )

    neighbourhood = await container.graph.neighbourhood(
        entity_id,
        depth=depth,
        relationship_types=relationship_types,
        limit=node_cap,
    )
    seed_node = await container.graph.find_by_id(entity_id)
    if seed_node is None:
        raise NotFoundError("entity not found", details={"entity_id": entity_id})

    nodes = [
        GraphNodeView(
            id=str(seed_node["id"]),
            label=str(seed_node.get("label") or ""),
            name=str(seed_node.get("name") or ""),
            distance=0,
            source_document_ids=[str(s) for s in seed_node.get("source_ids") or []],
        )
    ]
    nodes.extend(GraphNodeView.from_node(node) for node in neighbourhood.nodes)

    known_ids = {node.id for node in nodes}
    edges: list[GraphEdgeView] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # Edges are collected per node rather than from the traversal paths so the
    # rendered graph shows connections *between* neighbours, not just spokes
    # back to the seed.
    for node_id in list(known_ids):
        for edge in await container.graph.relationships_of(node_id, limit=node_cap):
            if edge.other_id not in known_ids:
                continue
            source, target = (
                (node_id, edge.other_id) if edge.is_outgoing else (edge.other_id, node_id)
            )
            key = (edge.type, source, target)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(
                GraphEdgeView(
                    id=edge.relationship_id,
                    type=edge.type,
                    source=source,
                    target=target,
                    source_document_ids=list(edge.source_ids),
                )
            )

    return GraphView(
        seed_entity_id=entity_id,
        depth=depth,
        nodes=nodes,
        edges=edges,
        truncated=len(neighbourhood.nodes) >= node_cap,
    )


@router.get("/path", response_model=PathView, summary="Shortest path between two entities")
async def shortest_path(
    container: ContainerDep,
    source: Annotated[str, Query(min_length=1, max_length=64)],
    target: Annotated[str, Query(min_length=1, max_length=64)],
    max_depth: Annotated[int, Query(ge=1, le=6)] = 4,
) -> PathView:
    """How two entities connect.

    This is the query Sentinel builds exposure explanations from: "this supplier
    reaches that customer in three hops" is a fact about the graph, and stating
    the path is what makes the claim checkable.

    Raises:
        NotFoundError: if no path exists within ``max_depth``.
    """
    result = await container.graph.shortest_path(source, target, max_depth=max_depth)
    if result is None:
        raise NotFoundError(
            "no path found between the entities within the depth limit",
            details={"source": source, "target": target, "max_depth": max_depth},
        )

    return PathView(
        source_entity_id=source,
        target_entity_id=target,
        length=int(result.get("length") or 0),
        nodes=[
            PathStep(
                id=str(node.get("id") or ""),
                name=str(node.get("name") or ""),
                label=str(node.get("label") or ""),
            )
            for node in result.get("nodes") or []
        ],
        relationship_types=[
            str(relationship.get("type") or "")
            for relationship in result.get("relationships") or []
        ],
    )


@router.get("/stats", response_model=GraphStatsView, summary="Graph size by label and type")
async def stats(container: ContainerDep) -> GraphStatsView:
    """Node and relationship counts."""
    raw = await container.graph.stats()
    return GraphStatsView(
        total_nodes=int(raw["total_nodes"]),
        total_relationships=int(raw["total_relationships"]),
        nodes_by_label={str(k): int(v) for k, v in raw["nodes_by_label"].items()},
        relationships_by_type={str(k): int(v) for k, v in raw["relationships_by_type"].items()},
    )


__all__ = ["router"]
