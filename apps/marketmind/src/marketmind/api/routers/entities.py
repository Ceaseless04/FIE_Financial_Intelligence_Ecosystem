"""Entity lookup, search, and relationship endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from fie_common.errors import NotFoundError
from marketmind.api.dependencies import (
    PERMISSION_READ_ENTITY,
    ContainerDep,
    require_permission,
)
from marketmind.api.schemas import EntityView, RelationshipView
from marketmind.domain.entities import EntityType

router = APIRouter(
    prefix="/api/v1/entities",
    tags=["entities"],
    dependencies=[Depends(require_permission(PERMISSION_READ_ENTITY))],
)

EntityId = Annotated[str, Path(min_length=1, max_length=64, description="Graph entity id")]


@router.get("/search", response_model=list[EntityView], summary="Search entities by name")
async def search_entities(
    container: ContainerDep,
    q: Annotated[str, Query(min_length=1, max_length=200, description="Name fragment")],
    entity_type: Annotated[EntityType | None, Query(description="Restrict to one type")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[EntityView]:
    """Substring search over entity names.

    Search renders nodes straight from the graph rather than rehydrating domain
    objects: a node written before a domain rule existed should still be
    findable, and the alternative is a search endpoint that fails on its own
    historical data.
    """
    nodes = await container.graph.search(q, entity_type=entity_type, limit=limit)
    return [
        EntityView.from_node(node, EntityType(str(node.get("label") or EntityType.COMPANY)))
        for node in nodes
    ]


@router.get("/{entity_id}", response_model=EntityView, summary="Fetch one entity")
async def get_entity(
    container: ContainerDep,
    entity_id: EntityId,
    entity_type: Annotated[
        EntityType | None, Query(description="Type hint; turns the lookup into one index probe")
    ] = None,
) -> EntityView:
    """Fetch an entity by id.

    Raises:
        NotFoundError: if no node carries the id. Translated to 404.
    """
    node = await container.graph.find_by_id(entity_id, entity_type=entity_type)
    if node is None:
        raise NotFoundError(
            "entity not found",
            details={
                "entity_id": entity_id,
                "entity_type": str(entity_type) if entity_type else None,
            },
        )
    return EntityView.from_node(node, EntityType(str(node.get("label") or EntityType.COMPANY)))


@router.get(
    "/{entity_id}/relationships",
    response_model=list[RelationshipView],
    summary="Relationships of one entity",
)
async def entity_relationships(
    container: ContainerDep,
    entity_id: EntityId,
    direction: Annotated[str, Query(pattern="^(both|incoming|outgoing)$")] = "both",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[RelationshipView]:
    """Direct relationships, each carrying the documents that support it."""
    edges = await container.graph.relationships_of(entity_id, direction=direction, limit=limit)
    return [RelationshipView.from_edge(edge) for edge in edges]


__all__ = ["router"]
