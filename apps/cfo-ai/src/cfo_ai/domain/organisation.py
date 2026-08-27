"""Cost centres, and the tree they form.

Every plan is reported twice: once by account ("what did we spend it on") and
once by cost centre ("who spent it"). The second view is a hierarchy, and a
hierarchy whose children do not sum to their parent is a budget nobody can
reconcile — the same class of defect as a balance sheet that does not balance,
and caught the same way, at construction.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from fie_common.errors import ValidationError
from fie_schemas.base import FrozenModel


class CostCentre(FrozenModel):
    """An organisational unit that owns budget."""

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    #: ``None`` for a root. Every other centre names its parent.
    parent_id: str | None = Field(default=None, max_length=64)
    #: Who answers for the variance. Carried because a variance report with no
    #: owner is a report nobody acts on.
    owner: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _not_its_own_parent(self) -> CostCentre:
        if self.parent_id is not None and self.parent_id == self.id:
            raise ValueError("a cost centre cannot be its own parent")
        return self


class CostCentreTree(FrozenModel):
    """The full hierarchy, validated as a tree rather than assumed to be one."""

    centres: list[CostCentre] = Field(min_length=1)

    @field_validator("centres")
    @classmethod
    def _unique_ids(cls, value: list[CostCentre]) -> list[CostCentre]:
        seen: set[str] = set()
        for centre in value:
            if centre.id in seen:
                raise ValueError(f"duplicate cost centre id {centre.id!r}")
            seen.add(centre.id)
        return value

    @model_validator(mode="after")
    def _is_a_tree(self) -> CostCentreTree:
        """Reject dangling parents and cycles.

        A cycle makes every roll-up non-terminating, and a dangling parent makes
        a centre's spend vanish from the total. Both are silent until someone
        asks why the numbers do not add up.
        """
        by_id = {centre.id: centre for centre in self.centres}

        for centre in self.centres:
            if centre.parent_id is not None and centre.parent_id not in by_id:
                raise ValueError(
                    f"cost centre {centre.id!r} names a parent {centre.parent_id!r} "
                    "that does not exist"
                )

        # Walk upward from every node; a tree of N nodes cannot have a path
        # longer than N without repeating one.
        for centre in self.centres:
            seen: set[str] = {centre.id}
            current = centre
            while current.parent_id is not None:
                if current.parent_id in seen:
                    raise ValueError(
                        f"cost centre hierarchy contains a cycle through {current.id!r}"
                    )
                seen.add(current.parent_id)
                current = by_id[current.parent_id]

        # No explicit "must have a root" check: it cannot fire. Every parent
        # resolves and no path repeats, so walking upward from any centre
        # terminates in a finite set — and the only way to terminate is at a
        # centre with no parent. A rootless hierarchy is always a cyclic one,
        # and the loop above has already rejected it.
        return self

    @property
    def by_id(self) -> dict[str, CostCentre]:
        return {centre.id: centre for centre in self.centres}

    @property
    def roots(self) -> list[CostCentre]:
        return [centre for centre in self.centres if centre.parent_id is None]

    def children_of(self, centre_id: str) -> list[CostCentre]:
        return [centre for centre in self.centres if centre.parent_id == centre_id]

    def is_leaf(self, centre_id: str) -> bool:
        return not self.children_of(centre_id)

    def leaves(self) -> list[CostCentre]:
        """Centres that own spend directly rather than through children."""
        return [centre for centre in self.centres if self.is_leaf(centre.id)]

    def ancestors_of(self, centre_id: str) -> list[str]:
        """Ids from the immediate parent up to the root, nearest first.

        Raises:
            ValidationError: if the centre is not in this tree.
        """
        by_id = self.by_id
        if centre_id not in by_id:
            raise ValidationError("unknown cost centre", details={"cost_centre_id": centre_id})
        chain: list[str] = []
        current = by_id[centre_id]
        while current.parent_id is not None:
            chain.append(current.parent_id)
            current = by_id[current.parent_id]
        return chain

    def descendants_of(self, centre_id: str) -> list[str]:
        """Every id beneath a centre, at any depth."""
        found: list[str] = []
        frontier = [centre_id]
        while frontier:
            current = frontier.pop()
            for child in self.children_of(current):
                found.append(child.id)
                frontier.append(child.id)
        return found

    def depth_of(self, centre_id: str) -> int:
        """Distance from the root. A root is depth 0."""
        return len(self.ancestors_of(centre_id))


__all__ = ["CostCentre", "CostCentreTree"]
