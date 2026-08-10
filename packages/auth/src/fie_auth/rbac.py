"""Role-based access control across the six products.

Permissions are colon-delimited ``product:resource:action`` strings so a single
grant can scope to one product without enumerating every endpoint. Wildcards
match a whole segment subtree — ``atlas:*`` covers ``atlas:research:read``.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field, field_validator

from fie_common.errors import AuthorizationError, ValidationError
from fie_schemas.base import FrozenModel

_PERMISSION_PATTERN = re.compile(r"^[a-z0-9_]+(:[a-z0-9_*]+)*$|^\*$")


class Product(StrEnum):
    """The six products, used as the leading permission segment."""

    ATLAS = "atlas"
    CFO_AI = "cfo_ai"
    MARKETMIND = "marketmind"
    SENTINEL = "sentinel"
    FINOPS = "finops"
    VENTURE = "venture"
    PLATFORM = "platform"


class Role(StrEnum):
    """Coarse role assigned to a principal."""

    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"
    #: Machine-to-machine caller (service account).
    SERVICE = "service"


#: Baseline grants per role. Explicit per-principal permissions are additive.
ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.ADMIN: frozenset({"*"}),
    Role.ANALYST: frozenset(
        {
            "atlas:*:read",
            "atlas:research:write",
            "marketmind:*:read",
            "sentinel:*:read",
            "venture:*:read",
            "cfo_ai:*:read",
            "finops:*:read",
        }
    ),
    Role.OPERATOR: frozenset(
        {
            "platform:*",
            "finops:*",
            "sentinel:alert:write",
            "marketmind:ingestion:write",
        }
    ),
    Role.VIEWER: frozenset(
        {
            "atlas:*:read",
            "marketmind:*:read",
            "sentinel:*:read",
            "venture:*:read",
            "cfo_ai:*:read",
            "finops:*:read",
        }
    ),
    Role.SERVICE: frozenset({"platform:events:write", "platform:events:read"}),
}


def validate_permission(permission: str) -> str:
    """Validate a permission string's shape.

    Raises:
        ValidationError: if the permission is malformed.
    """
    if not _PERMISSION_PATTERN.match(permission):
        raise ValidationError(
            f"malformed permission {permission!r}; expected 'product:resource:action'",
            details={"permission": permission},
        )
    return permission


def permission_matches(granted: str, required: str) -> bool:
    """Whether a granted permission satisfies a required one.

    ``*`` alone grants everything. Within a pattern, a ``*`` segment matches
    exactly one segment, and a trailing ``*`` also matches any remaining
    segments — so ``atlas:*`` satisfies ``atlas:research:read``.
    """
    if granted == "*":
        return True
    if granted == required:
        return True

    granted_parts = granted.split(":")
    required_parts = required.split(":")

    for index, granted_part in enumerate(granted_parts):
        if granted_part == "*" and index == len(granted_parts) - 1:
            # Trailing wildcard absorbs the rest of the required permission.
            return len(required_parts) >= index
        if index >= len(required_parts):
            return False
        if granted_part != "*" and granted_part != required_parts[index]:
            return False

    return len(granted_parts) == len(required_parts)


class Principal(FrozenModel):
    """The authenticated caller.

    Constructed from verified token claims — never from request input.
    """

    subject: str = Field(min_length=1, description="Stable user or service identifier")
    tenant_id: str | None = None
    roles: list[Role] = Field(default_factory=list)
    #: Grants beyond those implied by the principal's roles.
    permissions: list[str] = Field(default_factory=list)
    display_name: str | None = None
    is_service_account: bool = False

    @field_validator("permissions")
    @classmethod
    def _validate_permissions(cls, value: list[str]) -> list[str]:
        return [validate_permission(permission) for permission in value]

    @property
    def effective_permissions(self) -> frozenset[str]:
        """Role-derived grants unioned with explicit ones."""
        granted: set[str] = set(self.permissions)
        for role in self.roles:
            granted |= ROLE_PERMISSIONS.get(role, frozenset())
        return frozenset(granted)

    def has_permission(self, permission: str) -> bool:
        """Whether this principal satisfies ``permission``."""
        validate_permission(permission)
        return any(
            permission_matches(granted, permission) for granted in self.effective_permissions
        )

    def has_any_permission(self, *permissions: str) -> bool:
        return any(self.has_permission(permission) for permission in permissions)

    def has_all_permissions(self, *permissions: str) -> bool:
        return all(self.has_permission(permission) for permission in permissions)

    def require_permission(self, permission: str) -> None:
        """Assert a permission.

        Raises:
            AuthorizationError: if the principal lacks it.
        """
        if not self.has_permission(permission):
            raise AuthorizationError(
                f"principal lacks required permission '{permission}'",
                details={
                    "subject": self.subject,
                    "required": permission,
                    "roles": [str(role) for role in self.roles],
                },
            )

    def require_tenant(self, tenant_id: str) -> None:
        """Assert the principal belongs to ``tenant_id``.

        Cross-tenant reads are the failure mode that matters most in a
        multi-tenant financial platform, so tenancy is checked separately from
        permissions rather than folded into them.
        """
        if self.tenant_id is not None and self.tenant_id != tenant_id:
            raise AuthorizationError(
                "principal is not a member of the requested tenant",
                details={"subject": self.subject, "requested_tenant": tenant_id},
            )


ANONYMOUS = Principal(subject="anonymous", roles=[], permissions=[])

__all__ = [
    "ANONYMOUS",
    "ROLE_PERMISSIONS",
    "Principal",
    "Product",
    "Role",
    "permission_matches",
    "validate_permission",
]
