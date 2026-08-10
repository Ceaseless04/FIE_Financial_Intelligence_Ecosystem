"""Unit tests for role-based access control."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from fie_auth.rbac import (
    ANONYMOUS,
    ROLE_PERMISSIONS,
    Principal,
    Product,
    Role,
    permission_matches,
    validate_permission,
)
from fie_common.errors import AuthorizationError, ValidationError

pytestmark = pytest.mark.unit


class TestPermissionValidation:
    @pytest.mark.parametrize(
        "permission",
        ["*", "atlas:*", "atlas:research:read", "platform:events:write", "cfo_ai:budget:write"],
    )
    def test_accepts_well_formed_permissions(self, permission: str) -> None:
        assert validate_permission(permission) == permission

    @pytest.mark.parametrize(
        "permission", ["Atlas:Read", "atlas research", "atlas::read", "", "atlas:read!"]
    )
    def test_rejects_malformed_permissions(self, permission: str) -> None:
        with pytest.raises(ValidationError):
            validate_permission(permission)


class TestPermissionMatching:
    def test_exact_match(self) -> None:
        assert permission_matches("atlas:research:read", "atlas:research:read") is True

    def test_global_wildcard_grants_everything(self) -> None:
        assert permission_matches("*", "atlas:research:read") is True

    def test_trailing_wildcard_covers_the_subtree(self) -> None:
        assert permission_matches("atlas:*", "atlas:research:read") is True
        assert permission_matches("atlas:*", "atlas:filing:write") is True

    def test_segment_wildcard_matches_one_segment(self) -> None:
        assert permission_matches("atlas:*:read", "atlas:research:read") is True
        assert permission_matches("atlas:*:read", "atlas:research:write") is False

    def test_does_not_cross_product_boundaries(self) -> None:
        assert permission_matches("atlas:*", "marketmind:graph:read") is False

    def test_narrower_grant_does_not_satisfy_a_broader_requirement(self) -> None:
        assert permission_matches("atlas:research:read", "atlas:research") is False

    def test_length_mismatch_without_a_wildcard_fails(self) -> None:
        assert permission_matches("atlas:research", "atlas:research:read") is False


class TestPrincipal:
    def test_explicit_permissions_are_validated(self) -> None:
        # The field validator raises the platform's own ValidationError rather
        # than a Pydantic one, so callers get the same typed error (and HTTP
        # 422 mapping) whether the permission is rejected at construction or at
        # check time.
        with pytest.raises(ValidationError, match="malformed permission"):
            Principal(subject="user_1", permissions=["INVALID PERMISSION"])

    def test_requires_a_subject(self) -> None:
        with pytest.raises(PydanticValidationError):
            Principal(subject="")

    def test_effective_permissions_union_roles_and_explicit_grants(self) -> None:
        principal = Principal(
            subject="user_1", roles=[Role.VIEWER], permissions=["atlas:research:write"]
        )

        assert "atlas:research:write" in principal.effective_permissions
        assert "atlas:*:read" in principal.effective_permissions

    def test_admin_has_every_permission(self) -> None:
        admin = Principal(subject="admin_1", roles=[Role.ADMIN])

        assert admin.has_permission("atlas:research:read") is True
        assert admin.has_permission("finops:cost:write") is True

    def test_viewer_can_read_but_not_write(self) -> None:
        viewer = Principal(subject="viewer_1", roles=[Role.VIEWER])

        assert viewer.has_permission("atlas:research:read") is True
        assert viewer.has_permission("atlas:research:write") is False

    def test_analyst_can_write_research_only(self) -> None:
        analyst = Principal(subject="analyst_1", roles=[Role.ANALYST])

        assert analyst.has_permission("atlas:research:write") is True
        assert analyst.has_permission("marketmind:graph:write") is False

    def test_service_account_is_scoped_to_events(self) -> None:
        service = Principal(subject="svc_atlas", roles=[Role.SERVICE], is_service_account=True)

        assert service.has_permission("platform:events:write") is True
        assert service.has_permission("atlas:research:read") is False

    def test_multiple_roles_are_additive(self) -> None:
        principal = Principal(subject="user_1", roles=[Role.VIEWER, Role.SERVICE])

        assert principal.has_permission("atlas:research:read") is True
        assert principal.has_permission("platform:events:write") is True

    def test_anonymous_has_nothing(self) -> None:
        assert ANONYMOUS.has_permission("atlas:research:read") is False
        assert ANONYMOUS.effective_permissions == frozenset()

    def test_has_any_and_has_all(self) -> None:
        viewer = Principal(subject="viewer_1", roles=[Role.VIEWER])

        assert viewer.has_any_permission("atlas:research:write", "atlas:research:read") is True
        assert viewer.has_all_permissions("atlas:research:write", "atlas:research:read") is False
        assert viewer.has_all_permissions("atlas:research:read", "sentinel:risk:read") is True

    def test_checking_a_malformed_permission_raises(self) -> None:
        with pytest.raises(ValidationError):
            Principal(subject="user_1").has_permission("NOT VALID")


class TestRequirePermission:
    def test_passes_when_granted(self) -> None:
        Principal(subject="admin_1", roles=[Role.ADMIN]).require_permission("atlas:research:read")

    def test_raises_when_missing(self) -> None:
        viewer = Principal(subject="viewer_1", roles=[Role.VIEWER])

        with pytest.raises(AuthorizationError, match="lacks required permission"):
            viewer.require_permission("atlas:research:write")

    def test_error_details_identify_the_principal_and_requirement(self) -> None:
        viewer = Principal(subject="viewer_1", roles=[Role.VIEWER])

        with pytest.raises(AuthorizationError) as exc_info:
            viewer.require_permission("atlas:research:write")

        assert exc_info.value.details["subject"] == "viewer_1"
        assert exc_info.value.details["required"] == "atlas:research:write"


class TestTenantIsolation:
    def test_matching_tenant_passes(self) -> None:
        Principal(subject="user_1", tenant_id="tenant_a").require_tenant("tenant_a")

    def test_cross_tenant_access_is_rejected(self) -> None:
        # The failure mode that matters most in a multi-tenant financial
        # platform, so it is checked separately from permissions.
        principal = Principal(subject="user_1", tenant_id="tenant_a")

        with pytest.raises(AuthorizationError, match="not a member of the requested tenant"):
            principal.require_tenant("tenant_b")

    def test_an_admin_still_cannot_cross_tenants_implicitly(self) -> None:
        admin = Principal(subject="admin_1", roles=[Role.ADMIN], tenant_id="tenant_a")

        with pytest.raises(AuthorizationError):
            admin.require_tenant("tenant_b")

    def test_a_tenantless_principal_is_not_restricted(self) -> None:
        Principal(subject="svc_platform", tenant_id=None).require_tenant("any_tenant")


class TestRolePermissionTable:
    def test_every_role_has_an_entry(self) -> None:
        assert set(ROLE_PERMISSIONS) == set(Role)

    def test_all_configured_permissions_are_well_formed(self) -> None:
        for role, permissions in ROLE_PERMISSIONS.items():
            for permission in permissions:
                assert validate_permission(permission), f"{role} has a malformed grant"

    def test_every_product_is_addressable(self) -> None:
        analyst = Principal(subject="a", roles=[Role.ANALYST])
        readable = {
            Product.ATLAS,
            Product.MARKETMIND,
            Product.SENTINEL,
            Product.VENTURE,
            Product.CFO_AI,
            Product.FINOPS,
        }

        for product in readable:
            assert analyst.has_permission(f"{product}:resource:read") is True
