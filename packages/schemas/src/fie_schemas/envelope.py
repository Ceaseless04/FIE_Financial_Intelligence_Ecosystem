"""API response envelopes, error bodies, and pagination."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import Field, model_validator

from fie_common.errors import FIEError
from fie_common.utils import utc_now
from fie_schemas.base import FIEModel, FrozenModel

T = TypeVar("T")


class ErrorDetail(FrozenModel):
    """Machine-readable error body. Mirrors ``FIEError.to_dict``."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None

    @classmethod
    def from_error(cls, error: FIEError, *, correlation_id: str | None = None) -> ErrorDetail:
        return cls(
            code=error.code,
            message=error.message,
            retryable=error.retryable,
            details=error.details,
            correlation_id=correlation_id,
        )


class ResponseEnvelope(FIEModel, Generic[T]):
    """Uniform success/failure envelope for every service API."""

    success: bool = True
    data: T | None = None
    error: ErrorDetail | None = None
    correlation_id: str | None = None
    timestamp: Any = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_exclusivity(self) -> ResponseEnvelope[T]:
        if self.success and self.error is not None:
            raise ValueError("a successful response must not carry an error")
        if not self.success and self.error is None:
            raise ValueError("a failed response must carry an error")
        return self

    @classmethod
    def ok(cls, data: T, *, correlation_id: str | None = None) -> ResponseEnvelope[T]:
        return cls(success=True, data=data, correlation_id=correlation_id)

    @classmethod
    def failure(
        cls, error: FIEError | ErrorDetail, *, correlation_id: str | None = None
    ) -> ResponseEnvelope[T]:
        detail = (
            error
            if isinstance(error, ErrorDetail)
            else ErrorDetail.from_error(error, correlation_id=correlation_id)
        )
        return cls(success=False, data=None, error=detail, correlation_id=correlation_id)


class PageRequest(FIEModel):
    """Cursor-or-offset pagination request."""

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    cursor: str | None = None


class PageInfo(FrozenModel):
    """Pagination metadata returned alongside a page of results."""

    total: int | None = Field(default=None, ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(default=0, ge=0)
    has_more: bool = False
    next_cursor: str | None = None


class Page(FIEModel, Generic[T]):
    """A page of results plus its pagination metadata."""

    items: list[T] = Field(default_factory=list)
    page_info: PageInfo

    @classmethod
    def of(
        cls,
        items: list[T],
        *,
        limit: int,
        offset: int = 0,
        total: int | None = None,
        next_cursor: str | None = None,
    ) -> Page[T]:
        has_more = (offset + len(items)) < total if total is not None else len(items) == limit
        return cls(
            items=items,
            page_info=PageInfo(
                total=total,
                limit=limit,
                offset=offset,
                has_more=has_more,
                next_cursor=next_cursor,
            ),
        )


__all__ = ["ErrorDetail", "Page", "PageInfo", "PageRequest", "ResponseEnvelope"]
