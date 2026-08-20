"""API response envelopes, error bodies, and pagination."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import Field, field_validator, model_validator

from fie_common.errors import FIEError
from fie_common.utils import utc_now
from fie_schemas.base import FIEModel, FrozenModel

T = TypeVar("T")

#: Types that survive ``json.dumps`` unchanged.
_JSON_SCALARS = (str, int, float, bool, type(None))


def json_safe(value: Any) -> Any:
    """Coerce a value into something JSON can represent.

    Containers are walked; anything else that is not already a JSON scalar
    becomes its ``str``. Nothing is dropped, because a detail a caller cannot
    read is still better than one they never receive.
    """
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return str(value)


class ErrorDetail(FrozenModel):
    """Machine-readable error body. Mirrors ``FIEError.to_dict``."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None

    @field_validator("details", mode="before")
    @classmethod
    def _make_details_serializable(cls, value: Any) -> Any:
        """Guarantee the body can actually be written.

        Error details arrive from anywhere — including Pydantic's own validation
        errors, whose ``ctx`` carries the raised ``ValueError`` *object*.
        Serializing that raises inside the error handler, which turns a client's
        malformed request into a 500 and hides what was actually wrong. An error
        path that can itself fail is the one place that must not.
        """
        return json_safe(value) if isinstance(value, dict) else value

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


__all__ = ["ErrorDetail", "Page", "PageInfo", "PageRequest", "ResponseEnvelope", "json_safe"]
