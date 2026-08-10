"""Structured-output helpers: schema-constrained generation and validation.

The testing architecture forbids asserting on exact model text. Structured
output is the mechanism that makes AI behaviour testable instead: the model
fills a schema, the schema is validated deterministically, and tests assert on
typed fields rather than prose.

Grounding checks live here too — verifying that every citation a model emits
actually resolves to a source that was supplied to it, which is what turns
"cite source data" from an instruction into an enforced property.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from fie_ai.contracts import CompletionRequest, CompletionResponse
from fie_common.errors import AIProviderResponseError
from fie_schemas.base import FIEModel

T = TypeVar("T", bound=BaseModel)

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of model output.

    Tolerates the two things models do even under schema constraints: wrapping
    the payload in a code fence, and prefixing it with a sentence.

    Raises:
        AIProviderResponseError: if no JSON value can be recovered.
    """
    candidate = text.strip()
    if not candidate:
        raise AIProviderResponseError("model returned empty output where JSON was expected")

    fenced = _FENCE_PATTERN.search(candidate)
    if fenced:
        candidate = fenced.group("body").strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost balanced object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise AIProviderResponseError(
        "model output did not contain valid JSON",
        details={"output_preview": text[:200]},
    )


def parse_structured(response: CompletionResponse, output_model: type[T]) -> T:
    """Validate a completion against ``output_model``.

    Raises:
        AIProviderResponseError: on refusal, truncation, or schema violation.
    """
    if response.is_refusal:
        raise AIProviderResponseError(
            "model refused the request; no structured output produced",
            details={"refusal_category": response.refusal_category},
        )
    if response.is_truncated:
        raise AIProviderResponseError(
            "model output was truncated at max_tokens; structured output is incomplete",
            details={"model": response.model, "provider": response.provider},
        )

    payload = extract_json(response.text)
    try:
        return output_model.model_validate(payload)
    except PydanticValidationError as error:
        raise AIProviderResponseError(
            f"model output failed schema validation for {output_model.__name__}",
            details={"errors": error.errors(include_url=False)},
        ) from error


def structured_request(
    request: CompletionRequest, output_model: type[BaseModel]
) -> CompletionRequest:
    """Attach ``output_model``'s JSON Schema to a request."""
    return request.model_copy(update={"response_schema": output_model.model_json_schema()})


class GroundingReport(FIEModel):
    """Result of checking a model's citations against the supplied sources."""

    cited_ids: list[str]
    allowed_ids: list[str]
    unknown_ids: list[str]
    uncited: bool

    @property
    def is_grounded(self) -> bool:
        """Every citation resolves to a supplied source, and at least one exists."""
        return not self.unknown_ids and not self.uncited


def check_grounding(
    cited_source_ids: Iterable[str],
    allowed_source_ids: Iterable[str],
    *,
    require_citation: bool = True,
) -> GroundingReport:
    """Verify that model-emitted citations resolve to real supplied sources.

    A fabricated citation is the most dangerous hallucination in a financial
    research product, because it looks exactly like a real one. This makes it a
    test assertion rather than a review step.
    """
    cited = list(dict.fromkeys(cited_source_ids))
    allowed = list(dict.fromkeys(allowed_source_ids))
    allowed_set = set(allowed)
    unknown = [source_id for source_id in cited if source_id not in allowed_set]
    return GroundingReport(
        cited_ids=cited,
        allowed_ids=allowed,
        unknown_ids=unknown,
        uncited=require_citation and not cited,
    )


__all__ = [
    "GroundingReport",
    "check_grounding",
    "extract_json",
    "parse_structured",
    "structured_request",
]
