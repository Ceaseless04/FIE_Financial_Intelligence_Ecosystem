"""fie_ai — the provider abstraction every agent and service codes against.

    AIProvider
    ├── ClaudeProvider   (default production reasoning model)
    ├── OllamaProvider   (local development, evaluation, cost control)
    └── VLLMProvider     (self-hosted batch inference)

Architectural rule: Claude orchestrates, interprets, and explains. It is never
the source of truth for a financial calculation — valuation, forecasting,
accounting, risk scoring, and cost computation all belong to deterministic
Python services in their respective product packages.

Concrete providers are imported lazily via ``fie_ai.factory`` so importing this
package does not require every vendor SDK to be installed.
"""

from fie_ai.base import AIProvider, refusal_response
from fie_ai.config import (
    DEFAULT_CLAUDE_MODEL,
    AISettings,
    ClaudeSettings,
    OllamaSettings,
    ProviderName,
    VLLMSettings,
)
from fie_ai.contracts import (
    CompletionRequest,
    CompletionResponse,
    Effort,
    Message,
    ProviderCapabilities,
    Role,
    StopReason,
    ThinkingMode,
    TokenUsage,
)
from fie_ai.factory import build_registry, build_router
from fie_ai.registry import AIRouter, ProviderRegistry
from fie_ai.structured import (
    GroundingReport,
    check_grounding,
    extract_json,
    parse_structured,
    structured_request,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CLAUDE_MODEL",
    "AIProvider",
    "AIRouter",
    "AISettings",
    "ClaudeSettings",
    "CompletionRequest",
    "CompletionResponse",
    "Effort",
    "GroundingReport",
    "Message",
    "OllamaSettings",
    "ProviderCapabilities",
    "ProviderName",
    "ProviderRegistry",
    "Role",
    "StopReason",
    "ThinkingMode",
    "TokenUsage",
    "VLLMSettings",
    "__version__",
    "build_registry",
    "build_router",
    "check_grounding",
    "extract_json",
    "parse_structured",
    "refusal_response",
    "structured_request",
]
