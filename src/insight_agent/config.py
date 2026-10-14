"""Configuration, loaded once from the environment with a ``.env`` fallback.

Design rule that shapes this module: **a missing API key must never break the
project.** ``rules`` is a first-class provider, not a degraded mode, so the
system runs fully offline with no key at all. A key is an upgrade, not a
requirement.

That is why nothing here raises on absence. ``LLMConfig`` reports what it
found, and the CLI decides what to do about it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Provider = Literal["rules", "auto", "anthropic", "openai"]
PROVIDERS: tuple[str, ...] = ("rules", "auto", "anthropic", "openai")

DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4.1-mini",
}

ENV_PREFIX = "INSIGHT_"

# Named so tests can point it somewhere harmless; a unit test must never
# pick up the developer's own .env or make a network call.
DEFAULT_DOTENV = Path(".env")


@dataclass(slots=True)
class LLMConfig:
    """Which backend to use, and whether it is actually usable."""

    provider: str = "auto"
    model: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Provider-specific endpoints, falling back to the shared one.
    anthropic_base_url: str = ""
    openai_base_url: str = ""
    base_url: str = ""  # resolved for the active provider
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_s: float = 60.0
    source: dict[str, str] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        """True only when a live model can actually be reached.

        Not merely "the provider is a live one": `auto` with no key selects the
        rules planner, and reporting `live: true` for that would be a lie in
        the one place the user looks to check.
        """

        return self.provider in ("auto", "anthropic", "openai") and bool(self.api_key)

    @property
    def api_key(self) -> str:
        if self.provider == "anthropic":
            return self.anthropic_api_key
        if self.provider == "openai":
            return self.openai_api_key
        return self.openai_api_key or self.anthropic_api_key

    @property
    def resolved_base_url(self) -> str:
        """Endpoint for the active provider.

        A provider-specific variable wins over the shared one, so both protocols
        can be configured at once without one overriding the other.
        """

        if self.provider == "anthropic":
            return self.anthropic_base_url or self.base_url
        if self.provider == "openai":
            return self.openai_base_url or self.base_url
        if self.provider == "auto":
            # A single-key setup has only the shared endpoint to go on.
            return self.base_url
        return ""  # rules never talks to an endpoint

    @property
    def wants_live(self) -> bool:
        """True when a live model was asked for, regardless of whether it works."""

        return self.provider in ("auto", "anthropic", "openai")

    def missing_reason(self) -> str | None:
        """Why this configuration cannot run, or ``None`` if it can."""

        if not self.wants_live or self.api_key:
            return None
        if self.provider == "auto":
            return "INSIGHT_API_KEY (or a provider-specific key) is not set"
        var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        return f"{var} is not set"

    def to_payload(self) -> dict[str, Any]:
        """Serialisable view. Never includes key material."""

        return {
            "provider": self.provider,
            "model": self.model,
            "live": self.is_live,
            "base_url": self.resolved_base_url,
            "anthropic_base_url": self.anthropic_base_url,
            "openai_base_url": self.openai_base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "has_anthropic_key": bool(self.anthropic_api_key),
            "has_openai_key": bool(self.openai_api_key),
            "source": self.source,
        }


def _env(name: str, default: str = "") -> str:
    return os.environ.get(ENV_PREFIX + name, default)


def _env_any(*names: str, default: str = "") -> str:
    """First non-empty value among the given variables.

    Used for endpoints, which have two conventions in the wild: this project's
    ``INSIGHT_`` prefix, and the unprefixed ``ANTHROPIC_BASE_URL`` that relay
    services and SDKs already document. Both are accepted so a user copying a
    gateway's own instructions still works.
    """

    for name in names:
        value = os.environ.get(ENV_PREFIX + name) or os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def load_dotenv_file(path: Path | None = None, *, override: bool = False) -> Path | None:
    """Load ``.env`` into ``os.environ``.

    Uses ``python-dotenv`` when available and falls back to a minimal parser so
    the project does not hard-depend on it. Existing environment variables win
    by default, matching the usual twelve-factor expectation.
    """

    target = path or DEFAULT_DOTENV
    if not target.exists():
        return None

    try:
        from dotenv import load_dotenv

        load_dotenv(target, override=override)
        return target
    except ImportError:
        pass

    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    return target


def load_config(*, dotenv: Path | None = None, provider: str | None = None) -> LLMConfig:
    """Build the configuration from ``.env`` and the process environment."""

    loaded = load_dotenv_file(dotenv)
    source = {ENV_PREFIX + "provider": "env"} if loaded else {}

    resolved = provider or _env("PROVIDER", "auto").strip().lower() or "auto"
    if resolved not in PROVIDERS:
        resolved = "rules"

    model = _env("MODEL").strip() or DEFAULT_MODELS.get(resolved, "")

    return LLMConfig(
        provider=resolved,
        model=model,
        # A single generic key covers the common case: one relay, one key,
        # protocol unknown. Provider-specific keys still win.
        anthropic_api_key=(os.environ.get("ANTHROPIC_API_KEY", "").strip() or _env_any("API_KEY")),
        openai_api_key=(os.environ.get("OPENAI_API_KEY", "").strip() or _env_any("API_KEY")),
        base_url=_env_any("BASE_URL"),
        anthropic_base_url=_env_any("ANTHROPIC_BASE_URL"),
        openai_base_url=_env_any("OPENAI_BASE_URL"),
        max_tokens=int(_env("MAX_TOKENS", "2048") or 2048),
        temperature=float(_env("TEMPERATURE", "0") or 0),
        source=source,
    )


def build_client(config: LLMConfig) -> Any:
    """Instantiate the client for the configured provider.

    Raises a clear, actionable error rather than an import traceback, because
    the two common failures are "no key" and "SDK not installed".
    """

    reason = config.missing_reason()
    if reason:
        raise RuntimeError(f"{reason}. Copy .env.example to .env and fill it in.")

    if config.provider == "auto":
        from insight_agent.core.providers.auto import build_auto_client

        return build_auto_client(config)

    if config.provider == "anthropic":
        from insight_agent.core.providers.anthropic import AnthropicClient

        return AnthropicClient(
            model=config.model or None,
            api_key=config.anthropic_api_key,
            max_tokens=config.max_tokens,
            base_url=config.resolved_base_url or None,
        )

    if config.provider == "openai":
        from insight_agent.core.providers.openai_client import OpenAIClient

        return OpenAIClient(
            model=config.model or None,
            api_key=config.openai_api_key,
            base_url=config.resolved_base_url or None,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    raise ValueError(f"provider {config.provider!r} has no client; use 'rules' instead")
