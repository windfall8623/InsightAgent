"""Model provider adapters."""

__all__ = [
    "AnthropicClient",
    "CircuitBreaker",
    "CircuitBreakerState",
    "MultiProviderGateway",
    "ProviderEndpoint",
    "AllProvidersFailedError",
    "PromptCacheOptimizer",
    "PrefixCacheTracker",
]


def __getattr__(name: str) -> object:
    if name == "AnthropicClient":
        from insight_agent.core.providers.anthropic import AnthropicClient

        return AnthropicClient
    if name in (
        "CircuitBreaker",
        "CircuitBreakerState",
        "MultiProviderGateway",
        "ProviderEndpoint",
        "AllProvidersFailedError",
        "PromptCacheOptimizer",
        "PrefixCacheTracker",
    ):
        from insight_agent.core.providers import gateway

        return getattr(gateway, name)
    raise AttributeError(name)
