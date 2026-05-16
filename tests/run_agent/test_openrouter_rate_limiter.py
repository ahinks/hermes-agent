"""
Simplified integration test for OpenRouter rate limiter injection.
"""

import pytest
import sys
import os

# Add hermes-agent to path
sys.path.insert(0, "/home/alexanderh/.hermes/hermes-agent")

# Create a mock OpenAI client that stores kwargs
class TestOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.http_client = kwargs.get("http_client")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")

# Override the OpenAI proxy in run_agent
import run_agent
original_OpenAI = run_agent.OpenAI
run_agent.OpenAI = TestOpenAIClient

@pytest.fixture
def agent():
    """Create a minimal AIAgent instance for testing."""
    # Create instance without calling __init__
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.provider = "openrouter"
    agent._base_url_lower = "https://openrouter.ai/api/v1"
    agent.logger = run_agent.logging.getLogger("test")
    
    # Mock methods
    agent._is_openrouter_url = lambda: True
    agent._client_log_context = lambda: "test"
    agent._build_keepalive_http_client = lambda url: None
    
    return agent

def test_rate_limiter_injected_for_openrouter(agent):
    """Test that rate-limited http client is injected for OpenRouter."""
    client_kwargs = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key"
    }
    
    # Call _create_openai_client
    client = agent._create_openai_client(
        client_kwargs, reason="test", shared=False
    )
    
    # Verify the client has an http_client
    assert client.http_client is not None, "No http_client injected"
    
    # Verify it's a rate-limited client (check the marker)
    assert hasattr(client.http_client, '_rate_limited'), \
        "http_client missing _rate_limited marker"
    assert client.http_client._rate_limited is True, \
        "http_client is not marked as rate-limited"
    
    # Verify the limiter is attached
    assert hasattr(client.http_client, '_limiter'), \
        "http_client missing _limiter attribute"
    from agent.rate_limiter import OpenRouterRateLimiter
    assert isinstance(client.http_client._limiter, OpenRouterRateLimiter), \
        "http_client._limiter is not an OpenRouterRateLimiter"

def test_no_rate_limiter_for_other_providers(agent):
    """Test that rate limiter is NOT injected for non-OpenRouter providers."""
    agent.provider = "minimax"
    agent._is_openrouter_url = lambda: False
    
    client_kwargs = {
        "base_url": "http://localhost:8080/v1",
        "api_key": "test-key"
    }
    
    client = agent._create_openai_client(
        client_kwargs, reason="test", shared=False
    )
    
    # Either no http_client, or it's not our rate-limited one
    if client.http_client is not None:
        from agent.rate_limiter import RateLimitedTransport
        assert not isinstance(client.http_client.transport, RateLimitedTransport), \
            "Rate limiter incorrectly injected for non-OpenRouter provider"

# Cleanup
def teardown_module():
    run_agent.OpenAI = original_OpenAI

if __name__ == "__main__":
    pytest.main([__file__, "-v"])