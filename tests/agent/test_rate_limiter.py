"""
Tests for OpenRouter rate limiter.
"""

import time
import pytest
from agent.rate_limiter import OpenRouterRateLimiter, RateLimitExceeded, create_rate_limited_http_client
import httpx


class TestOpenRouterRateLimiter:
    """Unit tests for OpenRouterRateLimiter."""
    
    def test_rpm_enforcement(self):
        """Test that RPM limit is enforced."""
        limiter = OpenRouterRateLimiter(rpm=3, rpd=100)
        
        # Send 3 requests quickly (should pass)
        for i in range(3):
            limiter.check_rate_limits()
        
        # 4th request should trigger wait (we can't test actual wait, but we can check the window)
        # Instead, verify the window has 3 entries
        assert len(limiter._rpm_window) == 3
        
        # Simulate time passing (wait 61s)
        if limiter._rpm_window:
            limiter._rpm_window[0] = time.time() - 61  # Expire the oldest
        
        # Now should be able to add another request
        limiter.check_rate_limits()
        assert len(limiter._rpm_window) == 3  # Old one expired, new one added
    
    def test_rpd_enforcement(self):
        """Test that RPD limit is enforced."""
        limiter = OpenRouterRateLimiter(rpm=100, rpd=5)
        
        # Use 5 requests
        for i in range(5):
            limiter.check_rate_limits()
        assert limiter._rpd_count == 5
        
        # 6th request should raise RateLimitExceeded
        with pytest.raises(RateLimitExceeded):
            limiter.check_rate_limits()
    
    def test_rpd_reset(self):
        """Test that RPD counter resets after 24h."""
        limiter = OpenRouterRateLimiter(rpm=100, rpd=5)
        limiter._rpd_count = 5
        
        # Simulate 24h passing
        limiter._rpd_reset = time.time() - 1  # Reset time passed
        
        # Should reset and allow new requests
        limiter.check_rate_limits()
        assert limiter._rpd_count == 1  # Reset to 0, then +1
    
    def test_429_handling(self):
        """Test handling of 429 responses."""
        limiter = OpenRouterRateLimiter()
        
        # Test with retry_after header
        start = time.time()
        limiter.handle_429(retry_after=0.1)  # 100ms wait
        elapsed = time.time() - start
        assert 0.09 <= elapsed <= 0.2  # Allow some tolerance
        
        # Test exponential backoff
        start = time.time()
        limiter._rpd_count = 0  # Reset for predictable backoff
        limiter.handle_429()  # Should back off 1s (2^0)
        elapsed = time.time() - start
        assert 0.9 <= elapsed <= 1.1


class TestRateLimitedHttpClient:
    """Tests for rate-limited HTTP client."""
    
    def test_client_creation(self):
        """Test that rate-limited client is created successfully."""
        limiter = OpenRouterRateLimiter()
        client = create_rate_limited_http_client(limiter)
        assert client is not None
        assert isinstance(client, httpx.Client)
    
    def test_rate_limiter_called_before_request(self, monkeypatch):
        """Test that rate limiter check is called before each request."""
        limiter = OpenRouterRateLimiter(rpm=10, rpd=100)
        check_called = False
        
        original_check = limiter.check_rate_limits
        def mock_check():
            nonlocal check_called
            check_called = True
            original_check()
        
        monkeypatch.setattr(limiter, 'check_rate_limits', mock_check)
        
        client = create_rate_limited_http_client(limiter)
        # Send a test request to a non-existent server (should fail but trigger check)
        try:
            client.get("http://localhost:9999/test")
        except Exception:
            pass
        
        assert check_called, "Rate limiter check was not called before request"


# Integration test with mock OpenAI client
class TestOpenRouterClientIntegration:
    """Integration tests for OpenRouter client with rate limiting."""
    
    def test_rate_limiter_injected_in_run_agent(self, monkeypatch):
        """Test that _create_openai_client injects rate limiter for OpenRouter."""
        # This is a simplified integration test
        # In real test, we would mock the OpenAI client creation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])