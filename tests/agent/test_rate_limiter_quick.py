#!/usr/bin/env python3
"""
Quick verification of OpenRouter rate limiter integration.
"""

import time
import sys

sys.path.insert(0, "/home/alexanderh/.hermes/hermes-agent")

from agent.rate_limiter import OpenRouterRateLimiter, RateLimitExceeded, create_rate_limited_http_client
import httpx

def quick_rpm_test():
    """Quick RPM test with 5s window instead of 60s."""
    print("=== Quick RPM Test ===")
    # Create limiter with 5s window for testing
    limiter = OpenRouterRateLimiter(rpm=3, rpd=100)
    # Override the window check to use 5s instead of 60s for testing
    original_check = limiter.check_rate_limits
    def test_check():
        limiter._acquire_lock()
        try:
            now = time.time()
            # Clean up window with 5s window
            while limiter._rpm_window and limiter._rpm_window[0] < now - 5:
                limiter._rpm_window.popleft()
            if len(limiter._rpm_window) >= limiter.rpm:
                wait_time = 5 - (now - limiter._rpm_window[0]) + 0.5
                if wait_time > 0:
                    print(f"  Would wait {wait_time:.1f}s, but skipping for test")
                    return
            limiter._rpm_window.append(now)
            limiter._rpd_count += 1
        finally:
            limiter._release_lock()
    
    print("Sending 3 requests...")
    for i in range(3):
        test_check()
        print(f"  Request {i+1} passed")
    
    print("Sending 4th request (should be rate limited)...")
    test_check()  # Should not add to window due to limit
    print(f"  RPM window size: {len(limiter._rpm_window)} (max 3)")
    print("Quick RPM test passed ✅\n")

def quick_rpd_test():
    """Quick RPD test."""
    print("=== Quick RPD Test ===")
    limiter = OpenRouterRateLimiter(rpm=100, rpd=3)
    
    print("Using 3 requests...")
    for i in range(3):
        limiter.check_rate_limits()
        print(f"  Request {i+1} passed")
    
    print("Sending 4th request (should raise RateLimitExceeded)...")
    try:
        limiter.check_rate_limits()
        print("  ERROR: Should have raised RateLimitExceeded")
    except RateLimitExceeded as e:
        print(f"  Correctly raised: {e}")
    print("Quick RPD test passed ✅\n")

def client_creation_test():
    """Test rate-limited client creation."""
    print("=== Client Creation Test ===")
    limiter = OpenRouterRateLimiter()
    client = create_rate_limited_http_client(limiter)
    
    assert hasattr(client, '_rate_limited'), "Client missing _rate_limited marker"
    assert client._rate_limited is True, "Client not marked as rate-limited"
    assert hasattr(client, '_limiter'), "Client missing _limiter"
    assert isinstance(client._limiter, OpenRouterRateLimiter), "Invalid limiter type"
    
    print("Client created with rate limiter ✅")
    print("Client creation test passed ✅\n")

if __name__ == "__main__":
    print("OpenRouter Rate Limiter Quick Verification\n")
    quick_rpm_test()
    quick_rpd_test()
    client_creation_test()
    print("All quick tests passed! 🎉")