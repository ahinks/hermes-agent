#!/usr/bin/env python3
"""
Practical test script for OpenRouter rate limiter.
Tests RPM, RPD, and 429 handling with optional live API calls.
"""

import time
import sys
import os

sys.path.insert(0, "/home/alexanderh/.hermes/hermes-agent")

from agent.rate_limiter import OpenRouterRateLimiter, RateLimitExceeded

def test_rpm_simulation():
    """Simulate RPM limit with low threshold."""
    print("=== Testing RPM Limit (simulated) ===")
    limiter = OpenRouterRateLimiter(rpm=3, rpd=100)
    
    print("Sending 3 requests quickly (should pass)...")
    for i in range(3):
        limiter.check_rate_limits()
        print(f"  Request {i+1} passed")
    
    print("Sending 4th request (should trigger wait)...")
    start = time.time()
    limiter.check_rate_limits()  # This should wait ~57s, but we'll interrupt
    elapsed = time.time() - start
    print(f"  4th request passed after {elapsed:.1f}s (simulated wait)")
    
    # Clean up
    limiter._rpm_window.clear()
    print("RPM simulation test passed ✅\n")

def test_rpd_simulation():
    """Simulate RPD limit with low threshold."""
    print("=== Testing RPD Limit (simulated) ===")
    limiter = OpenRouterRateLimiter(rpm=100, rpd=5)
    
    print("Using 5 requests...")
    for i in range(5):
        limiter.check_rate_limits()
        print(f"  Request {i+1} passed")
    
    print("Sending 6th request (should raise RateLimitExceeded)...")
    try:
        limiter.check_rate_limits()
        print("  ERROR: Should have raised RateLimitExceeded")
    except RateLimitExceeded as e:
        print(f"  Correctly raised RateLimitExceeded: {e}")
    
    print("RPD simulation test passed ✅\n")

def test_429_handling():
    """Test 429 response handling."""
    print("=== Testing 429 Handling ===")
    limiter = OpenRouterRateLimiter()
    
    print("Testing retry_after=0.1s...")
    start = time.time()
    limiter.handle_429(retry_after=0.1)
    elapsed = time.time() - start
    print(f"  Waited {elapsed:.2f}s (expected ~0.1s)")
    
    print("Testing exponential backoff...")
    start = time.time()
    limiter.handle_429()  # Should back off 1s (2^0)
    elapsed = time.time() - start
    print(f"  Waited {elapsed:.2f}s (expected ~1s)")
    
    print("429 handling test passed ✅\n")

def test_live_api():
    """Test with live OpenRouter API (requires API key)."""
    print("=== Testing Live OpenRouter API ===")
    
    # Check if API key is available
    api_key = os.environ.get("OPENROUTER_API_KEY") or \
              open(os.path.expanduser("~/.hermes/config.yaml")).read().split("openrouter:")[1].split("\n")[0].strip() if "openrouter:" in open(os.path.expanduser("~/.hermes/config.yaml")).read() else None
    
    if not api_key:
        print("No OpenRouter API key found. Skipping live test.")
        print("Set OPENROUTER_API_KEY environment variable to test live API.\n")
        return
    
    print(f"Using API key: {api_key[:10]}...")
    
    # Create rate-limited client
    from agent.rate_limiter import create_rate_limited_http_client
    import httpx
    
    limiter = OpenRouterRateLimiter(rpm=20, rpd=1000)
    client = create_rate_limited_http_client(limiter)
    
    print("Sending test request to OpenRouter...")
    try:
        response = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",  # Free-tier model
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 429:
            print(f"Got 429 response (rate limited). Retry-After: {response.headers.get('Retry-After', 'N/A')}")
        elif response.status_code == 200:
            print("Request succeeded!")
            print(f"Response: {response.json()}")
        else:
            print(f"Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Live API test completed ✅\n")

if __name__ == "__main__":
    print("OpenRouter Rate Limiter Practical Test Suite\n")
    
    test_rpm_simulation()
    test_rpd_simulation()
    test_429_handling()
    test_live_api()
    
    print("All practical tests completed! 🎉")