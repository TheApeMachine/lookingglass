#!/usr/bin/env python3
"""
Test script to verify the proxy system is working correctly.
"""
import os
import time
import subprocess
import signal
import sys
from lookingglass_crawler.proxy_refresher import refresh_proxies

def test_proxy_generation():
    """Test that proxy refresher can generate proxies."""
    print("Testing proxy generation...")
    
    # Remove existing proxy file if it exists
    if os.path.exists('proxies.txt'):
        os.remove('proxies.txt')
    
    # Generate proxies
    refresh_proxies()
    
    # Check if file was created and has content
    if os.path.exists('proxies.txt'):
        with open('proxies.txt', 'r') as f:
            proxies = f.read().strip().split('\n')
            proxies = [p for p in proxies if p.strip()]
            print(f"✓ Generated {len(proxies)} proxies")
            for proxy in proxies[:3]:  # Show first 3
                print(f"  - {proxy}")
            return True
    else:
        print("✗ No proxies.txt file generated")
        return False

def test_proxy_middleware():
    """Test that middleware can read the proxy file."""
    print("\nTesting proxy middleware...")
    
    try:
        from lookingglass_crawler.resilient_proxy_middleware import ResilientProxyMiddleware
        from scrapy.utils.test import get_crawler
        
        # Create a mock crawler with settings
        crawler = get_crawler(settings_dict={
            'PROXY_LIST': 'proxies.txt'
        })
        
        # Initialize middleware
        middleware = ResilientProxyMiddleware.from_crawler(crawler)
        
        if middleware.enabled and middleware.proxy_list:
            print(f"✓ Middleware loaded {len(middleware.proxy_list)} proxies")
            return True
        else:
            print("✗ Middleware not enabled or no proxies loaded")
            return False
            
    except Exception as e:
        print(f"✗ Middleware test failed: {e}")
        return False

def test_background_refresher():
    """Test that background refresher starts properly."""
    print("\nTesting background proxy refresher...")
    
    try:
        # Start the refresher in background
        proc = subprocess.Popen([
            sys.executable, '-u', 
            'lookingglass_crawler/proxy_refresher.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Let it run for a few seconds
        time.sleep(5)
        
        # Check if it's still running
        if proc.poll() is None:
            print("✓ Background refresher started successfully")
            # Kill the process
            proc.terminate()
            proc.wait()
            return True
        else:
            stdout, stderr = proc.communicate()
            print(f"✗ Background refresher failed: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"✗ Background refresher test failed: {e}")
        return False

def main():
    """Run all proxy system tests."""
    print("🔍 Testing LookingGlass Proxy System\n")
    
    tests = [
        test_proxy_generation,
        test_proxy_middleware,
        test_background_refresher
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All proxy system tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 