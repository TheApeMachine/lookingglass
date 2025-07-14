#!/usr/bin/env python3
"""
Test script to verify undetected Playwright is working correctly.
This script tests common detection vectors that anti-bot systems use.
"""

import asyncio
from playwright.async_api import async_playwright
import json

async def test_undetected_features():
    """Test various anti-detection features."""
    
    print("🔍 Testing undetected Playwright setup...")
    
    async with async_playwright() as p:
        # Launch browser with our anti-detection settings
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-automation',
                '--disable-extensions',
            ]
        )
        
        # Create context with realistic settings
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        page = await context.new_page()
        
        # Test 1: Check webdriver property
        webdriver_test = await page.evaluate('() => navigator.webdriver')
        print(f"✅ navigator.webdriver: {webdriver_test} (should be undefined)")
        
        # Test 2: Check plugins
        plugins_test = await page.evaluate('() => navigator.plugins.length')
        print(f"✅ navigator.plugins.length: {plugins_test} (should be > 0)")
        
        # Test 3: Check languages
        languages_test = await page.evaluate('() => navigator.languages')
        print(f"✅ navigator.languages: {languages_test}")
        
        # Test 4: Check permissions API
        try:
            permissions_test = await page.evaluate('''
                () => navigator.permissions.query({name: 'notifications'})
                    .then(result => result.state)
                    .catch(err => 'error: ' + err.message)
            ''')
            print(f"✅ permissions API: {permissions_test}")
        except Exception as e:
            print(f"⚠️ permissions API test failed: {e}")
        
        # Test 5: Visit a real detection test site
        print("\n🌐 Testing with real detection site...")
        try:
            await page.goto('https://bot.sannysoft.com/', timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Check if any red (detected) indicators
            red_elements = await page.query_selector_all('.red, [style*="color: red"], [style*="background: red"]')
            print(f"📊 Detection test results: {len(red_elements)} red flags detected")
            
            # Get the page title to confirm we reached the site
            title = await page.title()
            print(f"📄 Page title: {title}")
            
        except Exception as e:
            print(f"⚠️ Detection site test failed: {e}")
        
        await browser.close()
    
    print("\n✅ Undetected Playwright test completed!")

if __name__ == "__main__":
    asyncio.run(test_undetected_features()) 