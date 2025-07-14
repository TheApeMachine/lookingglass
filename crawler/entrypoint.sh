#!/bin/bash

echo "Starting LookingGlass crawler with enhanced stealth..."

# Note: Patchright handles stealth automatically, no manual patching needed

# Ensure proxy file exists (empty is fine)
echo "Initializing proxy system..."
touch proxies.txt

# Start the proxy refresher script in the background
echo "Starting proxy refresher in the background..."
python3 -u lookingglass_crawler/proxy_refresher.py &

# Set default mode if not provided
CRAWL_MODE=${CRAWL_MODE:-image}

echo "Starting crawler in mode: $CRAWL_MODE"
echo "Target URL: $START_URL"

# Start the spider
exec scrapy crawl media_spider 