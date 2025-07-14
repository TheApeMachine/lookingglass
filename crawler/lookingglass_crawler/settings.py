import os

BOT_NAME = 'lookingglass_crawler'

SPIDER_MODULES = ['lookingglass_crawler.spiders']
NEWSPIDER_MODULE = 'lookingglass_crawler.spiders'

# --- Playwright Settings ---
# Enable the Playwright downloader handler with Patchright
DOWNLOAD_HANDLERS = {
    "http": "scrapy_patchright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_patchright.handler.ScrapyPlaywrightDownloadHandler",
}

# Set the correct asynchronous event loop for Twisted
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# --- Custom Settings ---
START_URL = os.getenv('START_URL', 'https://www.flickr.com/search/?text=people/')

# --- MinIO Settings ---
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_USER = os.getenv('MINIO_USER', 'miniouser')
MINIO_PASSWORD = os.getenv('MINIO_PASSWORD', 'miniopassword')
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'scraped')

# --- Obey robots.txt rules (disabled for more permissive crawling) ---
ROBOTSTXT_OBEY = False

# --- Configure item pipelines ---
# The single pipeline handles both images and videos and uploads them to MinIO.
ITEM_PIPELINES = {
    'lookingglass_crawler.pipelines.MinioMediaPipeline': 100,
}

# --- DeltaFetch Settings ---
SPIDER_MIDDLEWARES = {
     'scrapy_deltafetch.DeltaFetch': 100,
}

DELTAFETCH_ENABLED = True

# Enhanced Resilience Settings for High-Memory System with Anti-Blocking
# More aggressive retry configuration
RETRY_TIMES = 1
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403]  # Include 403 in retries

# Don't stop on HTTP errors - continue crawling
HTTPERROR_ALLOWED_CODES = [403, 404, 429, 500, 502, 503, 504]

# Smart delays - more respectful to avoid blocks
DOWNLOAD_DELAY = 3  # Increased delay between requests
RANDOMIZE_DOWNLOAD_DELAY = 0.5  # 0.5 * to 1.5 * DOWNLOAD_DELAY (1.5s to 4.5s)
DOWNLOAD_TIMEOUT = 60  # Longer timeout for large files

# Reduced concurrency to avoid overwhelming sites
CONCURRENT_REQUESTS = 100  # Reduced from 16
CONCURRENT_REQUESTS_PER_DOMAIN = 1  # Reduced from 3
REACTOR_THREADPOOL_MAXSIZE = 20

# Smart AutoThrottle - backs off automatically when sites get unhappy
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 15  # Back off significantly if needed
AUTOTHROTTLE_TARGET_CONCURRENCY = 4.0  # Conservative target
AUTOTHROTTLE_DEBUG = False

# Advanced anti-blocking measures
COOKIES_ENABLED = True
REDIRECT_ENABLED = True

# Enhanced memory settings - no limits with 1TB RAM
MEMUSAGE_ENABLED = False  # Disable memory monitoring entirely

# Enable response caching for better performance
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600  # Cache for 1 hour
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 505, 500, 429, 403]

SCHEDULER_PRIORITY_QUEUE = "scrapy.pqueues.DownloaderAwarePriorityQueue"

# Anti-blocking middleware stack
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
    'scrapy_fake_useragent.middleware.RetryUserAgentMiddleware': 401,
    'lookingglass_crawler.resilient_proxy_middleware.ResilientProxyMiddleware': 110,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 120,
}

# --- scrapy-fake-useragent settings ---
FAKEUSERAGENT_PROVIDERS = [
    'scrapy_fake_useragent.providers.FakeUserAgentProvider',  # this is the first provider we'll try
    'scrapy_fake_useragent.providers.FakerProvider',  # if FakeUserAgentProvider fails, we'll use faker to generate a user-agent string for us
    'scrapy_fake_useragent.providers.FixedUserAgentProvider',  # fall back to USER_AGENT value
]
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


# --- Proxy List ---
# A file with one proxy per line, e.g., http://host:port
PROXY_LIST = 'proxies.txt'

# User agent rotation is now handled by RotateUserAgentMiddleware
# No need for static user agent list

# Logging configuration - error level only (user preference)
LOG_LEVEL = 'ERROR'  # Only show errors
LOG_FILE = None  # Log to stdout

# Duplication filter
DUPEFILTER_DEBUG = False

# Don't close spider on various conditions - keep it running
CLOSESPIDER_ERRORCOUNT = 0  # Don't close on errors
CLOSESPIDER_PAGECOUNT = 0   # Don't close on page count
CLOSESPIDER_ITEMCOUNT = 0   # Don't close on item count
CLOSESPIDER_TIMEOUT = 0     # Don't close on timeout

# Playwright specific settings - Undetected version for maximum stealth
PLAYWRIGHT_BROWSER_TYPE = 'chromium'
PLAYWRIGHT_LAUNCH_OPTIONS = {
    'headless': True,
    'args': [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu',
        '--disable-web-security',
        '--disable-features=VizDisplayCompositor',
        '--disable-blink-features=AutomationControlled',  # Key anti-detection
        '--disable-automation',
        '--disable-extensions',
        '--disable-plugins',
        '--disable-ipc-flooding-protection',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--disable-background-timer-throttling',
        '--disable-field-trial-config',
        '--disable-back-forward-cache',
        '--disable-hang-monitor',
        '--disable-prompt-on-repost',
        '--disable-sync',
        '--metrics-recording-only',
        '--no-report-upload',
        '--use-mock-keychain',
    ]
}

# Enhanced Playwright settings for undetected browsing
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000  # 30 seconds
PLAYWRIGHT_CONTEXT_ARGS = {
    'viewport': {'width': 1920, 'height': 1080},
    'user_agent': None,  # Will be set by our RotateUserAgentMiddleware
    'java_script_enabled': True,
    'accept_downloads': False,
    'ignore_https_errors': True,
    'bypass_csp': True,
    'locale': 'en-US',
    'timezone_id': 'America/New_York',
    'geolocation': {'longitude': -74.006, 'latitude': 40.7128},  # New York
    'permissions': [],
    'color_scheme': 'light',
    'reduced_motion': 'no-preference',
    'forced_colors': 'none',
}

# Page-level settings for natural behavior
PLAYWRIGHT_PAGE_METHODS = [
    {'method': 'set_extra_http_headers', 'args': [{
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-platform-version': '"10.0.0"',
    }]},
    {'method': 'wait_for_timeout', 'args': [3000]},  # Wait for dynamic content
    {'method': 'evaluate', 'args': ['''
        // Remove webdriver property
        delete navigator.webdriver;
        
        // Override the plugins property to use a custom getter
        Object.defineProperty(navigator, 'plugins', {
            get: function() { return [1, 2, 3, 4, 5]; },
        });
        
        // Override the languages property to use a custom getter
        Object.defineProperty(navigator, 'languages', {
            get: function() { return ['en-US', 'en']; },
        });
        
        // Override the permissions property
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
    ''']},
] 