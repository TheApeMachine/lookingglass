import time
import logging
import threading
from fp.fp import FreeProxy

# Configure logging - error level only (user preference)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROXY_FILE = 'proxies.txt'
REFRESH_INTERVAL_SECONDS = 300  # 5 minutes - more frequent updates

def refresh_proxies():
    """Fetches a list of proxies, checks them, and writes the good ones to a file."""
    logger.debug("Starting proxy refresh cycle...")
    good_proxies = set()
    attempts = 0
    max_attempts = 30  # Reduced attempts for faster cycles
    max_time = 60  # Maximum 60 seconds per refresh cycle
    start_time = time.time()
    
    # Try to get at least 3 good proxies, but limit time and attempts
    while len(good_proxies) < 3 and attempts < max_attempts:
        # Check if we've exceeded max time
        if time.time() - start_time > max_time:
            break
            
        attempts += 1
        try:
            proxy_str = FreeProxy(timeout=1, rand=True).get()
            # Basic validation
            if proxy_str and proxy_str.startswith('http'):
                good_proxies.add(proxy_str)
        except Exception as e:
            continue

    if len(good_proxies) > 0:
        print(f"✓ Found {len(good_proxies)} working proxies")
        try:
            with open(PROXY_FILE, 'w') as f:
                for proxy in good_proxies:
                    f.write(f"{proxy}\n")
        except Exception as e:
            logger.warning(f"Failed to write proxy file: {e}")
    else:
        logger.warning(f"No working proxies found after {attempts} attempts. Keeping existing {PROXY_FILE}.")
        # Don't overwrite existing file if no new proxies found
        # Just ensure file exists (empty is ok)
        try:
            with open(PROXY_FILE, 'a') as f:
                pass  # Just ensure file exists
        except Exception as e:
            logger.warning(f"Failed to touch proxy file: {e}")

def continuous_proxy_refresh():
    """Continuously refresh proxies in background thread."""
    print("🔄 Starting continuous proxy refresh in background")
    while True:
        try:
            refresh_proxies()
            time.sleep(REFRESH_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Proxy refresh failed: {e}. Retrying in {REFRESH_INTERVAL_SECONDS} seconds.")
            time.sleep(REFRESH_INTERVAL_SECONDS)

if __name__ == "__main__":
    continuous_proxy_refresh() 