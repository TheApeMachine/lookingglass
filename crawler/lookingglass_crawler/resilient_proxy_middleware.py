import random
import logging
from scrapy.exceptions import NotConfigured
from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware

logger = logging.getLogger(__name__)

class ResilientProxyMiddleware:
    """
    A fault-tolerant proxy middleware that never blocks the scraper.
    If proxy fails, immediately continues without proxy.
    """
    
    def __init__(self, settings):
        self.proxy_list = []
        self.enabled = False
        
        # Load proxy list if available
        proxy_file = settings.get('PROXY_LIST', 'proxies.txt')
        try:
            with open(proxy_file, 'r') as f:
                self.proxy_list = [line.strip() for line in f if line.strip()]
            
            if self.proxy_list:
                self.enabled = True
                print(f"✓ Loaded {len(self.proxy_list)} proxies")
            else:
                print("ℹ️  No proxies found, disabled")
        except (FileNotFoundError, IOError) as e:
            print(f"ℹ️  Could not load proxy file: {e}")
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_request(self, request, spider):
        """
        Apply proxy to request if available, but never fail.
        """
        if not self.enabled or not self.proxy_list:
            return None
            
        # Try to get a random proxy
        try:
            proxy = random.choice(self.proxy_list)
            if proxy:
                request.meta['proxy'] = proxy
        except Exception as e:
            # Never fail - just log and continue without proxy
            logger.warning(f"Proxy selection failed, continuing without proxy: {e}")
            
        return None
    
    def process_exception(self, request, exception, spider):
        """
        If proxy-related exception occurs, retry without proxy.
        """
        if 'proxy' in request.meta:
            logger.warning(f"Proxy failed: {exception}. Retrying without proxy.")
            # Remove proxy and retry
            request.meta.pop('proxy', None)
            return request.replace(dont_filter=True)
        
        return None
    
    def refresh_proxy_list(self):
        """
        Refresh proxy list from file (called periodically by background task).
        """
        proxy_file = 'proxies.txt'
        try:
            with open(proxy_file, 'r') as f:
                new_proxies = [line.strip() for line in f if line.strip()]
            
            if new_proxies != self.proxy_list:
                self.proxy_list = new_proxies
                self.enabled = len(self.proxy_list) > 0
                print(f"🔄 Proxy list refreshed: {len(self.proxy_list)} proxies")
        except (FileNotFoundError, IOError):
            # Silently handle - proxy file might not exist yet
            pass 