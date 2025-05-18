from fp.fp import FreeProxy
from selenium.webdriver.chrome.options import Options
import time
import logging

class ProxyAcquisitionError(Exception):
    pass

class ProxyManager:
    def __init__(self, max_retries=3, timeout=1.0, countries=None):
        self.max_retries = max_retries
        self.timeout = timeout
        self.countries = countries or ['US', 'GB', 'DE', 'FR']  # Default to some reliable countries
        self.current_proxy = None
        self.failed_proxies = set()
        self.logger = logging.getLogger(__name__)

    def get_new_proxy(self):
        """Get a new working proxy that hasn't failed before."""
        for _ in range(self.max_retries):
            try:
                # Try to get a proxy with our specified parameters
                proxy = FreeProxy(
                    country_id=self.countries,
                    timeout=self.timeout,
                    https=True,  # We want HTTPS proxies for security
                    anonym=True  # Only use anonymous proxies
                ).get()
                
                # Check if this proxy has failed before
                if proxy not in self.failed_proxies:
                    self.current_proxy = proxy
                    self.logger.info(f"Successfully obtained new proxy: {proxy}")
                    return proxy
            except Exception as e:
                self.logger.warning(f"Error getting new proxy: {str(e)}")
                time.sleep(1)  # Small delay before retry
        
        # If we get here, we couldn't get a new working proxy
        raise ProxyAcquisitionError("Unable to obtain a working proxy after multiple attempts")

    def mark_proxy_failed(self):
        """Mark the current proxy as failed and get a new one."""
        if self.current_proxy:
            self.failed_proxies.add(self.current_proxy)
            self.logger.warning(f"Marked proxy as failed: {self.current_proxy}")
        return self.get_new_proxy()

    def configure_chrome_options(self, chrome_options=None):
        """Configure Chrome options with the current proxy."""
        if chrome_options is None:
            chrome_options = Options()

        if self.current_proxy:
            # Remove http:// or https:// from proxy string
            proxy_addr = self.current_proxy.replace('http://', '').replace('https://', '')
            chrome_options.add_argument(f'--proxy-server={proxy_addr}')
            
        # Add additional options for proxy handling
        chrome_options.add_argument('--proxy-bypass-list=<-loopback>')  # Bypass proxy for localhost
        chrome_options.add_argument('--ignore-certificate-errors')  # Handle SSL issues with proxies
        
        return chrome_options

    def with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic on proxy failure."""
        retries = 0
        while retries < self.max_retries:
            try:
                if not self.current_proxy:
                    self.get_new_proxy()
                
                return func(*args, **kwargs)
            
            except Exception as e:
                retries += 1
                self.logger.warning(f"Attempt {retries} failed with error: {str(e)}")
                
                if retries < self.max_retries:
                    self.mark_proxy_failed()
                    time.sleep(1)  # Small delay before retry
                else:
                    raise ProxyAcquisitionError(f"Operation failed after {self.max_retries} attempts with different proxies") 