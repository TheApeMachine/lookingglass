import os
import time
import urllib.parse
from urllib.parse import urljoin, urlparse
import requests
from retinaface import RetinaFace
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging
from proxy_manager import ProxyManager

class FaceCrawler:
    def __init__(self, start_url, output_dir="crawled_faces", max_pages=100):
        self.start_url = start_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.visited_urls = set()
        self.url_queue = [start_url]
        self.proxy_manager = ProxyManager(max_retries=3, timeout=5.0)
        self.setup_logging()
        self.setup_output_dir()

    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_browser(self):
        """Configure and initialize the headless Chrome browser with proxy."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Configure proxy
        chrome_options = self.proxy_manager.configure_chrome_options(chrome_options)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(30)

    def setup_output_dir(self):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_domain(self, url):
        """Extract the domain from a URL."""
        return urlparse(url).netloc

    def normalize_url(self, url):
        """Normalize URL by removing fragments and query parameters."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def make_path_from_url(self, url, image_name):
        """Create a file system path that mirrors the URL structure."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if not path_parts[0]:
            path_parts = [parsed.netloc]
        else:
            path_parts.insert(0, parsed.netloc)
        
        dir_path = os.path.join(self.output_dir, *path_parts[:-1])
        os.makedirs(dir_path, exist_ok=True)
        
        return os.path.join(dir_path, image_name)

    def download_image(self, image_url, source_url):
        """Download an image and check if it contains faces using RetinaFace."""
        def _download():
            response = requests.get(
                image_url, 
                timeout=10,
                proxies={'http': self.proxy_manager.current_proxy, 'https': self.proxy_manager.current_proxy}
            )
            if response.status_code != 200:
                raise Exception(f"Failed to download image: HTTP {response.status_code}")
            return response.content

        try:
            # Download image with retry logic
            image_content = self.proxy_manager.with_retry(_download)

            # Generate a filename from the URL
            image_name = os.path.basename(urlparse(image_url).path)
            if not image_name:
                image_name = f"image_{hash(image_url)}.jpg"

            # Create temporary file to check for faces
            temp_path = os.path.join(self.output_dir, "temp_" + image_name)
            with open(temp_path, 'wb') as f:
                f.write(image_content)

            # Check for faces using RetinaFace
            try:
                faces = RetinaFace.detect_faces(temp_path)
                
                if faces and isinstance(faces, dict):  # RetinaFace found faces
                    permanent_path = self.make_path_from_url(source_url, image_name)
                    os.rename(temp_path, permanent_path)
                    self.logger.info(f"Found {len(faces)} face(s) in {image_url}")
                    return permanent_path
                else:
                    os.remove(temp_path)
                    return None
            except Exception as e:
                self.logger.error(f"Error processing image {image_url}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None

        except Exception as e:
            self.logger.error(f"Error downloading {image_url}: {e}")
            return None

    def get_page_links(self, current_url):
        """Extract all links from the current page."""
        def _get_links():
            elements = self.driver.find_elements(By.TAG_NAME, "a")
            domain = self.get_domain(current_url)
            links = set()
            
            for element in elements:
                try:
                    href = element.get_attribute("href")
                    if href and href.startswith(("http://", "https://")):
                        if self.get_domain(href) == domain:
                            normalized_url = self.normalize_url(href)
                            if normalized_url not in self.visited_urls:
                                links.add(normalized_url)
                except Exception:
                    continue
            
            return links

        try:
            return self.proxy_manager.with_retry(_get_links)
        except Exception as e:
            self.logger.error(f"Error getting links from {current_url}: {e}")
            return set()

    def get_page_images(self, current_url):
        """Extract all image URLs from the current page."""
        def _get_images():
            elements = self.driver.find_elements(By.TAG_NAME, "img")
            images = set()
            
            for element in elements:
                try:
                    src = element.get_attribute("src")
                    if src:
                        absolute_url = urljoin(current_url, src)
                        if absolute_url.startswith(("http://", "https://")):
                            images.add(absolute_url)
                except Exception:
                    continue
            
            return images

        try:
            return self.proxy_manager.with_retry(_get_images)
        except Exception as e:
            self.logger.error(f"Error getting images from {current_url}: {e}")
            return set()

    def crawl(self):
        """Main crawling method."""
        pages_visited = 0
        
        try:
            while self.url_queue and pages_visited < self.max_pages:
                current_url = self.url_queue.pop(0)
                if current_url in self.visited_urls:
                    continue

                self.logger.info(f"\nVisiting {current_url}")
                
                try:
                    # Setup browser with new proxy
                    self.setup_browser()
                    
                    def _load_page(url):
                        self.driver.get(url)
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                    
                    # Load page with retry logic
                    self.proxy_manager.with_retry(_load_page, current_url)
                    
                    # Mark as visited
                    self.visited_urls.add(current_url)
                    pages_visited += 1

                    # Get all images
                    images = self.get_page_images(current_url)
                    self.logger.info(f"Found {len(images)} images")
                    
                    # Process images
                    for img_url in images:
                        self.download_image(img_url, current_url)
                    
                    # Get new links and add to queue
                    new_links = self.get_page_links(current_url)
                    self.url_queue.extend(list(new_links - self.visited_urls))
                    self.logger.info(f"Added {len(new_links)} new links to queue")
                    
                    # Small delay to be nice to servers
                    time.sleep(2)

                except Exception as e:
                    self.logger.error(f"Error processing {current_url}: {e}")
                    continue
                finally:
                    self.driver.quit()

        except Exception as e:
            self.logger.error(f"Crawling error: {e}")
        finally:
            self.logger.info("\nCrawling completed!")
            self.logger.info(f"Visited {pages_visited} pages")
            self.logger.info(f"Found faces are saved in {self.output_dir}")

if __name__ == "__main__":
    START_URL = os.getenv("START_URL", "https://example.com")
    crawler = FaceCrawler(START_URL, output_dir="crawled_faces", max_pages=100)
    crawler.crawl() 
