import scrapy
import io
import time
from minio import Minio
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.http import HtmlResponse
from scrapy_patchright.page import PageMethod
from ..items import MediaItem
import os

class MediaSpider(CrawlSpider):
    name = 'media_spider'
    
    def __init__(self, *args, **kwargs):
        super(MediaSpider, self).__init__(*args, **kwargs)
        
        # Get start_urls from spider arguments or Scrapy settings.
        # This allows configuring the crawl target via docker-compose or command line.
        urls = os.getenv('START_URL', 'https://www.flickr.com')

        # Ensure that start_urls is a list of strings.
        if isinstance(urls, str):
            self.start_urls = [url.strip() for url in urls.split(',')]
        else:
            self.start_urls = urls
        
        media_extensions = {
            'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'mp4', 
            'webm', 'avi', 'mov', 'wmv', 'flv', 'mkv'
        }

        # Create a regex to match URLs ending with media extensions.
        # This is used in the 'allow' and 'deny' parameters of LinkExtractor.
        media_regex = r'\.(' + '|'.join(media_extensions) + r')$'

        self.rules = (
            # Rule 1: media files
            Rule(
                LinkExtractor(
                    allow=media_regex,
                    tags=('a', 'img', 'video', 'source'),
                    attrs=('href', 'src', 'poster', 'srcset', 'data-src', 'data-srcset'),
                    canonicalize=True,
                    unique=True
                ),
                callback='parse_item',
                follow=False  # Do not follow media links
            ),
            # Rule 2: Process regular HTML pages (upload to MinIO) and keep crawling.
            Rule(
                LinkExtractor(
                    deny=media_regex,
                    canonicalize=True,
                    unique=True
                ),
                callback='parse_page',
                follow=True
            ),
        )

        # Re-compile the rules after defining them in __init__
        super(MediaSpider, self)._compile_rules()

        # MinIO setup for page HTML upload
        self.minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.getenv('MINIO_USER', 'miniouser'),
            secret_key=os.getenv('MINIO_PASSWORD', 'miniopassword'),
            secure=False
        )
        self.bucket = os.getenv('MINIO_BUCKET', 'scraped')

        # ensure bucket exists
        try:
            if not self.minio_client.bucket_exists(self.bucket):
                self.minio_client.make_bucket(self.bucket)
                self.logger.warning(f"✅ Created MinIO bucket: {self.bucket}")
        except Exception as e:
            self.logger.error(f"MinIO bucket error: {e}")

    def start_requests(self):
        """Generate initial requests - backward compatible version."""
        for url in self.start_urls:
            try:
                yield scrapy.Request(
                    url=url,
                    meta={
                        "playwright": True,
                        "playwright_page_methods": [
                            PageMethod("wait_for_load_state", "load"),
                            PageMethod("wait_for_timeout", 2000),  # Wait for dynamic content
                        ],
                    },
                    headers={
                        # More realistic browser headers for anti-blocking
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                        'DNT': '1',
                    },
                    errback=self.handle_error,
                    dont_filter=False
                )
            except Exception as e:
                # Log errors at error level only
                self.logger.error(f'Failed to create request for {url}: {e}')

    def handle_error(self, failure):
        """Handle request errors and continue crawling."""
        # Don't log every failed request - too noisy
        self.logger.error(f"Request failed: {failure.request.url} - {failure.value}")

    # ----------------------------------------
    # HTML page handling
    # ----------------------------------------
    def parse_page(self, response):
        """Store full HTML of crawled page to MinIO for later NER processing."""
        if not isinstance(response, HtmlResponse):
            self.logger.debug(f"Skipping non-HTML response: {response.url}")
            return

        url = response.url
        try:
            html_bytes = response.body  # already bytes
            # Use hash of URL as filename to avoid collisions
            obj_name = f"pages/{abs(hash(url))}.html"

            metadata = {
                "x-amz-meta-source-url": url,
                "x-amz-meta-downloaded-at": str(time.time()),
                "x-amz-meta-content-type": "text/html"
            }

            self.minio_client.put_object(
                self.bucket,
                obj_name,
                io.BytesIO(html_bytes),
                length=len(html_bytes),
                part_size=10 * 1024 * 1024,
                content_type="text/html",
                metadata=metadata,
            )
        except Exception as e:
            self.logger.error(f"❌ Failed uploading page {url}: {e}")
        
        # We explicitly return nothing here, because the `follow=True` in the Rule
        # will handle the link extraction and following.
        return

    # ----------------------------------------
    # Media item handling (images/videos)
    # ----------------------------------------
    def parse_item(self, response):
        """This callback is for pages that are media files."""

        video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.wmv', '.flv', '.mkv']
        is_video = any(response.url.lower().endswith(ext) for ext in video_extensions)
        
        source_url = response.request.headers.get('Referer', b'').decode('utf-8', 'ignore')

        item = MediaItem()
        item['media_urls'] = [response.url]
        item['is_video'] = is_video
        item['source_url'] = source_url
        yield item