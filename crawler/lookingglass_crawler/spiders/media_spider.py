import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy_patchright.page import PageMethod
from ..items import MediaItem

class MediaSpider(CrawlSpider):
    name = 'media_spider'
    
    def __init__(self, *args, **kwargs):
        super(MediaSpider, self).__init__(*args, **kwargs)
        self.start_urls = ['https://www.flickr.com/search/?text=portrait']
        
        media_extensions = {
            'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'mp4', 
            'webm', 'avi', 'mov', 'wmv', 'flv', 'mkv'
        }

        # Create a regex to match URLs ending with media extensions.
        # This is used in the 'allow' and 'deny' parameters of LinkExtractor.
        media_regex = r'\.(' + '|'.join(media_extensions) + r')$'

        self.rules = (
            # Rule 1: Extract media links and send them to the pipeline.
            # It finds links in `<a>`, `<img>`, `<video>` tags and processes them 
            # with `parse_item`. It does NOT follow these links.
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
            # Rule 2: Extract all other page links and follow them for crawling.
            # It ignores media links (to avoid duplication) and follows all
            # other links to discover new pages.
            Rule(
                LinkExtractor(
                    deny=media_regex,
                    canonicalize=True,
                    unique=True
                ),
                follow=True
            ),
        )

        # Re-compile the rules after defining them in __init__
        super(MediaSpider, self)._compile_rules()

    def start_requests(self):
        """Generate initial requests - backward compatible version."""
        for url in self.start_urls:
            try:
                self.logger.debug(f"Starting request for {url}")
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

    def parse_item(self, response):
        """This callback is for pages that are media files."""
        self.logger.debug(f"Found media: {response.url}")

        video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.wmv', '.flv', '.mkv']
        is_video = any(response.url.lower().endswith(ext) for ext in video_extensions)
        
        source_url = response.request.headers.get('Referer', b'').decode('utf-8', 'ignore')

        item = MediaItem()
        item['media_urls'] = [response.url]
        item['is_video'] = is_video
        item['source_url'] = source_url
        yield item