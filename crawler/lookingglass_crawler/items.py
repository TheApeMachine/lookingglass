import scrapy

class MediaItem(scrapy.Item):
    """
    Item for storing scraped media.
    Uses 'image_urls' for ImagesPipeline and 'file_urls' for FilesPipeline.
    """
    source_url = scrapy.Field()
    media_urls = scrapy.Field() # For MinioMediaPipeline
    is_video = scrapy.Field()   # For MinioMediaPipeline
    
    # For Scrapy's ImagesPipeline
    image_urls = scrapy.Field()
    images = scrapy.Field()

    # For Scrapy's FilesPipeline (used for videos)
    file_urls = scrapy.Field()
    files = scrapy.Field() 