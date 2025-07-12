import os
import io
import time
from urllib.parse import urlparse
from minio import Minio
from itemadapter import ItemAdapter
import scrapy

class MinioMediaPipeline:
    """
    This pipeline intercepts media items (images and videos), downloads
    them, and uploads them directly to a MinIO bucket along with
    their metadata.
    """
    def __init__(self, settings):
        self.minio_client = Minio(
            settings.get('MINIO_ENDPOINT', 'minio:9000'),
            access_key=settings.get('MINIO_USER', 'minioadmin'),
            secret_key=settings.get('MINIO_PASSWORD', 'miniopassword'),
            secure=False
        )
        self.bucket = settings.get('MINIO_BUCKET', 'scraped')
        self.items_processed = 0

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def open_spider(self, spider):
        """Create bucket if it doesn't exist when the spider is opened."""
        try:
            if not self.minio_client.bucket_exists(self.bucket):
                self.minio_client.make_bucket(self.bucket)
                spider.logger.warning(f"✅ Created MinIO bucket: {self.bucket}")
            # Don't log if bucket exists - too noisy
        except Exception as e:
            spider.logger.error(f"❌ MinIO bucket error: {e}")

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        media_urls = adapter.get('media_urls', [])
        
        if not media_urls:
            return item
            
        # We only process the first media URL, to keep consistency with previous implementation.
        media_url = media_urls[0]
        
        request = scrapy.Request(media_url)
        dfd = spider.crawler.engine.download(request)
        dfd.addCallback(self.media_downloaded, item=item, spider=spider)
        dfd.addErrback(self.media_failed, item=item, media_url=media_url, spider=spider)
        return dfd

    def media_downloaded(self, response, item, spider):
        """
        This method is called when the media is successfully downloaded.
        It uploads the media to MinIO.
        """
        self.items_processed += 1
        media_url = response.url
        adapter = ItemAdapter(item)
        is_video = adapter.get('is_video', False)
        source_url = adapter.get('source_url', '')
        
        try:
            # Create a unique object name with prefix
            file_ext = os.path.splitext(urlparse(media_url).path)[1] or ('.mp4' if is_video else '.jpg')
            object_base_name = str(abs(hash(media_url)))  # Use abs to avoid negative numbers
            prefix = "videos/" if is_video else "images/"
            object_name = f"{prefix}{object_base_name}{file_ext}"

            # Prepare metadata for the object
            metadata = {
                "x-amz-meta-media-url": media_url,
                "x-amz-meta-source-url": source_url,
                "x-amz-meta-downloaded-at": str(time.time()),
                "x-amz-meta-is-video": str(is_video).lower()
            }
            
            # Upload the media content with metadata - larger part size for better performance
            content_type_bytes = response.headers.get('Content-Type', b'application/octet-stream')
            content_type = content_type_bytes.decode('utf-8')
            
            self.minio_client.put_object(
                self.bucket,
                object_name,
                io.BytesIO(response.body),
                length=len(response.body),
                part_size=50*1024*1024,  # 50MB parts for better performance with large files
                content_type=content_type,
                metadata=metadata
            )
            
            # Log progress every 5 items instead of 10 for better feedback
            if self.items_processed % 5 == 0:
                spider.logger.warning(f"✅ Uploaded {self.items_processed} items (latest: {object_name})")

        except Exception as e:
            spider.logger.error(f"❌ Upload failed {media_url}: {e}")
        
        return item

    def media_failed(self, failure, item, media_url, spider):
        """
        This method is called when the media download fails.
        """
        spider.logger.error(f"❌ Download failed {media_url}: {failure.getErrorMessage()}")
        return item
    
    def close_spider(self, spider):
        """Log statistics when spider closes."""
        if self.items_processed > 0:
            spider.logger.warning(f"🏁 Spider finished. Total items processed: {self.items_processed}") 