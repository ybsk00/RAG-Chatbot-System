import asyncio
import os
import sys
import random

# Add current directory to path
sys.path.append(os.getcwd())

from ingestion.youtube_collector import YouTubeCollector
from ingestion.blog_crawler import BlogCrawler
from ingestion.preprocessor import Preprocessor
from database.supabase_client import SupabaseManager

async def process_and_save_item(item, preprocessor, db_manager, source_type):
    """Helper function to process a single item and save to DB immediately."""
    try:
        if not item:
            return False
            
        # 1. Chunking
        chunks = preprocessor.process_content(item)
        if not chunks:
            # print(f"  - No chunks generated for {source_type} item.")
            return False
            
        # 2. Store in DB
        db_manager.insert_documents(chunks)
        print(f"  >>> Successfully saved {len(chunks)} chunks to DB.")
        return True
    except Exception as e:
        print(f"  !!! Error saving {source_type} item: {e}")
        return False

async def main():
    print("=== Starting Real-time Ingestion Pipeline (Immediate Save) ===")
    
    # Initialize Components
    yt_collector = YouTubeCollector()
    blog_crawler = BlogCrawler(blog_url="https://blog.naver.com/baravo")
    preprocessor = Preprocessor()
    
    try:
        db_manager = SupabaseManager()
    except Exception as e:
        print(f"Critical Error: Could not connect to Database. {e}")
        return

    # --- Phase 1: YouTube Real-time Collection ---
    print("\n--- Phase 1: YouTube Collection & Save ---")
    
    # 1. Get List of IDs first
    video_ids = await yt_collector.get_video_ids()
    print(f"Found total {len(video_ids)} YouTube videos to process.")
    
    for i, vid in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] Processing YouTube: {vid}")
        
        # 1. Collect
        item = await yt_collector.process_video(vid)
        
        # 2. Save Immediately
        if item:
            await process_and_save_item(item, preprocessor, db_manager, "YouTube")
        else:
            print("  - Skipped (No transcript or metadata)")
            
        # 3. Sleep
        if i < len(video_ids) - 1:
            sleep_time = random.uniform(5, 12)
            print(f"  ... Sleeping {sleep_time:.1f}s ...")
            await asyncio.sleep(sleep_time)

    # --- Phase 2: Blog Real-time Collection ---
    print("\n--- Phase 2: Blog Collection & Save ---")
    
    # 1. Get List of URLs
    blog_urls = await blog_crawler.get_post_urls()
    print(f"Found total {len(blog_urls)} Blog posts to process.")
    
    for i, url in enumerate(blog_urls):
        print(f"\n[{i+1}/{len(blog_urls)}] Processing Blog: {url}")
        
        # 1. Collect
        item = await blog_crawler.get_post_content(url)
        
        # 2. Save Immediately
        if item:
            await process_and_save_item(item, preprocessor, db_manager, "Blog")
        else:
            print("  - Skipped (Failed to fetch content)")
            
        # 3. Sleep
        if i < len(blog_urls) - 1:
            sleep_time = random.uniform(5, 10)
            print(f"  ... Sleeping {sleep_time:.1f}s ...")
            await asyncio.sleep(sleep_time)

    print("\n=== Ingestion Complete! ===")

if __name__ == "__main__":
    asyncio.run(main())