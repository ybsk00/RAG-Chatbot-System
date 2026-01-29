import asyncio
from ingestion.youtube_collector import YouTubeCollector
from ingestion.blog_crawler import BlogCrawler
from ingestion.preprocessor import Preprocessor
from database.supabase_client import SupabaseManager

async def main():
    print("Starting Ingestion Pipeline...")
    
    # 1. Collect Data
    yt_collector = YouTubeCollector()
    blog_crawler = BlogCrawler()
    
    # Run collection in parallel
    print("Collecting data from YouTube and Blog...")
    yt_data, blog_data = await asyncio.gather(
        yt_collector.collect_all(limit=5), # Limit for testing
        blog_crawler.collect_recent_posts(limit=5)
    )
    
    all_data = yt_data + blog_data
    print(f"Collected {len(all_data)} items.")
    
    # 2. Preprocess
    print("Preprocessing data...")
    preprocessor = Preprocessor()
    all_chunks = []
    for item in all_data:
        chunks = preprocessor.process_content(item)
        all_chunks.extend(chunks)
        
    print(f"Generated {len(all_chunks)} chunks.")
    
    # 3. Store in DB
    if all_chunks:
        print("Storing in Supabase...")
        try:
            db_manager = SupabaseManager()
            db_manager.insert_documents(all_chunks)
            print("Ingestion Complete!")
        except Exception as e:
            print(f"Database error: {e}")
            print("Ensure Supabase credentials are set and table is created.")
            # Print SQL for user
            print("Run this SQL in Supabase SQL Editor if table missing:")
            # We can't easily access the method without instance, but it's in the file.

if __name__ == "__main__":
    asyncio.run(main())
