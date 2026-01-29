import asyncio
from ingestion.blog_crawler import BlogCrawler

async def test():
    crawler = BlogCrawler()
    urls = await crawler.get_post_urls()
    print(f"Found {len(urls)} URLs")
    for url in urls[:3]:
        print(f"URL: {url}")
        content = await crawler.get_post_content(url)
        if content:
            print(f"Title: {content['title']}")
        else:
            print(f"Failed to get content for {url}")

asyncio.run(test())
