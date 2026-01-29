import asyncio
import re
import json
import random
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from config.settings import BLOG_URL

class BlogCrawler:
    def __init__(self, blog_url: str = BLOG_URL):
        self.blog_url = blog_url.rstrip('/')
        self.blog_id = self.blog_url.split('/')[-1]
        self.base_url = "https://m.blog.naver.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://m.blog.naver.com/',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        }

    def _get_post_urls_sync(self, page: int = 1) -> List[str]:
        """Fetches post URLs using multiple fallback methods."""
        url = f"{self.base_url}/PostList.naver?blogId={self.blog_id}&currentPage={page}"
        print(f"Fetching blog post list from: {url}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            html = response.text
            
            links = []

            # Method 1: Try parsing __INITIAL_STATE__ JSON
            state_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?})\s*;?\s*</script>', html, re.DOTALL)
            if not state_match:
                 state_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?});', html, re.DOTALL)

            if state_match:
                try:
                    state = json.loads(state_match.group(1))
                    items = []
                    if 'postList' in state and 'data' in state['postList']:
                        items = state['postList']['data'].get('items', [])
                    
                    for item in items:
                        log_no = item.get('logNo')
                        if log_no:
                            links.append(f"https://m.blog.naver.com/{self.blog_id}/{log_no}")
                except Exception as e:
                    print(f"  - JSON parse failed, trying HTML fallback... ({e})")

            # Method 2: HTML Tag Fallback (If Method 1 failed or found nothing)
            if not links:
                soup = BeautifulSoup(html, 'html.parser')
                # Mobile Naver Blog post link patterns
                for a in soup.select('ul.list_post a.link, a.item_link'):
                    href = a.get('href')
                    if href and 'logNo=' in href:
                        log_no_match = re.search(r'logNo=(\d+)', href)
                        if log_no_match:
                            links.append(f"https://m.blog.naver.com/{self.blog_id}/{log_no_match.group(1)}")
                
                # Broad pattern match for blog post URLs
                if not links:
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if f"/{self.blog_id}/" in href and any(char.isdigit() for char in href):
                             links.append(href if href.startswith('http') else f"https://m.blog.naver.com{href}")

            # Remove duplicates while preserving order
            seen = set()
            unique_links = []
            for link in links:
                if link not in seen:
                    unique_links.append(link)
                    seen.add(link)
                    
            return unique_links
        except Exception as e:
            print(f"Error fetching blog list: {e}")
            return []

    async def get_post_urls(self, page: int = 1) -> List[str]:
        return await asyncio.to_thread(self._get_post_urls_sync, page)

    def _get_post_content_sync(self, url: str) -> Optional[Dict]:
        """Fetches content of a single blog post."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Title extraction
            title_elem = soup.select_one('div.se-title-text, h3.tit_h3, .se-viewer .se-title-text')
            title = title_elem.get_text(strip=True) if title_elem else "No Title"
            
            # Content extraction (Naver Smart Editor 3.0 & ONE)
            content_elem = soup.select_one('div.se-main-container, div.post_ct, #viewTypeSelector, .se-viewer')
            
            if content_elem:
                # Remove unwanted elements
                for tag in content_elem(["script", "style", "iframe", "button"]):
                    tag.extract()
                content = content_elem.get_text(separator='\n', strip=True)
            else:
                # Last resort fallback
                content = ""
                for p in soup.find_all(['p', 'div'], class_=re.compile('se-text|post_ct')):
                    content += p.get_text(strip=True) + "\n"

            if not content or len(content) < 50: # Skip very short posts/failed parses
                return None

            return {
                'url': url,
                'title': title,
                'content': content,
                'source': 'blog'
            }
        except Exception as e:
            print(f"Error fetching content for {url}: {e}")
            return None

    async def get_post_content(self, url: str) -> Optional[Dict]:
        return await asyncio.to_thread(self._get_post_content_sync, url)

    async def collect_recent_posts(self, limit: int = 10) -> List[Dict]:
        """Collects recent blog posts sequentially with delay."""
        urls = await self.get_post_urls()
        target_urls = urls[:limit]
        
        results = []
        for i, url in enumerate(target_urls):
            print(f"[{i+1}/{len(target_urls)}] Collecting blog post: {url}")
            result = await self.get_post_content(url)
            if result:
                results.append(result)
            
            if i < len(target_urls) - 1:
                sleep_time = random.uniform(3, 7)
                await asyncio.sleep(sleep_time)
                
        return results

if __name__ == "__main__":
    crawler = BlogCrawler()
    results = asyncio.run(crawler.collect_recent_posts(limit=2))
    print(f"Collected {len(results)} posts.")