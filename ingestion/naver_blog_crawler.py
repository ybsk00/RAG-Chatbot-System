"""
Naver Blog Crawler using RSS feed
"""
import os
import re
import ssl
import time
import random
import feedparser
from typing import List, Dict, Optional
from html.parser import HTMLParser

ssl._create_default_https_context = ssl._create_unverified_context


class MLStripper(HTMLParser):
    """HTML 태그 제거용"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    
    def handle_data(self, d):
        self.fed.append(d)
    
    def get_data(self):
        return ''.join(self.fed)


def strip_html(html):
    """HTML 태그 제거"""
    s = MLStripper()
    try:
        s.feed(html)
        return s.get_data()
    except:
        clean = re.compile('<.*?>')
        return re.sub(clean, ' ', html)


class NaverBlogCrawler:
    """네이버 블로그 RSS 크롤러"""
    
    def __init__(self, blog_id: str = "baravo"):
        self.blog_id = blog_id
        self.rss_url = f"https://rss.blog.naver.com/{blog_id}.xml"
    
    def get_post_urls(self, max_posts: int = 50) -> List[Dict]:
        """RSS에서 글 목록 수집"""
        print(f"[Crawler] Fetching RSS: {self.rss_url}")
        
        feed = feedparser.parse(self.rss_url)
        
        if feed.bozo:
            print(f"[Crawler] RSS Error: {feed.bozo_exception}")
            return []
        
        posts = []
        entries = feed.entries[:max_posts]
        
        print(f"[Crawler] Found {len(entries)} entries in RSS")
        
        for entry in entries:
            try:
                link = entry.link
                match = re.search(r'/baravo/(\d+)', link)
                if not match:
                    continue
                post_id = match.group(1)
                
                # Get full description/content from RSS
                content = entry.get('description', '')
                # Some RSS feeds have 'content' field with full text
                if hasattr(entry, 'content'):
                    content = entry.content[0].value if isinstance(entry.content, list) else str(entry.content)
                
                # Clean HTML
                content = strip_html(content)
                
                posts.append({
                    'url': link,
                    'title': strip_html(entry.get('title', '제목 없음')),
                    'post_id': post_id,
                    'date': entry.get('published', ''),
                    'content': content,  # Full content from RSS
                })
                
            except Exception as e:
                print(f"[Crawler] Error parsing entry: {e}")
                continue
        
        print(f"[Crawler] Total posts: {len(posts)}")
        return posts
    
    def get_post_detail(self, post: Dict) -> Optional[Dict]:
        """상세 정보 구성"""
        content = post.get('content', '')
        
        # If content is too short, create placeholder
        if len(content) < 50:
            content = f"{post['title']}\n\n[원문 보기: {post['url']}]"
        
        return {
            'url': post['url'],
            'title': post['title'],
            'content': content,
            'date': post.get('date', ''),
            'post_id': post.get('post_id', ''),
            'thumbnail': '',
            'author': self.blog_id,
            'source_type': 'blog'
        }
    
    def crawl_all_posts(self, max_posts: int = 50, **kwargs) -> List[Dict]:
        """전체 파이프라인"""
        post_list = self.get_post_urls(max_posts=max_posts)
        
        if not post_list:
            print("[Crawler] No posts found")
            return []
        
        detailed_posts = []
        for i, post in enumerate(post_list):
            print(f"[{i+1}/{len(post_list)}] {post['title'][:40]}...")
            
            detail = self.get_post_detail(post)
            detailed_posts.append(detail)
            print(f"    OK (content: {len(detail['content'])} chars)")
        
        print(f"\n{'='*60}")
        print(f"[Crawler] Completed: {len(detailed_posts)} posts")
        return detailed_posts


if __name__ == "__main__":
    crawler = NaverBlogCrawler(blog_id="baravo")
    posts = crawler.crawl_all_posts(max_posts=5)
    
    print(f"\n{'='*60}")
    print(f"Crawled {len(posts)} posts")
    for post in posts[:2]:
        print(f"\n- {post['title'][:50]}...")
        print(f"  Content: {len(post['content'])} chars")
        print(f"  Preview: {post['content'][:200]}...")
