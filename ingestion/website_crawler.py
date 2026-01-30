"""
Seoul OnCare Hospital Website Crawler
Crawls all pages from https://seouloncare.co.kr/
"""
import re
import ssl
import time
import random
import urllib.request
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser

ssl._create_default_https_context = ssl._create_unverified_context


class MLStripper(HTMLParser):
    """HTML 태그 제거용"""
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.skip = False
        self.skip_tags = ['script', 'style', 'nav', 'footer']
    
    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip = True
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip = False
    
    def handle_data(self, d):
        if not self.skip:
            self.fed.append(d)
    
    def get_data(self):
        return ' '.join(self.fed)


def strip_html(html):
    """HTML 태그 제거 및 정제"""
    s = MLStripper()
    try:
        s.feed(html)
        text = s.get_data()
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except:
        # Fallback
        clean = re.compile('<[^>]+>')
        text = re.sub(clean, ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class SeoulOnCareCrawler:
    """서울온케어의원 웹사이트 크롤러"""
    
    def __init__(self, base_url: str = "https://seouloncare.co.kr"):
        self.base_url = base_url.rstrip('/')
        self.visited_urls: Set[str] = set()
        self.crawled_pages: List[Dict] = []
        
        # Setup request handler
        self.opener = urllib.request.build_opener()
        self.opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
            ('Accept-Language', 'ko-KR,ko;q=0.9'),
        ]
    
    def fetch_page(self, url: str) -> Optional[str]:
        """단일 페이지 fetch"""
        try:
            if not url.startswith('http'):
                url = urljoin(self.base_url, url)
            
            response = self.opener.open(url, timeout=15)
            content_type = response.headers.get('Content-Type', '')
            
            if 'text/html' not in content_type:
                return None
            
            html = response.read()
            
            # Detect encoding
            encoding = 'utf-8'
            if 'charset=' in content_type:
                match = re.search(r'charset=([^;]+)', content_type)
                if match:
                    encoding = match.group(1).strip()
            
            try:
                return html.decode(encoding)
            except:
                return html.decode('utf-8', errors='ignore')
                
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def extract_links(self, html: str, current_url: str) -> List[str]:
        """HTML에서 남부 링크 추출"""
        links = []
        
        # Skip non-HTML content
        skip_extensions = ['.css', '.js', '.jpg', '.jpeg', '.png', '.gif', '.svg', 
                          '.ico', '.woff', '.woff2', '.ttf', '.eot', '.pdf', '.zip']
        
        # Find all href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(href_pattern, html)
        
        for href in matches:
            # Skip external links and anchors
            if href.startswith('http') and not href.startswith(self.base_url):
                continue
            if href.startswith('#') or href.startswith('javascript:'):
                continue
            if href.startswith('mailto:') or href.startswith('tel:'):
                continue
            
            # Skip asset files
            href_lower = href.lower()
            if any(href_lower.endswith(ext) for ext in skip_extensions):
                continue
            
            # Convert to absolute URL
            full_url = urljoin(current_url, href)
            
            # Keep only same domain
            if full_url.startswith(self.base_url):
                # Remove fragment
                full_url = full_url.split('#')[0]
                links.append(full_url)
        
        return list(set(links))
    
    def extract_content(self, html: str, url: str) -> Dict:
        """HTML에서 콘텐츠 추출"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for elem in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            elem.extract()
        
        # Extract title
        title = "제목 없음"
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)
        # Try h1
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        
        # Extract main content
        content = ""
        
        # Try common content containers
        content_selectors = [
            'main', '.content', '#content', '.container',
            'article', '.article', '.post',
            '#bo_vc',  # 게시판 댓글
            '.tbl_head01',  # 테이블
        ]
        
        for selector in content_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(separator='\n', strip=True)
                if len(text) > len(content):
                    content = text
        
        # Fallback to body
        if not content or len(content) < 100:
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)
        
        # Clean up content
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Extract tables (for price info, hours, etc.)
        tables = []
        for table in soup.find_all('table'):
            table_text = table.get_text(separator=' | ', strip=True)
            if table_text:
                tables.append(table_text)
        
        if tables:
            content += "\n\n[표 데이터]\n" + "\n".join(tables)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'content_length': len(content),
        }
    
    def crawl(self, max_pages: int = 50, delay: tuple = (1, 2)) -> List[Dict]:
        """
        전체 사이트 크롤링
        """
        print(f"[Crawler] Starting crawl of {self.base_url}")
        print("=" * 60)
        
        # Start with homepage
        to_visit = [self.base_url]
        self.visited_urls.add(self.base_url)
        
        while to_visit and len(self.crawled_pages) < max_pages:
            url = to_visit.pop(0)
            
            print(f"\n[{len(self.crawled_pages) + 1}] Crawling: {url}")
            
            # Fetch page
            html = self.fetch_page(url)
            if not html:
                continue
            
            # Extract content
            page_data = self.extract_content(html, url)
            
            if page_data['content_length'] > 50:  # Skip empty pages
                self.crawled_pages.append(page_data)
                print(f"    [OK] Title: {page_data['title'][:50]}...")
                print(f"    [OK] Content: {page_data['content_length']} chars")
            else:
                print(f"    [SKIP] Content too short ({page_data['content_length']} chars)")
            
            # Extract and queue new links
            links = self.extract_links(html, url)
            new_links = 0
            for link in links:
                if link not in self.visited_urls:
                    self.visited_urls.add(link)
                    to_visit.append(link)
                    new_links += 1
            
            if new_links > 0:
                print(f"    [+] Found {new_links} new links (queue: {len(to_visit)})")
            
            # Rate limiting
            if to_visit and len(self.crawled_pages) < max_pages:
                time.sleep(random.uniform(delay[0], delay[1]))
        
        print("\n" + "=" * 60)
        print(f"[Crawler] Completed: {len(self.crawled_pages)} pages crawled")
        return self.crawled_pages


if __name__ == "__main__":
    crawler = SeoulOnCareCrawler()
    pages = crawler.crawl(max_pages=30)
    
    print(f"\n{'='*60}")
    print("CRAWLED PAGES SUMMARY")
    print(f"{'='*60}")
    for i, page in enumerate(pages, 1):
        print(f"\n{i}. {page['title'][:50]}...")
        print(f"   URL: {page['url']}")
        print(f"   Content: {page['content_length']} chars")
        print(f"   Preview: {page['content'][:150]}...")
