import os
import re
import json
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from dotenv import load_dotenv
import sys

# Ensure we can import from project modules by adding current directory to path
sys.path.append(os.getcwd())

from ingestion.preprocessor import Preprocessor
from database.supabase_client import SupabaseManager

load_dotenv()

class StandaloneBlogCrawler:
    def __init__(self, blog_url: str):
        # Handle input url clean up
        self.blog_url = blog_url.rstrip('/')
        self.blog_id = self.blog_url.split('/')[-1]
        self.base_url = "https://m.blog.naver.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Mobile Safari/537.36',
            'Referer': f'https://m.blog.naver.com/{self.blog_id}'
        }

    def get_post_urls(self) -> List[str]:
        """Fetches post URLs parsing __INITIAL_STATE__ from mobile site."""
        url = f"{self.base_url}/{self.blog_id}"
        print(f"Fetching list from: {url}")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            html = response.text
            
            # Regex to find the JSON state
            # Matches: window.__INITIAL_STATE__ = { ... }; (or similar)
            state_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?})\s*;?\s*</script>', html, re.DOTALL)
            if not state_match:
                 state_match = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?});', html, re.DOTALL)

            if state_match:
                state_json = state_match.group(1)
                try:
                    state = json.loads(state_json)
                except json.JSONDecodeError:
                    print("Failed to decode JSON state.")
                    return []
                
                items = []
                # Navigate JSON structure
                if 'postList' in state and 'data' in state['postList']:
                    items = state['postList']['data'].get('items', [])
                elif 'post' in state and 'data' in state['post']:
                     items = [state['post']['data']]
                
                links = []
                for item in items:
                    log_no = item.get('logNo')
                    title = item.get('title', 'No Title')
                    # Clean title (sometimes contains escaped chars)
                    title = BeautifulSoup(title, 'html.parser').get_text()
                    
                    if log_no:
                        link = f"https://m.blog.naver.com/{self.blog_id}/{log_no}"
                        print(f"Found post: {title}")
                        links.append(link)
                return links
            else:
                print("Could not find __INITIAL_STATE__ in page source. The blog might use a different theme or structure.")
                return []

        except Exception as e:
            print(f"Error fetching post list: {e}")
            return []

    def get_post_content(self, url: str) -> Optional[Dict]:
        """Fetches content of a single blog post."""
        print(f"Processing: {url}")
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to load {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Naver Smart Editor classes (Mobile)
            title_elem = soup.select_one('div.se-title-text') or soup.select_one('h3.tit_h3')
            content_elem = soup.select_one('div.se-main-container') or soup.select_one('div.post_ct') or soup.select_one('#viewTypeSelector')
            
            title = "No Title"
            content = ""

            if title_elem:
                title = title_elem.get_text(strip=True)
            else:
                 # Fallback title extraction
                 if soup.title:
                     title = soup.title.get_text(strip=True)

            if content_elem:
                # Remove scripts/styles
                for script in content_elem(["script", "style"]):
                    script.extract()
                content = content_elem.get_text(separator=' ', strip=True)
            else:
                print("  Warning: specific content selector failed. Trying broader extraction.")
                # Fallback: remove common header/footer elements and get body text
                for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
                    tag.decompose()
                content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""

            if not content:
                print(f"  Skipping {url} (No content found)")
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

def main():
    target_blog = "https://blog.naver.com/baravo"
    print(f"Starting standalone ingestion for: {target_blog}")
    
    crawler = StandaloneBlogCrawler(target_blog)
    
    # 1. Get Links
    urls = crawler.get_post_urls()
    print(f"Found {len(urls)} posts to process.")
    
    if not urls:
        print("No posts found. Please check if the blog ID is correct and accessible.")
        return

    # 2. Extract Content
    documents = []
    # Process all found urls
    for url in urls:
        doc = crawler.get_post_content(url)
        if doc:
            documents.append(doc)
    
    print(f"Successfully scraped {len(documents)} posts.")
    
    if not documents:
        return

    # 3. Preprocess (Chunking)
    print("Chunking content...")
    try:
        preprocessor = Preprocessor()
        all_chunks = []
        for doc in documents:
            chunks = preprocessor.process_content(doc)
            all_chunks.extend(chunks)
        
        print(f"Generated {len(all_chunks)} chunks.")
        
        # 4. Store in Supabase
        if all_chunks:
            print("Storing in Supabase...")
            db_manager = SupabaseManager()
            db_manager.insert_documents(all_chunks)
            print("Ingestion Complete! Data successfully stored in Supabase.")
            
    except Exception as e:
        print(f"Processing/Database Error: {e}")
        print("Ensure your .env file is correctly configured with SUPABASE_URL and SUPABASE_KEY.")

if __name__ == "__main__":
    main()
