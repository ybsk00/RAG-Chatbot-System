import os
import asyncio
import random
import yt_dlp
import whisper
import warnings
import time
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from config.settings import YOUTUBE_CHANNEL_URL

# Ignore redundant warnings
warnings.filterwarnings("ignore")

class YouTubeCollector:
    def __init__(self, channel_url: str = YOUTUBE_CHANNEL_URL):
        self.channel_url = channel_url
        
        # Load Whisper Model
        print("Loading Whisper Model...")
        self.model = whisper.load_model("base") 

        # Selenium Driver (Initialized only when needed)
        self.driver = None

    def get_cookies_via_selenium(self, url: str) -> str:
        """Opens a real browser to fetch fresh cookies."""
        print("  -> Fetching fresh cookies via Selenium (Real Browser)...")
        if not self.driver:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new") # Run in background
            options.add_argument("--mute-audio")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            # User agent to look like real PC
            options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            try:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            except Exception as e:
                print(f"  !! Failed to init Selenium: {e}")
                return None

        try:
            self.driver.get(url)
            time.sleep(3) # Wait for page load
            
            # Extract cookies and format for yt-dlp
            selenium_cookies = self.driver.get_cookies()
            cookie_file = os.path.join(os.getcwd(), "selenium_cookies.txt")
            
            with open(cookie_file, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in selenium_cookies:
                    # Netscape format: domain flag path secure expiration name value
                    domain = cookie.get('domain', '')
                    flag = 'TRUE' if domain.startswith('.') else 'FALSE'
                    path = cookie.get('path', '/')
                    secure = 'TRUE' if cookie.get('secure') else 'FALSE'
                    expiry = str(int(cookie.get('expiry', time.time() + 3600)))
                    name = cookie.get('name', '')
                    value = cookie.get('value', '')
                    f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
            
            return cookie_file
        except Exception as e:
            print(f"  !! Selenium Cookie Fetch Error: {e}")
            return None

    def close_driver(self):
        if self.driver:
            self.driver.quit()
            self.driver = None

    async def get_video_ids(self) -> List[str]:
        """Fetches all video IDs using yt-dlp."""
        base_url = self.channel_url.split('/videos')[0].split('/shorts')[0].split('/streams')[0]
        targets = [f"{base_url}/videos", f"{base_url}/shorts", f"{base_url}/streams"]
        
        all_ids = set()
        loop = asyncio.get_event_loop()
        
        # Simple extraction options for IDs (usually works without cookies)
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': False,
            'ignoreerrors': True,
        }

        def _extract(url):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    result = ydl.extract_info(url, download=False)
                    ids = []
                    if 'entries' in result:
                        for entry in result['entries']:
                            if entry and entry.get('id'):
                                ids.append(entry['id'])
                    return ids
                except Exception as e:
                    print(f"Error extracting from {url}: {e}")
                    return []

        for target in targets:
            print(f"Fetching IDs from {target}...")
            ids = await loop.run_in_executor(None, _extract, target)
            all_ids.update(ids)
            await asyncio.sleep(1)
            
        return list(all_ids)

    async def get_transcript_from_cc(self, video_id: str) -> Optional[str]:
        """Fetches official YouTube transcript (CC)."""
        try:
            api = YouTubeTranscriptApi()
            fetched = await asyncio.to_thread(api.fetch, video_id, languages=['ko', 'en'])
            return " ".join([s.text for s in fetched])
        except Exception:
            return None

    async def get_transcript_from_audio(self, video_id: str) -> Optional[str]:
        """Downloads audio using Selenium-fetched cookies and transcribes."""
        print(f"  -> No CC found. Starting STT (Audio Transcription) for {video_id}...")
        
        # Setup Paths
        ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
        if os.path.exists(ffmpeg_path):
             os.environ["PATH"] += os.pathsep + os.getcwd()

        temp_file = f"temp_{video_id}"
        abs_temp_path = os.path.join(os.getcwd(), temp_file)

        # 1. Get Fresh Cookies via Selenium
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        cookie_file = await asyncio.to_thread(self.get_cookies_via_selenium, video_url)
        
        # 2. Download with yt-dlp using those cookies
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': abs_temp_path + ".%(ext)s",
            'quiet': True,
            'no_warnings': True,
            'overwrites': True,
            'ffmpeg_location': ffmpeg_path if os.path.exists(ffmpeg_path) else None,
        }
        
        if cookie_file and os.path.exists(cookie_file):
            ydl_opts['cookiefile'] = cookie_file

        try:
            print(f"  -> Downloading audio...")
            loop = asyncio.get_event_loop()
            def _download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
            
            await loop.run_in_executor(None, _download)
            
            # Verify Download
            found_file = None
            for f in os.listdir('.'):
                if f.startswith(temp_file) and os.path.getsize(f) > 0:
                    found_file = f
                    break
            
            if not found_file:
                print(f"  !! Download failed even with Selenium cookies.")
                return None

            # 3. Transcribe
            print(f"  -> Transcribing audio with Whisper...")
            result = await asyncio.to_thread(self.model.transcribe, found_file, language='ko')
            transcript = result.get('text', '').strip()
            
            # Cleanup
            if found_file and os.path.exists(found_file):
                try: os.remove(found_file)
                except: pass
            
            # Cleanup cookie file
            # if cookie_file and os.path.exists(cookie_file):
            #     os.remove(cookie_file)

            return transcript if transcript else None

        except Exception as e:
            print(f"  !! STT Error: {e}")
            return None

    async def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """Fetches metadata."""
        ydl_opts = {'quiet': True, 'no_warnings': True, 'ignoreerrors': True}
        def _extract():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                except Exception:
                    return None
        
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _extract)
        if not info: return None
            
        return {
            'id': video_id,
            'title': info.get('title', ''),
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'upload_date': info.get('upload_date', ''),
            'description': info.get('description', '')
        }

    async def process_video(self, video_id: str) -> Optional[Dict]:
        """Process a single video."""
        metadata = await self.get_video_metadata(video_id)
        if not metadata: return None
            
        transcript = await self.get_transcript_from_cc(video_id)
        if not transcript:
            transcript = await self.get_transcript_from_audio(video_id)
            
        if transcript:
            metadata['transcript'] = transcript
            return metadata
        return None

    async def collect_all(self, limit: int = None) -> List[Dict]:
        """Collects all videos."""
        video_ids = await self.get_video_ids()
        print(f"Found total {len(video_ids)} unique video IDs.")
        
        target_ids = video_ids[:limit] if limit else video_ids
        results = []
        
        try:
            for i, vid in enumerate(target_ids):
                print(f"[{i+1}/{len(target_ids)}] Processing YouTube: {vid}")
                result = await self.process_video(vid)
                if result:
                    results.append(result)
                
                if i < len(target_ids) - 1:
                    await asyncio.sleep(random.uniform(3, 7))
        finally:
            # Ensure browser closes at end
            self.close_driver()
        
        return results

if __name__ == "__main__":
    collector = YouTubeCollector()
    results = asyncio.run(collector.collect_all(limit=1))
    print(f"Collected {len(results)} videos.")