import asyncio
import os
import sys
import random
from typing import List, Dict

# Add current directory to path
sys.path.append(os.getcwd())

# Import our modules
from ingestion.youtube_collector import YouTubeCollector
from ingestion.blog_crawler import BlogCrawler
from database.supabase_client import SupabaseManager

# Import LangChain & Gemini for refinement
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Gemini Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1 # Low temperature for factual consistency
)

# Refine Prompt (Optimized for FAQ Generation)
refine_prompt = PromptTemplate(
    template="""
    당신은 20년 경력의 전문 의학 에디터입니다. 
    아래 텍스트는 유튜브 영상/블로그에서 추출한 내용으로, 오타와 부정확한 문장이 많을 수 있습니다.

    [지침]
    1. 텍스트를 분석하여 **환자들이 가장 궁금해할 만한 핵심 질문과 답변(Q&A)** 1~3개 세트로 요약 및 재구성하세요.
    2. 문맥상 명백한 의학 용어 오타(예: 간함 -> 간암, 상곡보송 -> 상복부 통증)를 정확하게 교정하세요.
    3. 답변은 병원의 전문성이 느껴지도록 친절하고 신뢰감 있게 작성하세요. (해요체 사용)
    4. 의학적 확진이 아닌 '정보 제공'의 톤(예: "~할 수 있습니다")을 유지하세요.
    5. **반드시 한국어로 작성하세요.**

    [원본 텍스트]:
    {text}

    [정제된 결과 (Q&A 형식)]:
    """,
    input_variables=["text"]
)

chain = refine_prompt | llm | StrOutputParser()

async def refine_content(text: str) -> str:
    """Uses Gemini to clean and structure the text into Q&A."""
    if not text or len(text) < 50:
        return text # Too short to refine
    
    try:
        # print("    ... Gemini is refining the content ...")
        refined_text = await chain.ainvoke({"text": text})
        return refined_text
    except Exception as e:
        print(f"    !! Gemini Refinement Error: {e}")
        return text # Return original if AI fails

async def process_and_save_item(item: Dict, db_manager, source_type: str):
    """Refines content AND saves to DB immediately."""
    try:
        if not item or not item.get('content'):
            print(f"  - No content to process for {source_type} item.")
            return False
            
        original_content = item['content']
        title = item.get('title', 'Unknown Title')
        
        # 1. AI Refinement
        print(f"  -> Sending to Gemini 2.0 Flash for refinement...")
        refined_content = await refine_content(original_content)
        
        # Log a snippet of the result to the user
        preview = refined_content[:150].replace('\n', ' ')
        print(f"  [Refined Preview]: {preview}...")
        
        # 2. Update item with refined content
        item['metadata'] = item.get('metadata', {})
        item['metadata']['original_content'] = original_content[:500]
        item['metadata']['is_refined'] = True
        
        doc = {
            'content': refined_content,
            'metadata': item['metadata']
        }
        
        # 3. Store in DB
        print(f"  -> Uploading to Supabase Vector DB...")
        db_manager.insert_documents([doc])
        print(f"  >>> SUCCESS: Refined content for '{title}' saved to DB.")
        return True
        
    except Exception as e:
        print(f"  !!! CRITICAL ERROR processing/saving {source_type}: {e}")
        return False

async def main():
    print("=== Starting All-in-One Pipeline (Collect -> Refine -> Save) ===")
    print(f"Using Gemini Model: gemini-2.0-flash-exp")
    
    # Initialize Components
    yt_collector = YouTubeCollector()
    blog_crawler = BlogCrawler(blog_url="https://blog.naver.com/baravo")
    
    try:
        db_manager = SupabaseManager()
    except Exception as e:
        print(f"Critical Error: Database connection failed. {e}")
        return

    # --- Phase 1: YouTube ---
    print("\n--- Phase 1: YouTube Processing ---")
    video_ids = await yt_collector.get_video_ids()
    print(f"Found total {len(video_ids)} YouTube videos.")
    
    for i, vid in enumerate(video_ids):
        print(f"\n[{i+1}/{len(video_ids)}] Processing YouTube: {vid}")
        
        # 1. Collect (Download/STT)
        item = await yt_collector.process_video(vid)
        
        # Normalize Key: YouTube uses 'transcript', we need 'content'
        if item and 'transcript' in item:
            item['content'] = item.pop('transcript')
        
        # 2. Refine & Save
        if item:
            await process_and_save_item(item, db_manager, "YouTube")
        else:
            print("  - Skipped (No content/transcript found)")
            
        # Sleep to be polite to APIs
        if i < len(video_ids) - 1:
            await asyncio.sleep(random.uniform(3, 7))

    # --- Phase 2: Blog ---
    print("\n--- Phase 2: Blog Processing ---")
    blog_urls = await blog_crawler.get_post_urls()
    print(f"Found total {len(blog_urls)} Blog posts.")
    
    for i, url in enumerate(blog_urls):
        print(f"\n[{i+1}/{len(blog_urls)}] Processing Blog: {url}")
        
        # 1. Collect
        item = await blog_crawler.get_post_content(url)
        
        # 2. Refine & Save
        if item:
            await process_and_save_item(item, db_manager, "Blog")
        else:
            print("  - Skipped (No content)")
            
        if i < len(blog_urls) - 1:
            await asyncio.sleep(random.uniform(2, 5))

    print("\n=== All Tasks Completed! ===")

if __name__ == "__main__":
    asyncio.run(main())