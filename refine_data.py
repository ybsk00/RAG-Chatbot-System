import os
import asyncio
import json
from typing import List, Dict
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini Model Setup (Using Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Prompt Template
refine_prompt = PromptTemplate(
    template="""
    당신은 20년 경력의 전문 의학 에디터입니다. 
    아래 텍스트는 유튜브 영상에서 추출한 STT(Speech-to-Text) 결과로, 오타와 부정확한 문장이 많습니다.

    [지침]
    1. 문맥상 명백한 의학 용어 오타(예: 간함 -> 간암, 상곡보송 -> 상복부 통증)를 정확하게 교정하세요.
    2. 교정된 내용을 바탕으로 환자들이 궁금해할 법한 **[질문(Q) / 답변(A)]** 형식으로 재구성하세요.
    3. 답변은 병원의 전문성이 느껴지도록 친절하고 신뢰감 있게 작성하세요.
    4. 의학적 확진이 아닌 '정보 제공'의 톤(예: "~할 수 있습니다", "~로 보입니다")을 유지하세요.
    5. 불필요한 추임새나 문맥에 맞지 않는 내용은 과감히 삭제하세요.
    6. **반드시 한국어로 작성하세요.**

    [원본 텍스트]:
    {text}

    [정제된 결과 (Q&A 형식)]:
    """,
    input_variables=["text"]
)

chain = refine_prompt | llm | StrOutputParser()

async def fetch_documents(supabase: Client, batch_size: int = 10) -> List[Dict]:
    """Fetch documents that need refinement.
       (We can filter by a flag if we add one, currently just processing chunks)
    """
    # For now, fetch latest documents. 
    # In a real pipeline, you might want a 'refined' boolean column.
    response = supabase.table("documents").select("*").order("id", desc=True).limit(batch_size).execute()
    return response.data

async def refine_and_update(supabase: Client, documents: List[Dict]):
    """Process documents with Gemini and update them."""
    print(f"Processing {len(documents)} documents...")
    
    for doc in documents:
        original_text = doc.get("content", "")
        doc_id = doc.get("id")
        
        if not original_text or len(original_text.strip()) < 10:
            print(f"  - Skipping Doc ID {doc_id} (Text too short)")
            continue

        print(f"  > Refining Doc ID {doc_id}...")
        try:
            # 1. Call Gemini
            refined_text = await chain.ainvoke({"text": original_text})
            
            # 2. Update DB
            # We append the refined Q&A to the content or replace it.
            # Strategy: Replace 'content' with Refined Text for better search, 
            # and move original content to metadata for backup.
            
            new_metadata = doc.get("metadata") or {}
            new_metadata["original_content"] = original_text
            new_metadata["is_refined"] = True
            
            data = {
                "content": refined_text,
                "metadata": new_metadata
            }
            
            supabase.table("documents").update(data).eq("id", doc_id).execute()
            print(f"    -> Updated Doc ID {doc_id}")
            
            # Rate limit protection (Gemini is fast but let's be safe)
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"  !! Error processing Doc ID {doc_id}: {e}")

async def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: Supabase credentials missing in .env")
        return

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    print("=== Starting Data Refinement Pipeline (Gemini 2.0 Flash) ===")
    
    # Process in batches
    batch_size = 10
    total_processed = 0
    max_limit = 100 # Safety limit for one run
    
    while total_processed < max_limit:
        # Fetch documents
        # Ideally, we should filter by 'metadata->>is_refined is null'
        # But since Supabase Python client filter syntax can be tricky with JSONB,
        # we will fetch and check in Python or rely on manual range.
        
        # Let's fetch ALL and filter in python for this script simplicity
        # Or better: Add a filter query if possible.
        
        # Fetching documents where metadata does NOT contain 'is_refined'
        # Note: This query syntax depends on postgrest-py support.
        # Simpler approach: Fetch by ID range or just fetch recent ones.
        
        # Let's just fetch latest 20 for test run
        docs = await fetch_documents(supabase, batch_size)
        
        # Filter out already refined ones locally
        docs_to_process = [d for d in docs if not d.get("metadata", {}).get("is_refined")]
        
        if not docs_to_process:
            print("No unrefined documents found (in this batch).")
            break
            
        await refine_and_update(supabase, docs_to_process)
        total_processed += len(docs_to_process)
        print(f"--- Processed {total_processed} documents so far ---")

    print("=== Refinement Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
