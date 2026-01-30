import asyncio
import json
import re
import time
from typing import List, Dict
from google import genai
from config.settings import GOOGLE_API_KEY, GENERATION_MODEL
from database.supabase_client import SupabaseManager
from utils.embeddings import get_embedding

# Initialize Gemini Client
client = genai.Client(api_key=GOOGLE_API_KEY)
db_manager = SupabaseManager()

def refine_content_sync(text: str, metadata: Dict) -> str:
    """
    Uses Gemini 2.0 Flash to refine the content into Q&A format.
    Synchronous wrapper for simplicity in batch processing.
    """
    source_title = metadata.get('title', '병원 자료')
    
    prompt = f"""
    너는 암 및 자율신경 치료 전문 병원의 AI 에디터야.
    아래 텍스트는 유튜브 영상 자막이나 블로그 글의 일부야.
    이 내용을 바탕으로 환자들이 궁금해할 만한 핵심 정보를 뽑아 [질문: 답변] 형식의 FAQ 세트로 변환해줘.

    **작성 규칙**:
    1. **오타 수정**: '간함' -> '간암', '상곡보송' -> '상복부 통증' 처럼 문맥에 맞게 의학 용어를 교정해.
    2. **Q&A 형식**: 질문은 환자의 구어체로, 답변은 전문적이고 친절하게 작성해.
    3. **출처 명시**: 답변 끝에 반드시 (출처: {source_title})를 붙여줘.
    4. **내용 없음**: 만약 텍스트에 영양가 있는 의학 정보가 없다면 "NO_CONTENT"라고만 출력해.

    [형식 예시]
    Q: 배가 아픈데 간암일 수 있나요?
    A: 상복부 통증은 간암의 증상 중 하나일 수 있습니다. 하지만 정확한 진단을 위해서는 초음파 검사가 필요합니다. (출처: {source_title})

    ---
    [원본 텍스트]
    {text}
    """
    
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3
            )
        )
        return response.text
    except Exception as e:
        print(f"  !! Gemini Error: {e}")
        return ""

def parse_qa_pairs(refined_text: str, original_metadata: Dict) -> List[Dict]:
    """
    Parses the Q&A text into individual chunks.
    """
    if "NO_CONTENT" in refined_text:
        return []

    qa_pairs = []
    # Split by "Q:" markers
    parts = re.split(r'\nQ:', refined_text)
    
    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue
            
        # Add back "Q:" if missing
        if not clean_part.startswith("Q:"):
            clean_part = "Q:" + clean_part
            
        # Basic validation: must have an answer
        if "A:" not in clean_part:
            continue

        # Create new metadata inheriting from original
        new_metadata = original_metadata.copy()
        new_metadata['type'] = 'faq' # Tag as refined FAQ
        
        qa_pairs.append({
            "content": clean_part,
            "metadata": new_metadata
        })
        
    return qa_pairs

def process_batch(documents: List[Dict]):
    """
    Processes a batch of documents: Refine -> Embed -> Insert
    """
    refined_rows = []
    
    print(f"Processing batch of {len(documents)} documents...")
    
    for doc in documents:
        original_content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        if len(original_content) < 50:
            continue

        # 1. Refine
        refined_text = refine_content_sync(original_content, metadata)
        if not refined_text:
            continue
            
        # 2. Parse
        qa_chunks = parse_qa_pairs(refined_text, metadata)
        
        # 3. Embed & Prepare (Q 부분만 임베딩하여 검색 정확도 향상)
        for chunk in qa_chunks:
            # Q 부분만 추출하여 임베딩
            question_text = SupabaseManager._parse_question(chunk['content'])
            embedding = get_embedding(question_text, task_type="RETRIEVAL_DOCUMENT")
            if embedding:
                refined_rows.append({
                    "content": chunk['content'],
                    "metadata": chunk['metadata'],
                    "embedding": embedding
                })
                
        # Respect rate limits slightly
        time.sleep(1)

    # 4. Insert into hospital_faqs
    if refined_rows:
        print(f"  -> Inserting {len(refined_rows)} refined FAQs into DB...")
        try:
            db_manager.insert_data("hospital_faqs", refined_rows)
        except Exception as e:
            print(f"  !! DB Insert Error: {e}")
    else:
        print("  -> No valid FAQs generated from this batch.")

def main():
    print("Starting Unified Ingestion: documents -> hospital_faqs")
    
    # 1. Fetch all documents from 'documents' table
    # We'll fetch in chunks to handle memory
    page_size = 20
    offset = 250
    total_processed = 0
    
    while True:
        print(f"\nFetching documents offset {offset}...")
        try:
            # Range query for pagination
            response = db_manager.client.table("documents")\
                .select("*")\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            docs = response.data
            if not docs:
                print("No more documents to process.")
                break
                
            # Process this batch
            process_batch(docs)
            
            total_processed += len(docs)
            offset += page_size
            
        except Exception as e:
            print(f"Error fetching documents: {e}")
            break

    print(f"\nCompleted! Processed {total_processed} source documents.")

if __name__ == "__main__":
    main()
