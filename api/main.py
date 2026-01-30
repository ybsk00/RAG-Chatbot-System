import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag.retriever import Retriever
from rag.generator import Generator
from rag.safety import SafetyGuard
import uvicorn

app = FastAPI(title="OnCare Clinic AI Chatbot")

# Initialize RAG components
# We initialize them here to load once on startup
try:
    retriever = Retriever()
    generator = Generator()
    print("RAG Components Initialized Successfully.")
except Exception as e:
    print(f"Error initializing RAG components: {e}")
    # We don't raise here to allow app to start, but endpoints might fail
    retriever = None
    generator = None

class ChatRequest(BaseModel):
    query: str
    category: str = "auto" # auto, cancer, nerve, general
    history: list = [] # List of {"role": "user"|"model", "content": "..."}

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)

from fastapi.responses import FileResponse, StreamingResponse
import json

# ... (existing imports)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized properly.")
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    category = request.category
    history = SafetyGuard.validate_history(request.history)
    print(f"Received query: {query}, Category: {category}, History length: {len(history)}")
    
    async def response_generator():
        try:
            import asyncio
            import time
            
            start_time = time.time()
            
            # 1. Parallelize Classification and Retrieval
            # We run both tasks concurrently to save time
            
            # Task 1: Retrieval (Blocking I/O wrapped in thread)
            retrieval_task = asyncio.create_task(asyncio.to_thread(retriever.retrieve, query))
            
            # Task 2: Classification (Network call, usually fast but better parallel)
            # Since generator.classify_query is sync, we wrap it too
            classification_task = asyncio.create_task(asyncio.to_thread(generator.classify_query, query))
            
            # Wait for both
            context_docs, classified_category = await asyncio.gather(retrieval_task, classification_task)
            
            # Use the classified category if request was "auto"
            final_category = classified_category if category == "auto" else category
            print(f"Auto-routed category: {final_category} (Original: {category})")
            
            retrieval_end_time = time.time()
            print(f"[Timing] Retrieval & Classification took: {retrieval_end_time - start_time:.4f}s")
            
            # 2. Generate Stream
            stream = generator.generate_answer_stream(query, context_docs, final_category, history)

            is_fallback = False
            for chunk in stream:
                if chunk.startswith("[일반 의학 정보 안내]"):
                    is_fallback = True
                yield chunk
                await asyncio.sleep(0)

            # 3. Send Sources (조건부)
            # - general 카테고리(인사, 일상): 유튜브 소스 없음
            # - 폴백 답변: 소스 없음
            # - RAG 의료 답변: 제목에 키워드 매칭되는 유튜브만
            sources = []
            if context_docs and final_category != "general" and not is_fallback:
                # 쿼리에서 핵심 키워드 추출
                from database.supabase_client import SupabaseManager
                query_keywords = SupabaseManager._extract_keywords(query)

                for doc in context_docs:
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}

                    if not metadata or not isinstance(metadata, dict):
                        continue

                    source_url = metadata.get('source', '')
                    title = metadata.get('title', '')

                    # 유튜브 소스: 제목에 쿼리 키워드가 포함된 경우만
                    is_youtube = 'youtube.com' in source_url or 'youtu.be' in source_url
                    if is_youtube and title and query_keywords:
                        title_normalized = title.replace(' ', '').lower()
                        has_keyword = any(
                            kw.lower().replace(' ', '') in title_normalized
                            for kw in query_keywords
                        )
                        if has_keyword:
                            sources.append(metadata)
                    elif not is_youtube and metadata.get('source'):
                        # 블로그 등 비유튜브 소스는 그대로 포함
                        sources.append(metadata)

                # 유튜브 소스 중복 제거 (URL 기준)
                seen_urls = set()
                unique_sources = []
                for s in sources:
                    url = s.get('source', '')
                    if url not in seen_urls:
                        seen_urls.add(url)
                        unique_sources.append(s)
                sources = unique_sources

            if sources:
                yield f"\n\n__SOURCES__\n{json.dumps(sources)}"
                
        except Exception as e:
            print(f"Error processing request: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
