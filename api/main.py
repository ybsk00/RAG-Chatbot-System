import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag.retriever import Retriever
from rag.generator import Generator
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

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized properly.")
    
    query = request.query
    category = request.category
    print(f"Received query: {query}, Category: {category}")
    
    try:
        import asyncio
        import time
        
        start_time = time.time()
        
        # 1. Parallelize Retrieval and Routing
        tasks = []
        
        # Task 1: Retrieve Documents (Blocking I/O wrapped in thread)
        tasks.append(asyncio.to_thread(retriever.retrieve, query))
        
        # Task 2: Classify Query (Blocking I/O wrapped in thread) - Only if auto
        if category == "auto":
            tasks.append(asyncio.to_thread(generator.classify_query, query))
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks)
        
        retrieval_end_time = time.time()
        print(f"[Timing] Retrieval & Classification took: {retrieval_end_time - start_time:.4f}s")
        
        context_docs = results[0]
        if category == "auto":
            category = results[1]
            print(f"Auto-routed category (Parallel): {category}")
        
        # 2. Generate with Category Context
        gen_start_time = time.time()
        answer = await asyncio.to_thread(generator.generate_answer, query, context_docs, category)
        gen_end_time = time.time()
        print(f"[Timing] Generation took: {gen_end_time - gen_start_time:.4f}s")
        print(f"[Timing] Total processing time: {gen_end_time - start_time:.4f}s")
        
        # Extract sources for API response
        sources = []
        if context_docs:
            print(f"[Debug] Raw metadata from {len(context_docs)} docs:")
            for i, doc in enumerate(context_docs):
                metadata = doc.get('metadata', {})
                print(f"  Doc {i} metadata type: {type(metadata)}, value: {metadata}")
                
                # Fix: Handle metadata if it's a JSON string
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        print(f"Error parsing metadata JSON: {metadata}")
                        metadata = {}

                try:
                    if metadata and isinstance(metadata, dict) and 'source' in metadata:
                        sources.append(metadata['source'])
                except Exception as e:
                    print(f"[Warning] Failed to extract source from doc {i}: {e}")
                    print(f"  Metadata content: {metadata}")
        
        print(f"[Debug] Extracted sources: {sources}")
        return ChatResponse(answer=answer, sources=sources)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
