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

from fastapi.responses import FileResponse, StreamingResponse
import json

# ... (existing imports)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized properly.")
    
    query = request.query
    category = request.category
    print(f"Received query: {query}, Category: {category}")
    
    async def response_generator():
        try:
            import asyncio
            import time
            
            start_time = time.time()
            
            # 1. Retrieve Documents (Blocking I/O wrapped in thread)
            # We can't easily parallelize routing here if we want to stream immediately, 
            # but routing is fast. Let's do retrieval first.
            context_docs = await asyncio.to_thread(retriever.retrieve, query)
            
            retrieval_end_time = time.time()
            print(f"[Timing] Retrieval took: {retrieval_end_time - start_time:.4f}s")
            
            # 2. Generate Stream
            # We need to pass the generator to iterate over
            # Since generate_answer_stream is synchronous generator, we iterate it
            # If we want async streaming, we might need to wrap it or use run_in_executor for each chunk?
            # Actually, for simple text streaming, iterating a sync generator in async function 
            # might block the loop if chunks take time. 
            # But here the generator yields quickly after network calls.
            # Ideally `generate_answer_stream` should be async or we run it in thread.
            # But `google.genai` stream might be sync.
            # Let's assume it's sync for now and just iterate.
            # To avoid blocking, we can use `asyncio.to_thread` for the whole generation 
            # but that defeats streaming purpose if we wait for all.
            # We need an async iterator wrapper if the underlying lib is sync blocking.
            
            # However, `generator.generate_answer_stream` calls `client.models.generate_content_stream`
            # which returns an iterable.
            
            # Let's try to iterate directly. If it blocks, it blocks this request processing.
            # For better concurrency, we should run the blocking generation in a thread 
            # and push to a queue, but that's complex.
            # Let's stick to direct iteration for MVP.
            
            stream = generator.generate_answer_stream(query, context_docs, category)
            
            for chunk in stream:
                yield chunk
                # await asyncio.sleep(0) # Yield control to event loop
            
            # 3. Send Sources
            sources = []
            if context_docs:
                for i, doc in enumerate(context_docs):
                    metadata = doc.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    try:
                        if metadata and isinstance(metadata, dict) and 'source' in metadata:
                            sources.append(metadata['source'])
                    except Exception as e:
                        print(f"[Warning] Failed to extract source from doc {i}: {e}")

            if sources:
                # Send a delimiter and then the sources as JSON
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
