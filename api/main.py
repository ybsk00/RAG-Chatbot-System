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
    
    query = request.query
    category = request.category
    history = request.history
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
            
            for chunk in stream:
                yield chunk
                await asyncio.sleep(0) # Yield control to event loop
            
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
                        if metadata and isinstance(metadata, dict):
                            # Append the whole metadata object or a subset
                            # We need 'source' for the URL and 'title' for display
                            sources.append(metadata)
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
