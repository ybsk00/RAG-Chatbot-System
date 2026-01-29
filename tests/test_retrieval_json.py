import sys
import os
import asyncio
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import Retriever
from config.settings import SIMILARITY_THRESHOLD

async def test_retrieval():
    print(f"Testing retrieval with SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
    retriever = Retriever()
    query = "고주파온열치료는 무엇인가요?"
    
    results = retriever.retrieve(query)
    
    output_data = {
        "query": query,
        "threshold": SIMILARITY_THRESHOLD,
        "count": len(results),
        "results": []
    }
    
    if results:
        for i, doc in enumerate(results):
            output_data["results"].append({
                "index": i,
                "similarity": doc.get('similarity', 'N/A'),
                "content_preview": doc['content'][:500] # First 500 chars
            })
    
    # Write to file directly
    with open("tests/retrieval_result.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print("Results written to tests/retrieval_result.json")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
