import sys
import os
import asyncio
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import Retriever
from config.settings import SIMILARITY_THRESHOLD

async def test_retrieval():
    # Force stdout to utf-8
    sys.stdout.reconfigure(encoding='utf-8')
    
    print(f"Testing retrieval with SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}")
    retriever = Retriever()
    query = "고주파온열치료는 무엇인가요?"
    
    print(f"Query: {query}")
    results = retriever.retrieve(query)
    
    if results:
        print(f"✅ Success! Found {len(results)} documents.")
        for i, doc in enumerate(results):
            print(f"\n--- Doc {i+1} ---")
            print(f"Similarity: {doc.get('similarity', 'N/A')}")
            content = doc['content']
            # Print first 200 chars
            print(f"Content Preview: {content[:200]}...")
            print("-" * 20)
    else:
        print("❌ Failed. No documents found.")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
