import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.retriever import Retriever

def verify():
    retriever = Retriever()
    query = "고주파온열치료가 효과가 있나요?"
    print(f"Searching for: {query}")
    
    # retrieve is synchronous
    results = retriever.retrieve(query)
    
    if not results:
        print("No results found.")
        return

    print(f"Found {len(results)} results.")
    for i, res in enumerate(results):
        print(f"--- Result {i+1} ---")
        # results are dicts, not objects with page_content attribute
        content = res.get('content', '')
        metadata = res.get('metadata', {})
        print(f"Content: {content[:200]}...")
        print(f"Metadata: {metadata}")
        print("------------------")

if __name__ == "__main__":
    verify()
