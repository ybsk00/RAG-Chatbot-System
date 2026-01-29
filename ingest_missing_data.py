import json
import os
import sys
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.supabase_client import SupabaseManager

async def ingest_from_json():
    file_path = "documents_rows (2).json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    print(f"Found {len(rows)} documents.")
    
    db_manager = SupabaseManager()
    
    docs_to_insert = []
    for row in rows:
        content = row.get('content')
        if not content:
            continue
            
        metadata = row.get('metadata')
        # Parse metadata if it's a string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse metadata for doc {row.get('id')}")
                metadata = {}
        
        docs_to_insert.append({
            "content": content,
            "metadata": metadata
        })
    
    print(f"Prepared {len(docs_to_insert)} documents for ingestion.")
    
    # Insert in batches (SupabaseManager handles batching but let's call it once if it handles list)
    # SupabaseManager.insert_documents iterates and inserts in batches of 50.
    # However, it re-generates embeddings. This might take time and cost quota.
    # Given we have embeddings in the JSON, maybe we should use them?
    # But the JSON has embeddings as STRINGS "[0.01, ...]".
    # And Supabase expects vector type.
    # If we use db_manager.insert_documents, it calls get_embedding().
    # Let's stick to get_embedding() to be safe and consistent with current code logic.
    
    db_manager.insert_documents(docs_to_insert)
    print("Ingestion complete.")

if __name__ == "__main__":
    # SupabaseManager is synchronous in insert_documents (it uses .execute()), 
    # but we wrap in async just in case we need async later or for consistency.
    # Actually insert_documents is sync.
    ingest_from_json() # It's a sync function in the script above? 
    # Wait, the script defines `async def ingest_from_json`, but `db_manager.insert_documents` is sync.
    # So we can just run it.
    # But `ingest_from_json` is async defined.
    asyncio.run(ingest_from_json())
