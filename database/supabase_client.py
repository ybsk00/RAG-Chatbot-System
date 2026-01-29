import os
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
from utils.embeddings import get_embedding_model
from config.settings import SUPABASE_URL, SUPABASE_KEY
from typing import List, Dict, Any

class SupabaseManager:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not found.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embeddings = get_embedding_model()
        self.vector_store = SupabaseVectorStore(
            client=self.client,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents"
        )

    def insert_documents(self, documents: List[Dict]):
        """
        Inserts processed chunks into Supabase.
        documents: List of dicts with 'content' and 'metadata'.
        """
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Generate Integer IDs for bigserial column
        # Using current timestamp + index to create unique integers
        import time
        import random
        
        # Base ID: timestamp (seconds) * 1000 + random offset
        # This fits in bigint and reduces collision
        base_id = int(time.time()) * 1000
        ids = [str(base_id + i + random.randint(0, 999)) for i in range(len(documents))]
        
        # Pass explicit integer IDs
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Inserted {len(documents)} documents into Supabase.")

    def hybrid_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        # ... (rest of the code)
        return filtered_results

    def create_table_sql(self):
        """Returns the SQL needed to set up the table in Supabase."""
        return """
        -- Enable the pgvector extension to work with embedding vectors
        create extension if not exists vector;

        -- Create a table to store your documents
        create table documents (
        id uuid primary key default gen_random_uuid(),
        content text, -- corresponds to Document.pageContent
        metadata jsonb, -- corresponds to Document.metadata
        embedding vector(768) -- 768 dimensions for Gemini embeddings
        );

        -- Create a function to search for documents
        create function match_documents (
        query_embedding vector(768),
        match_threshold float,
        match_count int
        )
        returns table (
        id uuid,
        content text,
        metadata jsonb,
        similarity float
        )
        language plpgsql
        as $$
        begin
        return query
        select
            documents.id,
            documents.content,
            documents.metadata,
            1 - (documents.embedding <=> query_embedding) as similarity
        from documents
        where 1 - (documents.embedding <=> query_embedding) > match_threshold
        order by documents.embedding <=> query_embedding
        limit match_count;
        end;
        $$;
        """
