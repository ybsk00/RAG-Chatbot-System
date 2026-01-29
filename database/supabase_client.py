import os
import time
import random
from typing import List, Dict, Any
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY
from utils.embeddings import get_embedding, get_query_embedding

class SupabaseManager:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not found.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def insert_documents(self, documents: List[Dict]):
        """
        Inserts processed chunks into Supabase.
        documents: List of dicts with 'content' and 'metadata'.
        """
        rows = []
        
        # Base ID generation
        base_id = int(time.time()) * 1000
        
        print(f"Generating embeddings for {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            content = doc.get('content')
            metadata = doc.get('metadata', {})
            
            if not content:
                continue
                
            # Generate embedding using the direct util
            embedding = get_embedding(content)
            
            if not embedding:
                print(f"Skipping document {i}: Embedding generation failed.")
                continue

            # Generate ID
            # Using str for UUID compatibility if needed, but schema uses uuid default gen_random_uuid?
            # The previous code passed explicit IDs. Let's stick to letting Supabase generate UUIDs 
            # OR pass them if we really want to control them.
            # The previous code used BigInt IDs for a UUID column? That would fail.
            # Let's check the SQL schema in create_table_sql.
            # "id uuid primary key default gen_random_uuid()"
            # So we should NOT pass integer IDs. We should let Supabase generate UUIDs or generate UUIDs here.
            # I will omit 'id' so Supabase generates it.
            
            row = {
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            }
            rows.append(row)

        if not rows:
            return

        # Insert in batches to avoid payload limits
        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            try:
                data = self.client.table("documents").insert(batch).execute()
                # print(f"Inserted batch {i//batch_size + 1}")
            except Exception as e:
                print(f"Error inserting batch: {e}")

        print(f"Inserted {len(rows)} documents into Supabase.")

    def hybrid_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """
        Performs vector similarity search using a Supabase RPC function.
        """
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            return []

        params = {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": k
        }

        try:
            response = self.client.rpc("match_documents", params).execute()
            # response.data is a list of dicts: [{'content': ..., 'metadata': ..., 'similarity': ...}]
            return response.data
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def keyword_search(self, query_text: str, k: int = 5, metadata_filter: Dict = None) -> List[Dict]:
        """
        Performs a simple keyword/text search using ilike.
        Useful as a fallback or supplement to vector search.
        """
        try:
            # Simple keyword extraction: split by space and take tokens > 2 chars
            # This is a naive heuristic. In production, use an NLP extractor.
            keywords = [word for word in query_text.split() if len(word) >= 2]
            
            if not keywords:
                return []

            # Construct a robust ILIKE query for ANY of the keywords
            # Supabase-py 'or_' filter syntax: "content.ilike.%keyword1%,content.ilike.%keyword2%"
            or_filter = ",".join([f"content.ilike.%{kw}%" for kw in keywords])
            
            query_builder = self.client.table("documents")\
                .select("id, content, metadata")\
                .or_(or_filter)
            
            # Apply metadata filter if provided
            if metadata_filter:
                query_builder = query_builder.match({"metadata": metadata_filter})

            # Since we can't easily do partial JSON matching in simple query builder without more complex syntax,
            # we'll assume exact match on the top-level keys if passed, or use the 'contains' operator on the jsonb column.
            # However, supabase-py 'match' does exact match on columns.
            # For JSONB column 'metadata', we should use .contains('metadata', filter_dict).
            
            if metadata_filter:
                # Override the previous match attempt which might be wrong for JSONB
                # Re-initializing purely to be safe isn't needed, just chaining correctly.
                # Actually, let's use the 'contains' filter for JSONB.
                query_builder = self.client.table("documents")\
                    .select("id, content, metadata")\
                    .or_(or_filter)\
                    .contains("metadata", metadata_filter)

            response = query_builder.limit(k).execute()
                
            # Add a dummy similarity score for compatibility
            results = []
            for item in response.data:
                item['similarity'] = 1.0 # Treat keyword matches as high relevance
                results.append(item)
                
            return results
        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def create_table_sql(self):
        """Returns the SQL needed to set up the table in Supabase."""
        return """
        -- Enable the pgvector extension to work with embedding vectors
        create extension if not exists vector;

        -- Create a table to store your documents
        create table if not exists documents (
        id uuid primary key default gen_random_uuid(),
        content text, 
        metadata jsonb, 
        embedding vector(768) 
        );

        -- Create a function to search for documents
        create or replace function match_documents (
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