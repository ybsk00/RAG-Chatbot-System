from typing import List, Dict
from database.supabase_client import SupabaseManager
from config.settings import SIMILARITY_THRESHOLD

class Retriever:
    def __init__(self):
        self.db_manager = SupabaseManager()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieves relevant documents using Hybrid Search (Vector + Keyword).
        Specific Logic:
        - YouTube: Prioritizes STRICT keyword matching to avoid irrelevant videos.
        - General: Uses Vector search for semantic understanding.
        """
        # 1. Vector Search (General Context - mostly Blog/Text)
        # We can try to exclude youtube from vector search if we want, but keeping it is fine as fallback.
        # However, user requested "YouTube videos should be keyword-centered".
        vector_results = self.db_manager.hybrid_search(query, k=k, threshold=SIMILARITY_THRESHOLD)
        print(f"Vector search found {len(vector_results)} docs.")

        # 2. Keyword Search (Specific for YouTube)
        # We enforce finding YouTube videos that explicitly contain the keywords.
        youtube_keyword_results = self.db_manager.keyword_search(query, k=k, metadata_filter={"type": "youtube"})
        print(f"YouTube Keyword search found {len(youtube_keyword_results)} docs.")
        
        # 3. Keyword Search (General Fallback)
        # We still want keyword fallback for blogs too, just in case vector misses it.
        general_keyword_results = self.db_manager.keyword_search(query, k=k) # This searches all types
        
        # 4. Merge and Deduplicate
        merged_docs = {}
        
        # Strategy: 
        # - Add YouTube Keyword Results FIRST (Highest Priority)
        # - Then Vector Results (Semantic matches)
        # - Then General Keyword Results (Catch-all)
        
        all_sources = [youtube_keyword_results, vector_results, general_keyword_results]
        
        for source_list in all_sources:
            for doc in source_list:
                doc_id = doc.get('id') or hash(doc.get('content'))
                if doc_id not in merged_docs:
                    merged_docs[doc_id] = doc
        
        final_results = list(merged_docs.values())
        print(f"Total merged docs: {len(final_results)}")
        
        # Limit to k (or slightly more to give context)
        return final_results[:k+2]
