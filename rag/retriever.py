from typing import List, Dict
from database.supabase_client import SupabaseManager
from config.settings import SIMILARITY_THRESHOLD

class Retriever:
    def __init__(self):
        self.db_manager = SupabaseManager()

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieves relevant documents using Hybrid Search (simulated via vector + threshold).
        """
        results = self.db_manager.hybrid_search(query, k=k, threshold=SIMILARITY_THRESHOLD)
        return results
