import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.supabase_client import SupabaseManager
from config.settings import (
    SIMILARITY_THRESHOLD, MAX_CONTEXT_DOCS, MAX_CONTEXT_CHARS,
    RESULT_CACHE_SIZE, RESULT_CACHE_TTL_SECONDS,
    HOSPITAL_FAQS_TABLE
)


class Retriever:
    def __init__(self):
        self.db_manager = SupabaseManager()
        self._cache: Dict[str, dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_cached(self, key: str) -> Optional[List[Dict]]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry["timestamp"]) < RESULT_CACHE_TTL_SECONDS:
            return entry["results"]
        if entry:
            del self._cache[key]
        return None

    def _set_cache(self, key: str, results: List[Dict]):
        if len(self._cache) >= RESULT_CACHE_SIZE:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        self._cache[key] = {"results": results, "timestamp": time.time()}

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        하이브리드 검색: documents + hospital_faqs 벡터/키워드 병렬 실행
        → 병합 → 재순위화 → 컨텍스트 절단.
        """
        # 캐시 확인
        cached = self._get_cached(query)
        if cached is not None:
            print(f"[Cache HIT] {query[:30]}...")
            return cached

        # 4개 검색을 병렬 실행
        futures = {
            # 1) documents 벡터 검색
            self._executor.submit(
                self.db_manager.hybrid_search, query, k, SIMILARITY_THRESHOLD
            ): "docs_vector",
            # 2) hospital_faqs 벡터 검색
            self._executor.submit(
                self.db_manager.hybrid_search_faqs, query, k, SIMILARITY_THRESHOLD
            ): "faqs_vector",
            # 3) documents 키워드 검색 (YouTube 우선)
            self._executor.submit(
                self.db_manager.keyword_search, query, k, {"type": "youtube"}
            ): "youtube_kw",
            # 4) documents 키워드 검색 (일반)
            self._executor.submit(
                self.db_manager.keyword_search, query, k
            ): "general_kw",
        }

        search_results = {}
        for future in as_completed(futures):
            label = futures[future]
            try:
                search_results[label] = future.result(timeout=8)
            except Exception as e:
                print(f"[Retriever] {label} search failed: {e}")
                search_results[label] = []

        docs_vector = search_results.get("docs_vector", [])
        faqs_vector = search_results.get("faqs_vector", [])
        youtube_kw = search_results.get("youtube_kw", [])
        general_kw = search_results.get("general_kw", [])

        print(f"[Retriever] docs_vector={len(docs_vector)}, faqs_vector={len(faqs_vector)}, "
              f"yt_kw={len(youtube_kw)}, gen_kw={len(general_kw)}")

        # 병합 + 중복 제거 (높은 유사도 보존)
        # hospital_faqs는 정제된 고품질 데이터이므로 우선 반영
        merged_docs = {}
        for source_list in [faqs_vector, youtube_kw, docs_vector, general_kw]:
            for doc in source_list:
                doc_id = doc.get('id') or hash(doc.get('content', ''))
                if doc_id not in merged_docs:
                    merged_docs[doc_id] = doc
                else:
                    existing = merged_docs[doc_id]
                    if doc.get('similarity', 0) > existing.get('similarity', 0):
                        merged_docs[doc_id] = doc

        # 유사도 기준 내림차순 정렬
        ranked = sorted(merged_docs.values(), key=lambda d: d.get('similarity', 0), reverse=True)

        # 컨텍스트 절단 (글자 수 + 문서 수 제한)
        final_results = []
        total_chars = 0
        for doc in ranked:
            content = doc.get('content', '')
            if total_chars + len(content) > MAX_CONTEXT_CHARS:
                break
            if len(final_results) >= MAX_CONTEXT_DOCS:
                break
            final_results.append(doc)
            total_chars += len(content)

        print(f"[Retriever] merged={len(merged_docs)}, final={len(final_results)}, "
              f"total_chars={total_chars}")

        self._set_cache(query, final_results)
        return final_results
