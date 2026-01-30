import os
from functools import lru_cache
from google import genai
from config.settings import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_CACHE_SIZE

client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)


def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """문서 임베딩 생성 (인제스트용, 캐시 불필요)."""
    if not client:
        raise ValueError("GOOGLE_API_KEY is not set.")

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=genai.types.EmbedContentConfig(
                task_type=task_type
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def _get_query_embedding_cached(text: str) -> tuple:
    """LRU 캐시 적용 내부 함수 (tuple 반환으로 hashable 보장)."""
    if not client:
        raise ValueError("GOOGLE_API_KEY is not set.")

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=genai.types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        return tuple(result.embeddings[0].values)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return ()


def get_query_embedding(text: str) -> list:
    """쿼리 임베딩 생성 (LRU 캐시 적용)."""
    result = _get_query_embedding_cached(text)
    return list(result) if result else []
