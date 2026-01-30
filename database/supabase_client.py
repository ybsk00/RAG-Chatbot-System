import os
import time
import random
from typing import List, Dict, Any
from supabase import create_client, Client
from config.settings import (
    SUPABASE_URL, SUPABASE_KEY,
    DOCUMENTS_TABLE, HOSPITAL_FAQS_TABLE
)
from utils.embeddings import get_embedding, get_query_embedding


class SupabaseManager:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not found.")
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    def insert_data(self, table_name: str, data: List[Dict]):
        """범용 데이터 삽입 메서드."""
        if not data:
            return

        try:
            batch_size = 50
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                self.client.table(table_name).insert(batch).execute()
            print(f"Inserted {len(data)} rows into {table_name}.")
        except Exception as e:
            print(f"Error inserting into {table_name}: {e}")
            raise e

    def insert_documents(self, documents: List[Dict], table_name: str = None):
        """
        문서를 임베딩과 함께 Supabase에 삽입합니다.
        table_name: 'documents' (bigserial id) 또는 'hospital_faqs' (uuid id)
        """
        if table_name is None:
            table_name = DOCUMENTS_TABLE

        rows = []
        print(f"Generating embeddings for {len(documents)} documents -> {table_name}...")

        for i, doc in enumerate(documents):
            content = doc.get('content')
            metadata = doc.get('metadata', {})

            if not content:
                continue

            embedding = get_embedding(content)
            if not embedding:
                print(f"Skipping document {i}: Embedding generation failed.")
                continue

            # documents: bigserial (자동 증가) → id 생략
            # hospital_faqs: uuid (gen_random_uuid()) → id 생략
            row = {
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            }
            rows.append(row)

        if not rows:
            return

        batch_size = 50
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            try:
                self.client.table(table_name).insert(batch).execute()
            except Exception as e:
                print(f"Error inserting batch into {table_name}: {e}")

        print(f"Inserted {len(rows)} documents into {table_name}.")

    # ──────────────────────────────────────────────
    # 벡터 검색 (RPC 기반)
    # ──────────────────────────────────────────────

    def hybrid_search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """documents 테이블 벡터 유사도 검색 (match_documents RPC)."""
        return self._rpc_vector_search("match_documents", query, k, threshold)

    def hybrid_search_faqs(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """hospital_faqs 테이블 벡터 유사도 검색 (match_hospital_faqs RPC)."""
        return self._rpc_vector_search("match_hospital_faqs", query, k, threshold)

    def _rpc_vector_search(self, rpc_name: str, query: str, k: int, threshold: float) -> List[Dict]:
        """RPC 함수를 호출하는 공통 벡터 검색 로직."""
        query_embedding = get_query_embedding(query)
        if not query_embedding:
            return []

        params = {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": k
        }

        try:
            response = self.client.rpc(rpc_name, params).execute()
            return response.data
        except Exception as e:
            print(f"Error during {rpc_name} search: {e}")
            return []

    # ──────────────────────────────────────────────
    # 키워드 검색
    # ──────────────────────────────────────────────

    # 한국어 불용어 (조사, 어미, 질문 접미사)
    _STOPWORDS = {
        "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로",
        "와", "과", "도", "만", "까지", "부터", "에게", "한테", "께",
        "하는", "하고", "해서", "하면", "합니다", "입니다", "있는", "없는",
        "어떤", "무엇", "어떻게", "왜", "좀", "것", "수", "때", "거",
        "알려줘", "알려주세요", "뭐야", "뭔가요", "인가요", "건가요",
        "대해", "대해서", "관해", "관해서", "뭐예요", "무엇인가요",
        "어떻", "그런", "이런", "저런", "있나요", "없나요", "해주세요"
    }

    # 한국어 조사/어미 접미사 (단어 끝에서 분리)
    _PARTICLES = [
        "에서는", "에서도", "에서의", "으로는", "으로도", "에서",
        "에게는", "에게도", "에게", "한테는", "한테도", "한테",
        "으로", "로는", "로도",
        "이란", "이라", "이든", "이나", "이고", "이에",
        "에는", "에도", "에의",
        "은요", "는요", "이요",
        "과는", "와는",
        "까지", "부터", "마저", "조차", "밖에",
        "은", "는", "이", "가", "을", "를", "의", "에", "로",
        "와", "과", "도", "만", "요"
    ]

    @staticmethod
    def _strip_particles(word: str) -> str:
        """한국어 단어에서 조사/어미를 분리합니다."""
        for p in SupabaseManager._PARTICLES:
            if word.endswith(p) and len(word) > len(p) + 1:
                return word[:-len(p)]
        return word

    @staticmethod
    def _extract_keywords(query_text: str) -> List[str]:
        """한국어 불용어 제거 + 조사 분리로 키워드를 추출합니다."""
        tokens = query_text.replace("?", "").replace("!", "").replace(".", "").split()
        keywords = []
        for w in tokens:
            if len(w) < 2 or w in SupabaseManager._STOPWORDS:
                continue
            stripped = SupabaseManager._strip_particles(w)
            if len(stripped) >= 2 and stripped not in SupabaseManager._STOPWORDS:
                keywords.append(stripped)
            elif len(w) >= 2:
                keywords.append(w)

        # 중복 제거 (순서 유지)
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        if not unique and tokens:
            unique = sorted(tokens, key=len, reverse=True)[:3]
        return unique

    def keyword_search(self, query_text: str, k: int = 5,
                       metadata_filter: Dict = None,
                       table_name: str = None) -> List[Dict]:
        """키워드 오버랩 비율 기반 점수를 적용한 키워드 검색."""
        if table_name is None:
            table_name = DOCUMENTS_TABLE

        try:
            keywords = self._extract_keywords(query_text)
            if not keywords:
                return []

            or_filter = ",".join([f"content.ilike.%{kw}%" for kw in keywords])

            query_builder = self.client.table(table_name)\
                .select("id, content, metadata")

            query_builder = query_builder.or_(or_filter)

            if metadata_filter:
                query_builder = query_builder.contains("metadata", metadata_filter)

            response = query_builder.limit(k * 2).execute()

            # 키워드 오버랩 비율로 점수 계산
            results = []
            for item in response.data:
                content_lower = item.get('content', '').lower()
                matched = sum(1 for kw in keywords if kw.lower() in content_lower)
                score = round(matched / len(keywords), 2) if keywords else 0.0
                if score >= 0.3:
                    item['similarity'] = score
                    results.append(item)

            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
        except Exception as e:
            print(f"Error during keyword search ({table_name}): {e}")
            return []

    # ──────────────────────────────────────────────
    # 스키마 DDL (참고용)
    # ──────────────────────────────────────────────

    def create_table_sql(self):
        """실제 DB 스키마에 대응하는 DDL (참고용)."""
        return """
        -- Enable the pgvector extension
        create extension if not exists vector;

        -- documents 테이블 (id: bigserial 자동 증가)
        create table if not exists documents (
            id bigserial primary key,
            content text,
            metadata jsonb,
            embedding vector(768)
        );

        create index if not exists documents_embedding_idx
            on documents using hnsw (embedding vector_cosine_ops);

        -- hospital_faqs 테이블 (id: uuid 자동 생성)
        create table if not exists hospital_faqs (
            id uuid primary key default gen_random_uuid(),
            content text not null,
            metadata jsonb,
            embedding vector(768)
        );

        -- documents 벡터 검색 함수
        create or replace function match_documents (
            query_embedding vector(768),
            match_threshold float,
            match_count int
        )
        returns table (
            id bigint,
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

        -- hospital_faqs 벡터 검색 함수
        create or replace function match_hospital_faqs (
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
            hospital_faqs.id,
            hospital_faqs.content,
            hospital_faqs.metadata,
            1 - (hospital_faqs.embedding <=> query_embedding) as similarity
        from hospital_faqs
        where 1 - (hospital_faqs.embedding <=> query_embedding) > match_threshold
        order by similarity desc
        limit match_count;
        end;
        $$;
        """
