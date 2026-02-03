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
from config.medical_synonyms import get_synonyms


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

    def update_row(self, table_name: str, row_id: str, data: dict):
        """특정 행의 필드를 업데이트합니다."""
        try:
            self.client.table(table_name).update(data).eq("id", row_id).execute()
        except Exception as e:
            print(f"Error updating row {row_id} in {table_name}: {e}")
            raise e

    @staticmethod
    def _parse_question(content: str) -> str:
        """'Q: ...\\nA: ...' 형식에서 질문만 추출합니다."""
        if "Q:" in content and "A:" in content:
            q_start = content.index("Q:") + 2
            a_start = content.index("A:")
            return content[q_start:a_start].strip()
        return content.split("\n")[0].strip()

    def insert_documents(self, documents: List[Dict], table_name: str = None):
        """
        문서를 임베딩과 함께 Supabase에 삽입합니다.
        table_name: 'documents' (bigserial id) 또는 'hospital_faqs' (uuid id)
        hospital_faqs의 경우 Q 부분만 임베딩하여 검색 정확도를 높입니다.
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

            # hospital_faqs: Q 부분만 임베딩 (쿼리↔질문 매칭 향상)
            if table_name == HOSPITAL_FAQS_TABLE:
                embed_text = self._parse_question(content)
            else:
                embed_text = content

            embedding = get_embedding(embed_text)
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
        "어떻", "그런", "이런", "저런", "있나요", "없나요", "해주세요",
        "싶어", "싶은", "싶다", "싶어요", "싶습니다", "소개", "설명", "궁금", "정보"
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
        "하고", "해서", "해요", "하면", "할까",
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

    @staticmethod
    def _expand_compound_keywords(keywords: List[str]) -> List[str]:
        """복합어 키워드를 3글자 서브워드로 확장합니다 (ilike 검색용).
        암 질환명(XX암)은 '암', '암치료', '암환자' 등 공통 키워드도 추가합니다."""
        expanded = list(keywords)

        # 암 질환명 패턴 감지 → 공통 암 키워드 추가
        cancer_base_terms = ["암", "암치료", "암환자", "항암", "암보조"]
        for kw in keywords:
            if kw.endswith("암") and len(kw) >= 2 and kw != "암":
                for term in cancer_base_terms:
                    if term not in expanded:
                        expanded.append(term)

        for kw in keywords:
            if len(kw) >= 4:
                for i in range(0, len(kw) - 2):
                    sub = kw[i:i+3]
                    if sub not in expanded and sub not in SupabaseManager._STOPWORDS:
                        expanded.append(sub)
        return expanded

    @staticmethod
    def _expand_synonyms(keywords: List[str]) -> List[str]:
        """의료 동의어 사전으로 키워드를 확장합니다."""
        expanded = list(keywords)
        for kw in keywords:
            for syn in get_synonyms(kw):
                if syn not in expanded:
                    expanded.append(syn)
        return expanded

    def keyword_search(self, query_text: str, k: int = 5,
                       metadata_filter: Dict = None,
                       table_name: str = None) -> List[Dict]:
        """키워드 오버랩 비율 기반 점수를 적용한 키워드 검색 (동의어 확장 포함)."""
        if table_name is None:
            table_name = DOCUMENTS_TABLE

        try:
            keywords = self._extract_keywords(query_text)
            if not keywords:
                return []

            # 동의어 확장 → 복합어 확장 → ilike 검색 (발견 범위 확대)
            synonyms_expanded = self._expand_synonyms(keywords)
            search_terms = self._expand_compound_keywords(synonyms_expanded)
            or_filter = ",".join([f"content.ilike.%{kw}%" for kw in search_terms])

            query_builder = self.client.table(table_name)\
                .select("id, content, metadata")

            query_builder = query_builder.or_(or_filter)

            if metadata_filter:
                query_builder = query_builder.contains("metadata", metadata_filter)

            response = query_builder.limit(k * 3).execute()

            # 키워드 오버랩 비율로 점수 계산 (동의어 매칭 포함)
            results = []
            for item in response.data:
                content_normalized = item.get('content', '').replace(' ', '').lower()
                matched = sum(
                    1 for kw in keywords
                    if any(
                        term.lower().replace(' ', '') in content_normalized
                        for term in [kw] + get_synonyms(kw)
                    )
                )
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
