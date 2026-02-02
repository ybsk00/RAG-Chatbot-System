import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag.retriever import Retriever
from rag.generator import Generator
from rag.safety import SafetyGuard
import uvicorn

app = FastAPI(title="OnCare Clinic AI Chatbot")

# Initialize RAG components
# We initialize them here to load once on startup
try:
    retriever = Retriever()
    generator = Generator()
    print("RAG Components Initialized Successfully.")
except Exception as e:
    print(f"Error initializing RAG components: {e}")
    # We don't raise here to allow app to start, but endpoints might fail
    retriever = None
    generator = None

class ChatRequest(BaseModel):
    query: str
    category: str = "auto" # auto, cancer, nerve, general
    history: list = [] # List of {"role": "user"|"model", "content": "..."}

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/")
async def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    return FileResponse(file_path)

from fastapi.responses import FileResponse, StreamingResponse
import json
import re

# ── 내원 의사 감지 (하이브리드: 키워드 1차 → LLM 맥락 확인) ──
VISIT_INTENT_STRONG = [
    "예약", "방문", "내원", "가보고싶", "가볼게", "가고싶",
    "진료받고싶", "치료받고싶", "상담받고싶", "진료예약",
    "어떻게가", "언제가면", "찾아가", "찾아뵙", "방문하고",
    "가보려고", "갈게요", "갈까요", "가볼까", "가봐야",
    "한번가", "직접가", "내원하고", "접수",
]
VISIT_INTENT_WEAK = [
    "어디", "위치", "주소", "오시는길", "길찾기", "지도",
    "진료시간", "운영시간", "몇시", "영업시간",
    "전화번호", "연락처", "전화",
]
VISIT_NEGATION = [
    "어렵", "못가", "안가", "안갈", "못갈", "힘들", "나중에",
    "다음에", "괜찮", "됐어", "아직", "생각해",
]

def _detect_visit_intent_keyword(query: str, history: list) -> str:
    """키워드 기반 1차 감지. 'strong' / 'weak' / None 반환."""
    q = query.replace(' ', '').lower()

    # 부정 표현 체크
    if any(neg in q for neg in VISIT_NEGATION):
        return None

    if any(kw in q for kw in VISIT_INTENT_STRONG):
        return "strong"
    if any(kw in q for kw in VISIT_INTENT_WEAK):
        return "weak"

    # 히스토리에서 최근 2턴 내 의료 상담 후 내원 표현 탐색
    recent = history[-4:] if history else []
    for turn in recent:
        if turn.get("role") == "user":
            t = turn.get("content", "").replace(' ', '').lower()
            if any(kw in t for kw in VISIT_INTENT_STRONG):
                return "strong"
    return None

def _detect_visit_intent_llm(query: str, history: list, gen: Generator) -> bool:
    """LLM 맥락 확인 (weak 키워드 감지 시에만 호출)."""
    hist_text = ""
    for turn in (history[-4:] if history else []):
        role = "환자" if turn.get("role") == "user" else "상담사"
        hist_text += f"{role}: {turn.get('content','')}\n"

    prompt = f"""다음 대화에서 환자(사용자)가 병원 방문이나 진료 예약 의사를 가지고 있는지 판단하세요.
단순 정보 확인(궁금함)이 아니라 실제 방문/예약 의향이 있는 경우에만 YES를 출력하세요.

[대화 기록]
{hist_text}환자: {query}

판단 (YES 또는 NO만 출력):"""

    try:
        resp = gen.client.models.generate_content(
            model=gen.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.0)
        )
        return "YES" in resp.text.strip().upper()
    except Exception:
        return False

# Gemini import for LLM intent detection
from google import genai

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not retriever or not generator:
        raise HTTPException(status_code=503, detail="RAG system not initialized properly.")
    
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    category = request.category
    history = SafetyGuard.validate_history(request.history)
    print(f"Received query: {query}, Category: {category}, History length: {len(history)}")
    
    async def response_generator():
        try:
            import asyncio
            import time
            
            start_time = time.time()
            
            # 1. Parallelize Classification and Retrieval
            # We run both tasks concurrently to save time
            
            # Task 1: Retrieval (Blocking I/O wrapped in thread)
            retrieval_task = asyncio.create_task(asyncio.to_thread(retriever.retrieve, query))
            
            # Task 2: Classification (Network call, usually fast but better parallel)
            # Since generator.classify_query is sync, we wrap it too
            classification_task = asyncio.create_task(asyncio.to_thread(generator.classify_query, query))
            
            # Wait for both
            context_docs, classified_category = await asyncio.gather(retrieval_task, classification_task)
            
            # Use the classified category if request was "auto"
            final_category = classified_category if category == "auto" else category
            print(f"Auto-routed category: {final_category} (Original: {category})")
            
            retrieval_end_time = time.time()
            print(f"[Timing] Retrieval & Classification took: {retrieval_end_time - start_time:.4f}s")
            
            # 2. Generate Stream
            stream = generator.generate_answer_stream(query, context_docs, final_category, history)

            is_fallback = False
            for chunk in stream:
                if chunk.startswith("[일반 의학 정보 안내]"):
                    is_fallback = True
                yield chunk
                await asyncio.sleep(0)

            # 3. Send Sources (조건부)
            # - general 카테고리(인사, 일상): 소스 없음
            # - 폴백 답변: 소스 없음
            # - RAG 의료 답변: 키워드 매칭 + 카테고리 기반 유튜브 추천
            sources = []
            if context_docs and final_category != "general" and not is_fallback:
                from database.supabase_client import SupabaseManager
                from config.settings import HOSPITAL_FAQS_TABLE

                def _parse_meta(raw):
                    if isinstance(raw, str):
                        try: return json.loads(raw)
                        except: return {}
                    return raw if isinstance(raw, dict) else {}

                def _is_youtube(url):
                    return 'youtube.com' in url or 'youtu.be' in url

                seen_urls = set()
                yt_sources = []
                other_sources = []

                # 동의어+복합어 확장 키워드
                query_keywords = SupabaseManager._extract_keywords(query)
                expanded_keywords = SupabaseManager._expand_synonyms(query_keywords)
                search_keywords = SupabaseManager._expand_compound_keywords(expanded_keywords)

                # 카테고리별 제외 키워드 (크로스 오염 방지)
                CATEGORY_EXCLUDE_KEYWORDS = {
                    "cancer": ["자율신경", "자율신경실조", "교감신경", "부교감신경", "자율신경장애"],
                    "nerve": ["고주파", "온열치료", "온코써미아", "하이퍼써미아", "항암", "화학요법", "항암치료", "항암제"],
                }
                exclude_keywords = CATEGORY_EXCLUDE_KEYWORDS.get(final_category, [])

                def _is_excluded_video(title, exclude_kws):
                    """제목에 다른 카테고리 키워드가 포함되어 있으면 제외"""
                    if not exclude_kws:
                        return False
                    title_norm = title.replace(' ', '').lower()
                    return any(kw.lower().replace(' ', '') in title_norm for kw in exclude_kws)

                # (A) context_docs에서 소스 수집
                for doc in context_docs:
                    metadata = _parse_meta(doc.get('metadata', {}))
                    if not metadata:
                        continue
                    source_url = metadata.get('source', '')
                    if source_url in seen_urls:
                        continue

                    if _is_youtube(source_url):
                        title = metadata.get('title', '')
                        if title and search_keywords:
                            # 다른 카테고리 영상 제외
                            if _is_excluded_video(title, exclude_keywords):
                                continue
                            title_norm = title.replace(' ', '').lower()
                            if any(kw.lower().replace(' ', '') in title_norm for kw in search_keywords):
                                seen_urls.add(source_url)
                                yt_sources.append(metadata)
                    elif source_url:
                        seen_urls.add(source_url)
                        other_sources.append(metadata)

                MAX_YT = 5

                # (B) 카테고리 기반 유튜브 추천 (우선)
                # 같은 카테고리의 유튜브 영상을 키워드 관련도 순으로 추천
                # 키워드 점수가 0인 영상(관련 없는 영상)은 제외
                if len(yt_sources) < MAX_YT:
                    try:
                        cat_resp = retriever.db_manager.client.table(HOSPITAL_FAQS_TABLE)\
                            .select("metadata")\
                            .contains("metadata", {"category": final_category})\
                            .limit(100)\
                            .execute()

                        cat_yt = []
                        cat_seen = set()
                        for row in cat_resp.data:
                            meta = _parse_meta(row.get('metadata', {}))
                            if not meta:
                                continue
                            src_url = meta.get('source', '')
                            title = meta.get('title', '')
                            if _is_youtube(src_url) and src_url not in seen_urls \
                                    and src_url not in cat_seen and title:
                                # 다른 카테고리 영상 제외
                                if _is_excluded_video(title, exclude_keywords):
                                    continue
                                cat_seen.add(src_url)
                                cat_yt.append(meta)

                        # 키워드 관련도 순으로 정렬 (매칭 키워드 수 기준)
                        # 점수가 0인 영상은 제외하여 크로스 오염 방지
                        if search_keywords:
                            for item in cat_yt:
                                title_norm = item.get('title', '').replace(' ', '').lower()
                                item['_score'] = sum(
                                    1 for kw in search_keywords
                                    if kw.lower().replace(' ', '') in title_norm
                                )
                            # 점수 > 0인 영상만 포함 (키워드 매칭 필수)
                            cat_yt = [item for item in cat_yt if item.get('_score', 0) > 0]
                            cat_yt.sort(key=lambda x: x.get('_score', 0), reverse=True)

                        need = MAX_YT - len(yt_sources)
                        for meta in cat_yt[:need]:
                            meta.pop('_score', None)
                            seen_urls.add(meta.get('source', ''))
                            yt_sources.append(meta)
                    except Exception as cat_err:
                        print(f"Category YouTube search error: {cat_err}")

                # (C) 키워드 검색으로 유튜브 보충 (카테고리 결과가 부족할 때)
                # 다른 카테고리 영상이 혼입되지 않도록 제외 키워드 적용
                if len(yt_sources) < MAX_YT and search_keywords:
                    try:
                        for table in [None, HOSPITAL_FAQS_TABLE]:
                            if len(yt_sources) >= MAX_YT:
                                break
                            yt_results = retriever.db_manager.keyword_search(
                                query, k=15,
                                metadata_filter={"type": "youtube"} if table is None else None,
                                table_name=table
                            )
                            for doc in yt_results:
                                if len(yt_sources) >= MAX_YT:
                                    break
                                meta = _parse_meta(doc.get('metadata', {}))
                                if not meta:
                                    continue
                                src_url = meta.get('source', '')
                                if not _is_youtube(src_url) or src_url in seen_urls:
                                    continue
                                title = meta.get('title', '')
                                if title:
                                    # 다른 카테고리 영상 제외
                                    if _is_excluded_video(title, exclude_keywords):
                                        continue
                                    title_norm = title.replace(' ', '').lower()
                                    if any(kw.lower().replace(' ', '') in title_norm for kw in search_keywords):
                                        seen_urls.add(src_url)
                                        yt_sources.append(meta)
                    except Exception as yt_err:
                        print(f"YouTube keyword search error: {yt_err}")

                sources = other_sources + yt_sources

            if sources:
                yield f"\n\n__SOURCES__\n{json.dumps(sources)}"

            # 4. 내원 의사 감지 → 예약 안내 카드 전송
            try:
                intent_level = _detect_visit_intent_keyword(query, history)
                show_booking = False

                if intent_level == "strong":
                    show_booking = True
                    print(f"[Booking] Strong intent detected: {query[:50]}")
                elif intent_level == "weak":
                    # weak 키워드 → LLM으로 맥락 확인
                    show_booking = await asyncio.to_thread(
                        _detect_visit_intent_llm, query, history, generator
                    )
                    print(f"[Booking] Weak intent, LLM confirmed: {show_booking}")

                if show_booking:
                    yield "\n\n__BOOKING__"
            except Exception as booking_err:
                print(f"[Booking] Detection error: {booking_err}")

        except Exception as e:
            print(f"Error processing request: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
