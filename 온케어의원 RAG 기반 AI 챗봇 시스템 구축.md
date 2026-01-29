## [작업 지시서] 병원 특화 RAG 기반 AI 챗봇 시스템 구축
### 1. 프로젝트 개요
목적: 병원 공식 유튜브 및 블로그 데이터를 기반으로 한 FAQ 중심의 환자 응대 챗봇 구축

핵심 가치: 신뢰할 수 있는 데이터(병원 자체 콘텐츠) 기반 답변, 환각(Hallucination) 방지, 의학적 진단/처방 금지 가드레일 적용

데이터 규모: 유튜브 영상 약 200개 및 공식 블로그 포스팅 전수

### 2. 기술 스택 (Tech Stack)
LLM API: Google Gemini 2.5 Flash

Embedding: Google Gemini text-embedding-004 (Output Dimensionality: 768)

Vector Database: Supabase (PostgreSQL + pgvector)

Data Pipeline: Python (yt-dlp, BeautifulSoap, Pandas)

Orchestration: LangChain 또는 LlamaIndex

### 3. 상세 개발 요구사항
#### 1단계: 데이터 수집 및 전처리 (Ingestion)
유튜브 파이프라인: * 지정된 채널 URL에서 모든 영상 ID 추출

youtube-transcript-api를 통한 자막 추출. 자막 부재 시 yt-dlp와 Whisper 또는 Gemini를 연동한 STT(Speech-to-Text) 수행

블로그 파이프라인: * 공식 블로그 본문 텍스트 크롤링 및 정제

데이터 정제: * 추출된 날것의 텍스트를 노트북LM 스타일의 FAQ 형태(Question-Answer-Source)로 변환 (LLM 활용 권장)

Chunking 전략: 의미 단위 분할(Semantic Chunking) 적용. 각 청크에는 반드시 원본 URL과 제목을 메타데이터로 포함

#### 2단계: 벡터 데이터베이스 구축 (Storage)
슈파베이스 설정:

pgvector 확장 활성화 및 embedding vector(768) 컬럼이 포함된 테이블 정의

검색 성능 최적화를 위한 HNSW(Hierarchical Navigable Small World) 인덱스 생성

임베딩 및 적재:

제미나이 768차원 모델을 사용하여 정제된 FAQ 데이터를 벡터화하여 업로드

#### 3단계: RAG 기반 답변 로직 (Orchestration)
검색(Retrieval): * 사용자 질문과 벡터 DB 간의 코사인 유사도(Cosine Similarity) 검색

필요 시 키워드 기반 Full-text Search를 결합한 Hybrid Search 구현

답변 생성(Generation): * 검색된 상위 K개의 컨텍스트를 제미나이 프롬프트에 주입

출처 표기: 답변 하단에 참고한 유튜브/블로그 링크 자동 생성

#### 4단계: 가드레일 및 안전장치 (Safety)
진단 금지 로직: "진단", "처방", "치료법 결정" 등에 대한 질문 시 전문의 상담을 유도하는 표준 문구 출력

환각 방지: 검색된 컨텍스트 내에 답이 없을 경우 "제공된 정보 내에서는 확인이 어렵습니다"라고 답변하도록 시스템 프롬프트 설계

### 4. 주요 마일스톤 (Deliverables)
데이터 수집 스크립트: 유튜브/블로그 자동 추출 및 FAQ 변환 코드

DB 스키마 및 적재 코드: 슈파베이스 연동 및 768차원 벡터 적재 스크립트

챗봇 엔진 API: 사용자 질문에 대해 RAG 과정을 거쳐 최종 답변을 반환하는 API 엔드포인트

관리자 도구 (선택 사항): 새로운 유튜브 링크 입력 시 실시간으로 DB를 업데이트하는 기능

### 5. 개발자 주의 사항
데이터 무결성: 의학 정보이므로 텍스트 추출 과정에서 발생하는 오타(특히 약물 이름, 균주 이름 등)가 임베딩 품질을 해치지 않도록 검수 로직을 포함할 것.

차원수 준수: 반드시 제미나이 768차원 설정을 준수하여 슈파베이스 인덱스와 일치시킬 것.