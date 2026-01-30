# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# YouTube Configuration
YOUTUBE_CHANNEL_URL = "https://www.youtube.com/@%EC%84%9C%EC%9A%B8%EC%98%A8%EC%BC%80%EC%96%B4%EC%9D%98%EC%9B%90"
SAMPLE_VIDEO_URL = "https://www.youtube.com/watch?v=jKqWK6Qhe1Q"
BLOG_URL = "https://blog.naver.com/rorees"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash" # Updated as per user request

# RAG Configuration
SIMILARITY_THRESHOLD = 0.40
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KEYWORD_SIMILARITY_FLOOR = 0.3
MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_DOCS = 5

# Table Names
DOCUMENTS_TABLE = "documents"
HOSPITAL_FAQS_TABLE = "hospital_faqs"

# Temperature Configuration
GENERAL_TEMPERATURE = 0.3
MEDICAL_TEMPERATURE = 0.2
ROUTER_TEMPERATURE = 0.0

# Cache Configuration
EMBEDDING_CACHE_SIZE = 256
RESULT_CACHE_SIZE = 128
RESULT_CACHE_TTL_SECONDS = 300

# Safety
MEDICAL_DISCLAIMER = "본 답변은 병원 콘텐츠를 기반으로 생성된 참고용 정보이며, 실제 진료를 대신할 수 없습니다."
NO_INFO_MESSAGE = "죄송합니다. 해당 내용에 대한 병원 공식 자료를 찾을 수 없습니다. 정확한 상담은 병원으로 전화 부탁드립니다."

# Fallback Configuration (RAG 결과 없을 때 일반 지식 폴백)
ENABLE_MEDICAL_FALLBACK = True
FALLBACK_TEMPERATURE = 0.1
FALLBACK_MAX_CHARS = 300
FALLBACK_PREFIX = "[일반 의학 정보 안내]\n\n"
FALLBACK_DISCLAIMER = (
    "위 내용은 서울온케어의원의 자료가 아닌 일반적인 의학 상식에 기반한 참고 정보입니다. "
    "정확한 진단과 치료를 위해 서울온케어의원에 내원하시거나 전화로 상담받으시기 바랍니다."
)
