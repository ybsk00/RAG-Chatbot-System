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
GENERATION_MODEL = "gemini-2.0-flash-exp" # Or gemini-1.5-flash as per availability

# RAG Configuration
SIMILARITY_THRESHOLD = 0.6
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Safety
MEDICAL_DISCLAIMER = "본 답변은 병원 콘텐츠를 기반으로 생성된 참고용 정보이며, 실제 진료를 대신할 수 없습니다."
NO_INFO_MESSAGE = "죄송합니다. 해당 내용에 대한 병원 공식 자료를 찾을 수 없습니다. 정확한 상담은 병원으로 전화 부탁드립니다."
