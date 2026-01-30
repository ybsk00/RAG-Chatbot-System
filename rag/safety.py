from typing import List, Dict
from config.settings import MEDICAL_DISCLAIMER, NO_INFO_MESSAGE


class SafetyGuard:
    FORBIDDEN_KEYWORDS = [
        "진단해줘", "처방해줘", "약 추천", "무슨 병이야",
        "진단해 줘", "처방해 줘", "약 좀 추천", "병명 알려",
        "무슨 병인지", "진단 내려", "약 처방"
    ]

    @staticmethod
    def check_relevance(retrieved_docs: List[Dict], min_similarity: float = 0.55) -> bool:
        """유사도 점수 기반으로 실제 관련성을 확인합니다."""
        if not retrieved_docs:
            return False
        return any(doc.get('similarity', 0) >= min_similarity for doc in retrieved_docs)

    @staticmethod
    def get_no_info_response() -> str:
        return NO_INFO_MESSAGE

    @staticmethod
    def append_disclaimer(response_text: str) -> str:
        return f"{response_text}\n\n---\n**{MEDICAL_DISCLAIMER}**"

    @staticmethod
    def check_medical_query(query: str) -> bool:
        """띄어쓰기 변형을 포함하여 진단/처방 요청을 감지합니다."""
        normalized = query.replace(" ", "")
        return any(kw.replace(" ", "") in normalized for kw in SafetyGuard.FORBIDDEN_KEYWORDS)

    @staticmethod
    def get_diagnosis_warning() -> str:
        return "죄송합니다. 저는 의학적 진단이나 처방을 내려드릴 수 없습니다. 정확한 진단은 병원에 내원하여 전문의와 상담해주세요."

    @staticmethod
    def validate_history(history) -> List[Dict]:
        """대화 이력을 검증하고 정리합니다."""
        if not isinstance(history, list):
            return []
        validated = []
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"] if item["role"] in ("user", "model") else "user"
                content = str(item.get("content", ""))[:2000]
                validated.append({"role": role, "content": content})
        return validated[-10:]
