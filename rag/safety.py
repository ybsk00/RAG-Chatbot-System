from typing import List, Dict
from config.settings import MEDICAL_DISCLAIMER, NO_INFO_MESSAGE, SIMILARITY_THRESHOLD


class SafetyGuard:
    FORBIDDEN_KEYWORDS = [
        "진단해줘", "처방해줘", "약 추천", "무슨 병이야",
        "진단해 줘", "처방해 줘", "약 좀 추천", "병명 알려",
        "무슨 병인지", "진단 내려", "약 처방"
    ]

    @staticmethod
    def check_relevance(retrieved_docs: List[Dict], min_similarity: float = None) -> bool:
        """
        검색된 문서의 관련성을 확인합니다.
        Retriever가 이미 임계값 필터링과 재순위화를 수행했으므로,
        여기서는 문서 존재 여부와 최소 유사도만 확인합니다.
        """
        if not retrieved_docs:
            return False
        if min_similarity is None:
            min_similarity = SIMILARITY_THRESHOLD * 0.7  # Retriever 임계값보다 낮게 설정
        return any(doc.get('similarity', 0) >= min_similarity for doc in retrieved_docs)

    @staticmethod
    def get_no_info_response() -> str:
        return NO_INFO_MESSAGE

    @staticmethod
    def append_disclaimer(response_text: str) -> str:
        return f"{response_text}\n\n---\n**{MEDICAL_DISCLAIMER}**"

    # LLM 출력물에서 차단해야 할 처방/진단 표현
    OUTPUT_FORBIDDEN = [
        "처방합니다", "처방드립니다", "진단합니다", "진단드립니다",
        "복용하세요", "투여", "처방전", "mg", "정을 드세요",
        "주사하세요", "수술하세요"
    ]

    @staticmethod
    def check_output_safety(response: str) -> bool:
        """LLM 출력에 처방/진단 표현이 없으면 True(안전)."""
        normalized = response.replace(" ", "")
        return not any(
            kw.replace(" ", "") in normalized
            for kw in SafetyGuard.OUTPUT_FORBIDDEN
        )

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
