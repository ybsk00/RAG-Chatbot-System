from typing import List, Dict
from config.settings import MEDICAL_DISCLAIMER, NO_INFO_MESSAGE

class SafetyGuard:
    @staticmethod
    def check_relevance(retrieved_docs: List[Dict]) -> bool:
        """
        Checks if retrieved documents are relevant enough.
        Since we already filtered by threshold in Retriever, if list is empty, it's irrelevant.
        """
        return len(retrieved_docs) > 0

    @staticmethod
    def get_no_info_response() -> str:
        return NO_INFO_MESSAGE

    @staticmethod
    def append_disclaimer(response_text: str) -> str:
        return f"{response_text}\n\n---\n**{MEDICAL_DISCLAIMER}**"

    @staticmethod
    def check_medical_query(query: str) -> bool:
        """
        Simple keyword check for medical diagnosis requests.
        In a real system, this would be an LLM call or more sophisticated classifier.
        """
        forbidden_keywords = ["진단해줘", "처방해줘", "약 추천", "무슨 병이야"]
        return any(keyword in query for keyword in forbidden_keywords)

    @staticmethod
    def get_diagnosis_warning() -> str:
        return "죄송합니다. 저는 의학적 진단이나 처방을 내려드릴 수 없습니다. 정확한 진단은 병원에 내원하여 전문의와 상담해주세요."
