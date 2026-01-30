import json
import logging
from typing import List, Dict
from google import genai
from config.settings import (
    GOOGLE_API_KEY, GENERATION_MODEL, MEDICAL_DISCLAIMER,
    GENERAL_TEMPERATURE, MEDICAL_TEMPERATURE, ROUTER_TEMPERATURE,
    ENABLE_MEDICAL_FALLBACK, FALLBACK_TEMPERATURE, FALLBACK_MAX_CHARS,
    FALLBACK_PREFIX, FALLBACK_DISCLAIMER
)
from rag.safety import SafetyGuard

logger = logging.getLogger("rag.generator")

# Gemini Client 싱글톤
_genai_client = None

def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")
        _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _genai_client


class Generator:
    def __init__(self):
        self.client = _get_genai_client()
        self.model = GENERATION_MODEL

        # 1. Router Prompt
        self.router_prompt = """
        당신은 사용자의 질문을 분석하여 가장 적절한 카테고리로 분류하는 AI입니다.

        [카테고리 정의]
        1. **cancer**: 암, 항암 치료, 면역 치료, 암 식단, 고주파 온열 치료 등 암과 관련된 모든 의학적 질문.
        2. **nerve**: 자율신경, 신경 주사, 어지러움, 두통, 실신, 기립성 빈맥 등 자율신경계와 관련된 모든 의학적 질문.
        3. **general**: 인사, 병원 위치 문의, 진료 시간, 비용 문의, 날씨, 일상적인 대화 등 의학적 전문 지식이 필요 없는 질문.

        사용자의 질문이 입력되면, 위 3가지 카테고리 중 하나를 선택하여 단어 하나만 출력하세요. (예: cancer)

        [예시]
        Q: 암 환자가 먹으면 좋은 음식은?
        A: cancer

        Q: 자율신경 실조증 치료 방법 알려줘
        A: nerve

        Q: 진료 시간이 언제인가요?
        A: general

        Q: 고주파 온열 치료 효과가 뭐야?
        A: cancer

        Q: 기립성 빈맥 증후군이 뭐야?
        A: nerve

        [질문]: {question}
        [분류]:
        """

        # 2. Consultation Manager Persona (General)
        self.general_prompt_template = """
        당신은 서울온케어의원의 **상담 실장 온케어봇**입니다.
        환자분들을 따뜻하고 친절하게 맞이하고, 병원 이용에 대한 기본적인 안내를 도와드립니다.

        [Previous Conversation]:
        {history}

        [질문]:
        {question}

        **가이드라인**:
        1. **친절함**: 항상 밝고 정중한 태도로 응대하세요.
        2. **역할 제한**: 의학적인 상담이나 진단은 하지 않습니다. 의학적인 질문이 들어오면 "죄송하지만, 그 부분은 원장님 진료 시 자세히 상담받으실 수 있습니다."라고 안내하세요.
        3. **병원 안내**: 진료 시간, 위치 등은 알고 있는 범위 내에서 안내하되, 모르는 내용은 "병원으로 전화 주시면 친절히 안내해 드리겠습니다."라고 답변하세요.
        4. **간결함**: 답변은 공백 포함 300자 이내로 핵심만 전달하세요.
        """

        # 3. Medical Doctor Persona (RAG)
        self.medical_prompt_template = """
        당신은 서울온케어의원의 **AI 상담 전문의 온케어봇**입니다.
        현재 상담 주제는 **{category_name}**입니다.

        환자(사용자)의 질문에 대해 아래 [Context]를 바탕으로 친절하고 전문적으로 답변해 주세요.

        [Previous Conversation]:
        {history}

        [Context] (참고 자료):
        {context}

        [Question]:
        {question}

        **답변 가이드라인 (필수 준수)**:
        1. **문맥 필터링 (중요)**: 위 [Context]에는 질문과 관련 없는 내용이 섞여 있을 수 있습니다. **반드시 질문과 직접적으로 관련된 내용만 골라서** 답변하세요. 단순히 단어가 같다고 해서 관련 없는 내용을 억지로 연결하지 마세요.
        2. **페르소나**: 당신은 의사 선생님처럼 공감하며 전문적인 어조를 사용합니다. 하지만 **절대로 확정적인 진단이나 처방을 내려서는 안 됩니다.**
        3. **안전장치**: "진단", "처방", "약물 추천" 등의 요청에는 "구체적인 진단과 처방은 내원하시어 전문의와 상담이 필요합니다"라는 취지로 안내하세요.
        4. **근거 기반**: 제공된 Context에 질문에 대한 답이 명확히 없다면, 솔직하게 "해당 내용은 병원 자료에 없어 정확한 답변이 어렵습니다"라고 말하세요. 지어내지 마세요.
        5. **길이 제한**: 답변은 공백 포함 500자 이내로 작성하되, 핵심 정보는 빠짐없이 전달하세요.

        **법적 고지 (답변 하단에 필수 포함)**:
        "본 상담 내용은 참고용이며, 의학적 진단이나 처방을 대신할 수 없습니다."
        """

        # 4. Medical Fallback Persona (RAG 결과 없을 때 일반 지식 기반)
        self.fallback_prompt_template = """
        당신은 서울온케어의원의 AI 상담 보조입니다.
        환자의 질문에 대해 **일반적인 의학 상식** 수준에서만 간단히 안내합니다.

        [Previous Conversation]:
        {history}

        [Question]:
        {question}

        **엄격한 답변 규칙 (반드시 준수)**:
        1. **병원 자료 없음**: 이 질문에 대한 서울온케어의원의 공식 자료는 현재 준비되어 있지 않습니다. 이를 반드시 답변 첫머리에 밝히세요.
        2. **일반 상식만**: 널리 알려진 의학 상식 수준으로만 답변하세요. 구체적인 치료법, 약물명, 용량은 절대 언급하지 마세요.
        3. **짧고 보수적**: 답변은 공백 포함 {max_chars}자 이내로, 핵심만 간결하게 작성하세요.
        4. **확정 금지**: "~입니다", "~해야 합니다" 같은 확정 표현 대신 "~일 수 있습니다", "~가 일반적입니다" 같은 유보적 표현을 사용하세요.
        5. **진단/처방 절대 금지**: 특정 질병을 진단하거나 약물을 처방하는 내용은 절대 포함하지 마세요.
        6. **내원 유도**: 답변 마지막에 반드시 "자세한 내용은 서울온케어의원에 내원하시어 전문의 상담을 받으시기 바랍니다."를 포함하세요.
        """

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "없음"
        formatted = []
        for turn in history[-5:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _format_context(self, context_docs: List[Dict]) -> str:
        """출처·관련도 레이블을 포함한 구조화된 컨텍스트를 생성합니다."""
        parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', '')
            similarity = doc.get('similarity', 0)
            metadata = doc.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            source_type = metadata.get('type', 'unknown')
            title = metadata.get('title', '자료')

            parts.append(
                f"[자료 {i}] (출처: {source_type}, 관련도: {similarity:.0%})\n"
                f"제목: {title}\n{content}"
            )
        return "\n\n---\n\n".join(parts)

    def classify_query(self, query: str) -> str:
        """질문을 cancer / nerve / general로 분류합니다."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.router_prompt.format(question=query),
                config=genai.types.GenerateContentConfig(
                    temperature=ROUTER_TEMPERATURE
                )
            )
            category = response.text.strip().lower()
            if category not in ["cancer", "nerve", "general"]:
                return "general"
            return category
        except Exception as e:
            print(f"Router Error: {e}")
            return "general"

    def generate_answer(self, query: str, context_docs: List[Dict], category: str = "auto", history: List[Dict] = []) -> str:
        history = SafetyGuard.validate_history(history)

        if category == "auto":
            category = self.classify_query(query)
            print(f"Auto-routed category: {category}")

        history_text = self._format_history(history)

        # General 질문 처리
        if category == "general":
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.general_prompt_template.format(question=query, history=history_text),
                    config=genai.types.GenerateContentConfig(
                        temperature=GENERAL_TEMPERATURE
                    )
                )
                return response.text
            except Exception as e:
                return f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"

        # Medical 질문 처리
        category_map = {
            "cancer": "암 보조 치료 (Cancer Support Treatment)",
            "nerve": "자율신경 치료 (Autonomic Nerve Treatment)"
        }
        category_name = category_map.get(category, "일반 상담")

        # 안전 체크: 진단/처방 요청 감지
        if SafetyGuard.check_medical_query(query):
            return SafetyGuard.get_diagnosis_warning()

        # 안전 체크: 관련 문서 존재 여부
        if not SafetyGuard.check_relevance(context_docs):
            if ENABLE_MEDICAL_FALLBACK:
                return self._generate_fallback(query, history_text)
            return SafetyGuard.get_no_info_response()

        formatted_context = self._format_context(context_docs)

        prompt = self.medical_prompt_template.format(
            category_name=category_name,
            context=formatted_context,
            question=query,
            history=history_text
        )

        try:
            response_obj = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=MEDICAL_TEMPERATURE
                )
            )
            response = response_obj.text
        except Exception as e:
            return f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"

        if "본 상담 내용은 참고용이며" not in response:
            return SafetyGuard.append_disclaimer(response)
        return response

    def generate_answer_stream(self, query: str, context_docs: List[Dict], category: str = "auto", history: List[Dict] = [], **kwargs):
        history = SafetyGuard.validate_history(history)
        print(f"DEBUG: generate_answer_stream called. History len: {len(history)}")

        if category == "auto":
            category = self.classify_query(query)
            print(f"Auto-routed category: {category}")

        history_text = self._format_history(history)

        # General 질문 스트리밍
        if category == "general":
            try:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=self.general_prompt_template.format(question=query, history=history_text),
                    config=genai.types.GenerateContentConfig(
                        temperature=GENERAL_TEMPERATURE
                    )
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"
            return

        # Medical 질문 스트리밍
        category_map = {
            "cancer": "암 보조 치료 (Cancer Support Treatment)",
            "nerve": "자율신경 치료 (Autonomic Nerve Treatment)"
        }
        category_name = category_map.get(category, "일반 상담")

        # 안전 체크: 진단/처방 요청 감지
        if SafetyGuard.check_medical_query(query):
            yield SafetyGuard.get_diagnosis_warning()
            return

        # 안전 체크: 관련 문서 존재 여부 → 폴백 분기
        if not SafetyGuard.check_relevance(context_docs):
            if ENABLE_MEDICAL_FALLBACK:
                yield from self._generate_fallback_stream(query, history_text)
                return
            yield SafetyGuard.get_no_info_response()
            return

        formatted_context = self._format_context(context_docs)

        prompt = self.medical_prompt_template.format(
            category_name=category_name,
            context=formatted_context,
            question=query,
            history=history_text
        )

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=MEDICAL_TEMPERATURE
                )
            )

            full_response_parts = []
            for chunk in response_stream:
                if chunk.text:
                    full_response_parts.append(chunk.text)
                    yield chunk.text

            full_response = "".join(full_response_parts)
            if "본 상담 내용은 참고용이며" not in full_response:
                yield f"\n\n---\n**{MEDICAL_DISCLAIMER}**"

        except Exception as e:
            yield f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"

    # ──────────────────────────────────────────────
    # 폴백: RAG 결과 없을 때 일반 의학 지식 기반 답변
    # ──────────────────────────────────────────────

    def _generate_fallback(self, query: str, history_text: str) -> str:
        """RAG 결과 없을 때 일반 의학 지식 기반 보수적 답변 (비스트리밍)."""
        logger.info(f"FALLBACK_TRIGGERED | query={query[:80]}")

        prompt = self.fallback_prompt_template.format(
            question=query,
            history=history_text,
            max_chars=FALLBACK_MAX_CHARS
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=FALLBACK_TEMPERATURE
                )
            )
            answer = response.text

            if not SafetyGuard.check_output_safety(answer):
                logger.warning(f"FALLBACK_BLOCKED_OUTPUT | query={query[:80]}")
                return SafetyGuard.get_no_info_response()

            return f"{FALLBACK_PREFIX}{answer}\n\n---\n**{FALLBACK_DISCLAIMER}**"
        except Exception as e:
            logger.error(f"FALLBACK_ERROR | query={query[:80]} | error={e}")
            return SafetyGuard.get_no_info_response()

    def _generate_fallback_stream(self, query: str, history_text: str):
        """RAG 결과 없을 때 일반 의학 지식 기반 스트리밍 답변."""
        logger.info(f"FALLBACK_STREAM_TRIGGERED | query={query[:80]}")

        prompt = self.fallback_prompt_template.format(
            question=query,
            history=history_text,
            max_chars=FALLBACK_MAX_CHARS
        )

        try:
            yield FALLBACK_PREFIX

            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=FALLBACK_TEMPERATURE
                )
            )

            full_parts = []
            for chunk in response_stream:
                if chunk.text:
                    full_parts.append(chunk.text)
                    yield chunk.text

            full_response = "".join(full_parts)
            if not SafetyGuard.check_output_safety(full_response):
                logger.warning(f"FALLBACK_BLOCKED_OUTPUT | query={query[:80]}")
                yield "\n\n(이 내용은 안전 검토를 통과하지 못했습니다. 병원에 직접 문의해 주세요.)"
                return

            yield f"\n\n---\n**{FALLBACK_DISCLAIMER}**"

        except Exception as e:
            logger.error(f"FALLBACK_STREAM_ERROR | query={query[:80]} | error={e}")
            yield SafetyGuard.get_no_info_response()
