from typing import List, Dict
from google import genai
from config.settings import GOOGLE_API_KEY, GENERATION_MODEL, MEDICAL_DISCLAIMER
from rag.safety import SafetyGuard

class Generator:
    def __init__(self):
        if not GOOGLE_API_KEY:
             raise ValueError("GOOGLE_API_KEY not set")
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
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

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "없음"
        formatted = []
        for turn in history[-5:]: # Keep last 5 turns
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def classify_query(self, query: str) -> str:
        """Classifies the query into 'cancer', 'nerve', or 'general'."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=self.router_prompt.format(question=query),
                config=genai.types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            category = response.text.strip().lower()
            if category not in ["cancer", "nerve", "general"]:
                return "general" # Default fallback
            return category
        except Exception as e:
            print(f"Router Error: {e}")
            return "general"

    def generate_answer(self, query: str, context_docs: List[Dict], category: str = "auto", history: List[Dict] = []) -> str:
        # 1. Auto-Routing
        if category == "auto":
            category = self.classify_query(query)
            print(f"Auto-routed category: {category}")

        history_text = self._format_history(history)

        # 2. Handle General Queries (Consultation Manager)
        if category == "general":
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.general_prompt_template.format(question=query, history=history_text),
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7
                    )
                )
                return response.text
            except Exception as e:
                return f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"

        # 3. Handle Medical Queries (RAG Agent)
        # Map category code to name
        category_map = {
            "cancer": "암 보조 치료 (Cancer Support Treatment)",
            "nerve": "자율신경 치료 (Autonomic Nerve Treatment)"
        }
        category_name = category_map.get(category, "일반 상담")

        # Safety Check: Diagnosis
        if SafetyGuard.check_medical_query(query):
            pass 

        # Safety Check: Relevance
        if not SafetyGuard.check_relevance(context_docs):
            # If RAG fails but it was routed as medical, maybe fallback to general or say I don't know
            return SafetyGuard.get_no_info_response()

        # Format Context
        formatted_context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Construct Prompt
        prompt = self.medical_prompt_template.format(
            category_name=category_name,
            context=formatted_context,
            question=query,
            history=history_text
        )
        
        # Generate
        try:
            response_obj = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3
                )
            )
            response = response_obj.text
        except Exception as e:
            return f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"
        
        # Append Disclaimer
        if "본 상담 내용은 참고용이며" not in response:
            final_response = SafetyGuard.append_disclaimer(response)
        else:
            final_response = response
        
        return final_response

    def generate_answer_stream(self, query: str, context_docs: List[Dict], category: str = "auto", history: List[Dict] = [], **kwargs):
        print(f"DEBUG: generate_answer_stream called. History len: {len(history)}")
        # 1. Auto-Routing
        if category == "auto":
            category = self.classify_query(query)
            print(f"Auto-routed category: {category}")

        history_text = self._format_history(history)

        # 2. Handle General Queries (Consultation Manager)
        if category == "general":
            try:
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=self.general_prompt_template.format(question=query, history=history_text),
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7
                    )
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                yield f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"
            return

        # 3. Handle Medical Queries (RAG Agent)
        # Map category code to name
        category_map = {
            "cancer": "암 보조 치료 (Cancer Support Treatment)",
            "nerve": "자율신경 치료 (Autonomic Nerve Treatment)"
        }
        category_name = category_map.get(category, "일반 상담")

        # Safety Check: Diagnosis
        if SafetyGuard.check_medical_query(query):
            pass 

        # Safety Check: Relevance
        if not SafetyGuard.check_relevance(context_docs):
            yield SafetyGuard.get_no_info_response()
            return

        # Format Context
        formatted_context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Construct Prompt
        prompt = self.medical_prompt_template.format(
            category_name=category_name,
            context=formatted_context,
            question=query,
            history=history_text
        )
        
        # Generate
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3
                )
            )
            
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    text_chunk = chunk.text
                    full_response += text_chunk
                    yield text_chunk
            
            # Append Disclaimer if not present
            if "본 상담 내용은 참고용이며" not in full_response:
                yield f"\n\n---\n**{MEDICAL_DISCLAIMER}**"

        except Exception as e:
            yield f"죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. (Error: {str(e)})"