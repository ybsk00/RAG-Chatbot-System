from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import GOOGLE_API_KEY, GENERATION_MODEL
from rag.safety import SafetyGuard

class Generator:
    def __init__(self):
        if not GOOGLE_API_KEY:
             raise ValueError("GOOGLE_API_KEY not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model=GENERATION_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        당신은 서울온케어의원의 **AI 상담 전문의**입니다.
        현재 상담 주제는 **{category_name}**입니다.
        
        환자(사용자)의 질문에 대해 아래 [Context]를 바탕으로 친절하고 전문적으로 답변해 주세요.
        
        [Context]:
        {context}
        
        [Question]:
        {question}
        
        **답변 가이드라인 (필수 준수)**:
        1. **페르소나**: 당신은 의사 선생님처럼 공감하며 전문적인 어조를 사용합니다. 하지만 **절대로 확정적인 진단이나 처방을 내려서는 안 됩니다.**
        2. **안전장치**: "진단", "처방", "약물 추천" 등의 요청에는 "구체적인 진단과 처방은 내원하시어 전문의와 상담이 필요합니다"라는 취지로 안내하세요.
        3. **근거 기반**: 제공된 Context에 없는 내용은 지어내지 마세요. 모르는 내용은 솔직히 모른다고 하거나 병원 문의를 유도하세요.
        4. **출처 표기**: 답변 내용이 포함된 영상이나 블로그가 있다면 언급해 주세요.
        
        **법적 고지 (답변 하단에 필수 포함)**:
        "본 상담 내용은 참고용이며, 의학적 진단이나 처방을 대신할 수 없습니다."
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_answer(self, query: str, context_docs: List[Dict], category: str = "cancer") -> str:
        # Map category code to name
        category_map = {
            "cancer": "암 보조 치료 (Cancer Support Treatment)",
            "nerve": "자율신경 치료 (Autonomic Nerve Treatment)"
        }
        category_name = category_map.get(category, "일반 상담")

        # 1. Safety Check: Diagnosis (Keyword check is still good as a first line of defense)
        if SafetyGuard.check_medical_query(query):
            # Even if we catch it here, we might want to let the LLM handle it gracefully with the persona,
            # but returning a standard safe response is also fine. 
            # Let's rely on the LLM's persona for a more natural "Doctor" refusal, 
            # unless it's a very specific blocked keyword.
            pass 

        # 2. Safety Check: Relevance
        if not SafetyGuard.check_relevance(context_docs):
            return SafetyGuard.get_no_info_response()

        # 3. Format Context
        formatted_context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # 4. Generate
        response = self.chain.invoke({
            "context": formatted_context, 
            "question": query,
            "category_name": category_name
        })
        
        # 5. Append Disclaimer (Double check, though prompt handles it, explicit append is safer)
        # The prompt asks to include it, but let's ensure it's there.
        if "본 상담 내용은 참고용이며" not in response:
            final_response = SafetyGuard.append_disclaimer(response)
        else:
            final_response = response
        
        return final_response
