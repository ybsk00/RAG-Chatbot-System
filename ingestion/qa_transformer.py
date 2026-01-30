"""
Q/A Transformer using Gemini 2.0 Flash
Transforms blog posts into FAQ format
"""
import os
import json
import time
from typing import List, Dict
from google import genai
from config.settings import GOOGLE_API_KEY


class QATransformer:
    """블로그 글을 FAQ 형식으로 변환"""
    
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = "gemini-2.0-flash"
    
    def transform_single(self, post: Dict) -> List[Dict]:
        """단일 블로그 글을 Q/A 목록으로 변환"""
        title = post.get('title', '')
        content = post.get('content', '')
        
        if not content or len(content) < 30:
            print(f"[QA] Skipping: content too short ({len(content)} chars)")
            return []
        
        # Simple prompt without complex JSON examples
        prompt = f"""다음 글을 읽고 Q/A 1-2개를 만들어주세요.

제목: {title}

내용:
{content[:3000]}

아래 형식으로만 답변해주세요:
질문1: (질문 내용)
답변1: (답변 내용)
카테고리1: general 또는 cancer 또는 nerve
질문2: (질문 내용 - 없으면 '없음')
답변2: (답변 내용 - 없으면 '없음')
카테고리2: general 또는 cancer 또는 nerve"""
        
        try:
            print(f"[QA] Transforming: {title[:40]}...")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                )
            )
            
            text = response.text.strip()
            return self._parse_response(text, post)
            
        except Exception as e:
            print(f"[QA] Error: {e}")
            return []
    
    def _parse_response(self, text: str, post: Dict) -> List[Dict]:
        """응답 텍스트를 파싱"""
        results = []
        
        # Parse Q1/A1
        q1_match = re.search(r'질문1:\s*(.+?)(?=\n답변1:|$)', text, re.DOTALL)
        a1_match = re.search(r'답변1:\s*(.+?)(?=\n카테고리1:|$)', text, re.DOTALL)
        c1_match = re.search(r'카테고리1:\s*(\w+)', text)
        
        if q1_match and a1_match:
            q1 = q1_match.group(1).strip()
            a1 = a1_match.group(1).strip()
            c1 = c1_match.group(1).strip() if c1_match else 'general'
            
            if q1 and a1 and q1 != '없음' and a1 != '없음':
                results.append({
                    'question': q1,
                    'answer': a1,
                    'category': c1 if c1 in ['general', 'cancer', 'nerve'] else 'general',
                    'source_url': post.get('url', ''),
                    'source_title': post.get('title', ''),
                })
        
        # Parse Q2/A2
        q2_match = re.search(r'질문2:\s*(.+?)(?=\n답변2:|$)', text, re.DOTALL)
        a2_match = re.search(r'답변2:\s*(.+?)(?=\n카테고리2:|$)', text, re.DOTALL)
        c2_match = re.search(r'카테고리2:\s*(\w+)', text)
        
        if q2_match and a2_match:
            q2 = q2_match.group(1).strip()
            a2 = a2_match.group(1).strip()
            c2 = c2_match.group(1).strip() if c2_match else 'general'
            
            if q2 and a2 and q2 != '없음' and a2 != '없음':
                results.append({
                    'question': q2,
                    'answer': a2,
                    'category': c2 if c2 in ['general', 'cancer', 'nerve'] else 'general',
                    'source_url': post.get('url', ''),
                    'source_title': post.get('title', ''),
                })
        
        print(f"[QA] Generated {len(results)} Q/A pairs")
        return results
    
    def transform_batch(self, posts: List[Dict], delay: float = 1.0) -> List[Dict]:
        """여러 글 일괄 변환"""
        all_qas = []
        
        for i, post in enumerate(posts):
            print(f"\n[QA] {i+1}/{len(posts)}")
            qa_list = self.transform_single(post)
            all_qas.extend(qa_list)
            
            if i < len(posts) - 1 and delay > 0:
                time.sleep(delay)
        
        print(f"\n[QA] Total: {len(all_qas)} Q/A pairs")
        return all_qas
    
    def format_for_faqs_table(self, qa_list: List[Dict]) -> List[Dict]:
        """hospital_faqs 테이블 형식으로 변환"""
        formatted = []
        
        for qa in qa_list:
            content = f"Q: {qa['question']}\nA: {qa['answer']}"
            
            metadata = {
                "source": qa.get('source_url', ''),
                "title": qa.get('source_title', ''),
                "type": "blog_qa",
                "category": qa.get('category', 'general'),
                "question": qa['question'],
                "answer": qa['answer'],
            }
            
            formatted.append({
                "content": content,
                "metadata": metadata,
            })
        
        return formatted


import re  # for _parse_response


def run_qa_transformation(posts: List[Dict]) -> List[Dict]:
    """전체 Q/A 변환 파이프라인"""
    transformer = QATransformer()
    qa_list = transformer.transform_batch(posts, delay=1.0)
    formatted = transformer.format_for_faqs_table(qa_list)
    return formatted


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test
    sample = {
        "title": "암 환자 식단 관리",
        "content": "암 치료 중에는 균형 잡힌 영양 섭취가 중요합니다. 단백질을 충분히 섭취하고, 신선한 채소와 과일을 먹으세요.",
        "url": "https://example.com/1"
    }
    
    t = QATransformer()
    result = t.transform_single(sample)
    print("\nResult:", json.dumps(result, ensure_ascii=False, indent=2))
