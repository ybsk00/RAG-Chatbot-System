from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class Preprocessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def format_context_rich_chunk(self, chunk_text: str, metadata: Dict) -> str:
        """
        Adds context information to the chunk text.
        Format: "이 답변은 [Title]에서 추출되었으며, 주요 주제는 [Topic]입니다.\n\n{text}"
        """
        title = metadata.get('title', 'Unknown Source')
        # Topic extraction could be done via LLM, but for now we use a placeholder or title
        # If we had topics, we would use them.
        
        context_header = f"이 내용은 '{title}'에서 추출되었습니다."
        if 'upload_date' in metadata:
            context_header += f" (작성일: {metadata['upload_date']})"
        
        return f"{context_header}\n\n{chunk_text}"

    def process_content(self, content_item: Dict) -> List[Dict]:
        """
        Splits content into chunks and adds metadata.
        """
        text = content_item.get('transcript') or content_item.get('content')
        if not text:
            return []

        chunks = self.text_splitter.split_text(text)
        processed_chunks = []

        for chunk in chunks:
            rich_text = self.format_context_rich_chunk(chunk, content_item)
            processed_chunks.append({
                'content': rich_text,
                'original_content': chunk, # Keep original for display if needed
                'metadata': {
                    'source': content_item.get('url'),
                    'title': content_item.get('title'),
                    'type': 'youtube' if 'transcript' in content_item else 'blog'
                }
            })
            
        return processed_chunks
