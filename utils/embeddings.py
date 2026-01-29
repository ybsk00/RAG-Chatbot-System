import google.generativeai as genai
from config.settings import GOOGLE_API_KEY, EMBEDDING_MODEL

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def get_embedding(text: str) -> list:
    """
    Generates embedding for a single string using Google GenAI.
    Returns a list of floats.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    # model name usually "models/text-embedding-004" or similar
    # The config might have just "text-embedding-004", so we ensure "models/" prefix if needed
    model_name = EMBEDDING_MODEL
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    try:
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document" # or retrieval_query depending on usage, but document is safe default for storage
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_query_embedding(text: str) -> list:
    """
    Generates embedding for a query.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    model_name = EMBEDDING_MODEL
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    try:
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []