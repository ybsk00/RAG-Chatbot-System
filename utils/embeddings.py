import os
from google import genai
from config.settings import GOOGLE_API_KEY, EMBEDDING_MODEL

# Initialize client globally or within functions. Doing it globally is efficient if key is present.
client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)

def get_embedding(text: str) -> list:
    """
    Generates embedding for a single string using Google GenAI.
    Returns a list of floats.
    """
    if not client:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    # model name usually "models/text-embedding-004" or similar
    # The config might have just "text-embedding-004", so we ensure "models/" prefix if needed
    model_name = EMBEDDING_MODEL
    # google-genai client typically expects just the model name, but let's keep the logic consistent
    # Check if "models/" is required for the new client. Usually "text-embedding-004" works.
    
    try:
        result = client.models.embed_content(
            model=model_name,
            contents=text,
            config=genai.types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT" # Uppercase enum usually in new client
            )
        )
        # The new client returns an object with 'embeddings'. 
        # For a single content, it's usually result.embeddings[0].values
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def get_query_embedding(text: str) -> list:
    """
    Generates embedding for a query.
    """
    if not client:
        raise ValueError("GOOGLE_API_KEY is not set.")
    
    model_name = EMBEDDING_MODEL

    try:
        result = client.models.embed_content(
            model=model_name,
            contents=text,
            config=genai.types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []