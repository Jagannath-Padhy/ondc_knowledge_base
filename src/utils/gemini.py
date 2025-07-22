import google.generativeai as genai
from ..utils.config import Config

class GeminiClient:
    """
    Utility for interacting with Gemini 2.5 Pro for LLM and embedding.
    """
    def __init__(self, config: Config = None):
        self.config = config or Config()
        api_key = self.config.gemini["api_key"]
        genai.configure(api_key=api_key)
        self.model = self.config.gemini.get("model", "gemini-2.5-pro")

    def llm(self, prompt: str) -> str:
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text if hasattr(response, 'text') else str(response)

    def embed(self, text: str):
        # Use the embedding endpoint (if available)
        # For Gemini, use the textembedding-gecko model for embeddings
        embed_model = "models/embedding-001"
        response = genai.embed_content(
            model=embed_model,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]

def get_gemini_llm(config: Config = None):
    client = GeminiClient(config)
    return client.llm

def get_gemini_embed_fn(config: Config = None):
    client = GeminiClient(config)
    return client.embed 