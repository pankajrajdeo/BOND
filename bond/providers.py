import os
from typing import Callable, List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .logger import logger

def _norm(vectors: List[List[float]]) -> List[List[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    return arr.tolist()



def resolve_embeddings(spec: str, **keys) -> Callable[[List[str]], List[List[float]]]:
    """
    Returns a function: List[str] -> List[List[float]].
    Supported specs:
      - "st:<hf-model-name>" (local SentenceTransformers)
      - "litellm:<provider/model>" (e.g., litellm:text-embedding-3-small or litellm:openai/text-embedding-3-small)
      - "ollama:<model>" (convenience alias for ollama/ models via LiteLLM)
      - implicit LiteLLM when the spec looks like a provider/model (contains '/')
    Notes:
      - Do NOT split on ':' generically; Ollama tags like ":latest" are valid model names.
      - Use OLLAMA_API_BASE env var for remote Ollama endpoints (defaults to localhost:11434).
    """
    if spec.startswith("st:"):
        provider, model = ("st", spec[len("st:"):])
    elif spec.startswith("litellm:"):
        provider, model = ("litellm", spec[len("litellm:"):])
    elif spec.startswith("ollama:"):
        # Convenience alias retained for BC (routes to LiteLLM)
        return resolve_embeddings(f"litellm:ollama/{spec.split(':', 1)[1]}")
    else:
        # Auto-detect: if spec looks like a LiteLLM model (contains '/'), route via LiteLLM
        provider, model = ("litellm", spec) if ("/" in spec) else ("st", spec)

    if provider == "st":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st_batch_size = int(os.getenv("BOND_EMB_BATCH", "16"))
        logger.debug(f"Loading SentenceTransformer model '{model}' on device '{device}' with batch_size={st_batch_size}")
        st_model = SentenceTransformer(model, device=device)
        def embed(texts: List[str]) -> List[List[float]]:
            vectors = st_model.encode(
                texts,
                batch_size=st_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _norm(vectors.tolist())
        return embed

    if provider == "litellm":
        # OpenAI-format across providers (OpenAI, Azure, Anthropic, Gemini/Vertex, Cohere,
        # Mistral, Groq, AWS Bedrock, NVIDIA NIM, HuggingFace Inference, Voyage, Together, Ollama, etc.)
        import litellm
        from litellm import embedding as llm_embedding
        # Some providers may not support all params; instruct LiteLLM to drop unknowns
        try:
            litellm.drop_params = True
        except Exception:
            pass

        # Single API base: only honor OLLAMA_API_BASE for ollama/* models; otherwise let provider defaults apply.
        # If OLLAMA_API_BASE is unset, LiteLLM will default to localhost for Ollama.
        api_base = os.getenv("OLLAMA_API_BASE") if model.startswith("ollama/") else None
        def embed(texts: List[str]) -> List[List[float]]:
            # Use LiteLLM for all Ollama endpoints
            kwargs = {"model": model, "input": texts, "api_base": api_base}
            kwargs.update(keys)
            resp = llm_embedding(**kwargs)
            vectors = [item["embedding"] for item in resp["data"]]
            return _norm(vectors)
        return embed

    # (kept for completeness; typical path handled above)
    if provider == "ollama":
        return resolve_embeddings(f"litellm:ollama/{model}")

    raise ValueError(f"Unknown embedding spec: {spec}")

class ChatLLM:
    """
    LiteLLM chat completions for all providers.
    Accepts model like "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet", "google/gemini-1.5-pro",
    "groq/llama-3.1-70b", "azure/gpt-4o-mini", "ollama/llama3", etc.
    """
    def __init__(self, model: str):
        self._model = model
        # LiteLLM for all models
        import litellm
        from litellm import completion
        self._completion = completion
        # Accept both "openai/gpt-4o-mini" and "text-embedding-3-small" style identifiers
        self._wire_model = model.replace(":", "/")
        # Single API base: only honor OLLAMA_API_BASE for ollama/* models
        self._api_base = os.getenv("OLLAMA_API_BASE") if self._wire_model.startswith("ollama/") else None
        # Ensure unknown params are safely dropped
        try:
            litellm.drop_params = True
        except Exception:
            pass

    def text(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        # Use LiteLLM for all models
        kwargs = {
            "model": self._wire_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_base": self._api_base,
        }
        
        resp = self._completion(**kwargs)
        return resp.choices[0].message.content.strip()
