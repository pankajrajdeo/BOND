import os
from typing import Callable, List
import time
import numpy as np

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

 

def _norm(vectors: List[List[float]]) -> List[List[float]]:
    arr = np.asarray(vectors, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    return arr.tolist()



def resolve_embeddings(spec: str, batch_size: int = 8, **keys) -> Callable[[List[str]], List[List[float]]]:
    """
    Returns a function: List[str] -> List[List[float]].
    Supported specs:
      - "litellm:<provider/model>" (e.g., litellm:text-embedding-3-small or litellm:openai/text-embedding-3-small)
      - "ollama:<model>" (convenience alias for ollama/ models via LiteLLM)
      - implicit LiteLLM when the spec looks like a provider/model (contains '/')
    Args:
      - batch_size: Default batch size for inference (default: 8). FAISS generation can override with env variable.
    Notes:
      - Do NOT split on ':' generically; Ollama tags like ":latest" are valid model names.
      - Use OLLAMA_API_BASE env var for remote Ollama endpoints (defaults to localhost:11434).
    """
    if spec.startswith("litellm:"):
        provider, model = ("litellm", spec[len("litellm:"):])
    elif spec.startswith("ollama:"):
        # Convenience alias retained for BC (routes to LiteLLM)
        return resolve_embeddings(f"litellm:ollama/{spec.split(':', 1)[1]}", batch_size=batch_size, **keys)
    else:
        # Auto-detect: if spec looks like a LiteLLM model (contains '/'), route via LiteLLM
        provider, model = ("litellm", spec)

    if provider == "litellm":
        # Handle Ollama models directly due to LiteLLM endpoint issues
        if model.startswith("ollama/"):
            import requests
            import json
            
            # Extract model name (remove "ollama/" prefix)
            ollama_model = model[7:]  # Remove "ollama/" prefix
            api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            
            embed_timeout = int(os.getenv("BOND_EMBED_TIMEOUT", "30"))
            embed_retries = int(os.getenv("BOND_EMBED_RETRIES", "3"))
            def embed(texts: List[str]) -> List[List[float]]:
                vectors = []
                for text in texts:
                    last_err = None
                    for attempt in range(embed_retries):
                        try:
                            response = requests.post(
                                f"{api_base}/api/embeddings",
                                json={"model": ollama_model, "prompt": text},
                                headers={"Content-Type": "application/json"},
                                timeout=embed_timeout,
                            )
                            response.raise_for_status()
                            data = response.json()
                            vectors.append(data["embedding"])
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            time.sleep(0.5 * (2 ** attempt))
                    if last_err is not None:
                        raise RuntimeError(f"Ollama embedding failed for text '{text[:50]}...': {last_err}")
                return _norm(vectors)
            return embed
        else:
            # OpenAI-format across providers (OpenAI, Azure, Anthropic, Gemini/Vertex, Cohere,
            # Mistral, Groq, AWS Bedrock, NVIDIA NIM, HuggingFace Inference, Voyage, Together, etc.)
            import litellm
            from litellm import embedding as llm_embedding
            # Some providers may not support all params; instruct LiteLLM to drop unknowns
            try:
                litellm.drop_params = True
            except Exception:
                pass

            api_base = None  # Use provider defaults for non-Ollama models
            embed_timeout = int(os.getenv("BOND_EMBED_TIMEOUT", "30"))
            embed_retries = int(os.getenv("BOND_EMBED_RETRIES", "3"))
            def embed(texts: List[str]) -> List[List[float]]:
                kwargs = {"model": model, "input": texts, "api_base": api_base, "timeout": embed_timeout, "request_timeout": embed_timeout}
                kwargs.update(keys)
                last_err = None
                for attempt in range(embed_retries):
                    try:
                        resp = llm_embedding(**kwargs)
                        vectors = [item["embedding"] for item in resp["data"]]
                        return _norm(vectors)
                    except Exception as e:
                        last_err = e
                        time.sleep(0.5 * (2 ** attempt))
                raise RuntimeError(f"Embedding provider failed after retries: {last_err}")
            return embed

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
        # Use LiteLLM for all models with retry/backoff + timeout
        timeout = int(os.getenv("BOND_LLM_TIMEOUT", "30"))
        max_retries = int(os.getenv("BOND_LLM_RETRIES", "3"))
        kwargs = {
            "model": self._wire_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_base": self._api_base,
            "timeout": timeout,
            "request_timeout": timeout,
        }
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._completion(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"LLM completion failed after retries: {last_err}")
