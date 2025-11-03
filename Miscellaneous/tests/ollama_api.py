import time
import os
from litellm import embedding

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
    pass

# Configuration from environment variables
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)
MODEL_NAME = "ollama/rajdeopankaj/bond-embed-v1-fp16:latest"

# Generate a small set of test sentences
test_texts = [
    "This is a test sentence for embedding generation.",
    "Machine learning models process natural language efficiently.",
    "Biomedical ontologies help standardize terminology.",
    "BOND provides entity linking for biomedical texts.",
    "Embedding models convert text into vector representations."
]


def test_embedding():
    """Simple test of embedding API using environment variable configuration"""
    print(f"🧪 Testing Embedding API")
    print(f"API Base: {OLLAMA_API_BASE}")
    print(f"Model: {MODEL_NAME}")
    if OLLAMA_API_KEY:
        print("Authentication: API Key provided")
    else:
        print("Authentication: None (localhost)")
    print("-" * 50)
    
    try:
        start = time.time()
        
        kwargs = {
            "model": MODEL_NAME,
            "input": test_texts,
            "api_base": OLLAMA_API_BASE,
            "timeout": 60  # 1 minute timeout
        }
        
        # Add API key if provided
        if OLLAMA_API_KEY:
            kwargs["api_key"] = OLLAMA_API_KEY
        
        print(f"📝 Processing {len(test_texts)} test sentences...")
        response = embedding(**kwargs)
        
        total_time = time.time() - start
        
        # Display results
        print(f"✅ SUCCESS!")
        print(f"   Time taken: {total_time:.2f} seconds")
        print(f"   Embeddings generated: {len(response.data)}")
        
        # Handle different response formats
        if response.data:
            # Check if it's a dict or object
            first_embedding = response.data[0]
            if hasattr(first_embedding, 'embedding'):
                embedding_vector = first_embedding.embedding
            elif isinstance(first_embedding, dict) and 'embedding' in first_embedding:
                embedding_vector = first_embedding['embedding']
            else:
                # Assume the data IS the embedding vector
                embedding_vector = first_embedding
            
            print(f"   Embedding dimension: {len(embedding_vector)}")
            print(f"   Average time per text: {total_time/len(test_texts):.3f} seconds")
            
            print(f"\n📊 Sample embedding (first text):")
            print(f"   Text: '{test_texts[0]}'")
            print(f"   Embedding vector length: {len(embedding_vector)}")
            print(f"   First 5 values: {embedding_vector[:5]}")
        else:
            print(f"   No embedding data returned")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 EMBEDDING API TEST")
    print("=" * 50)
    
    # Run simple embedding test
    test_embedding()
    
    print("\n💡 Usage Examples:")
    print("# For localhost (default):")
    print("python3 ollama_api.py")
    print()
    print("# For Windows GPU API:")
    print("OLLAMA_API_BASE=http://10.30.144.32:11434 python3 ollama_api.py")
    print()
    print("# For CCHMC API:")
    print("OLLAMA_API_BASE=https://llm.research.cchmc.org/ollama \\")
    print("OLLAMA_API_KEY=sk-45fd9465d2f2439fa94778d89d7d9f5c \\")
    print("python3 ollama_api.py")