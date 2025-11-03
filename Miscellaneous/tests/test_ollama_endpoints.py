#!/usr/bin/env python3
"""
Test script to verify BOND Ollama support across different API endpoints:
- localhost (default)
- Windows remote GPU 
- CCHMC authenticated API
"""

import os
import sys
sys.path.insert(0, '.')

from bond.providers import resolve_embeddings, ChatLLM

def test_embedding_api(name, api_base=None, api_key=None):
    """Test embedding functionality"""
    print(f"\n🧪 Testing {name}")
    print(f"API Base: {api_base or 'localhost:11434'}")
    print(f"API Key: {'✓ Provided' if api_key else '✗ None'}")
    print("-" * 40)
    
    # Set environment variables
    if api_base:
        os.environ["OLLAMA_API_BASE"] = api_base
    else:
        os.environ.pop("OLLAMA_API_BASE", None)
        
    if api_key:
        os.environ["OLLAMA_API_KEY"] = api_key
    else:
        os.environ.pop("OLLAMA_API_KEY", None)
    
    try:
        # Test embedding
        embed_fn = resolve_embeddings("ollama/rajdeopankaj/bond-embed-v1-fp16:latest")
        test_texts = ["This is a test sentence.", "BOND provides entity linking."]
        
        print("📝 Testing embeddings...")
        embeddings = embed_fn(test_texts)
        print(f"✅ Embeddings: {len(embeddings)} vectors, {len(embeddings[0])}D")
        
        # Test LLM
        print("🤖 Testing LLM...")
        llm = ChatLLM("ollama/llama3")
        response = llm.text("What is biomedical entity linking?", max_tokens=50)
        print(f"✅ LLM Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def main():
    print("🚀 BOND Ollama Endpoint Test")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Localhost (default)
    results["localhost"] = test_embedding_api("Localhost")
    
    # Test 2: Windows Remote GPU
    results["windows"] = test_embedding_api(
        "Windows Remote GPU",
        api_base="http://10.30.144.32:11434"
    )
    
    # Test 3: CCHMC Authenticated API  
    results["cchmc"] = test_embedding_api(
        "CCHMC Research API",
        api_base="https://llm.research.cchmc.org/ollama",
        api_key="sk-45fd9465d2f2439fa94778d89d7d9f5c"
    )
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    for endpoint, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{endpoint.ljust(15)}: {status}")
    
    total_pass = sum(results.values())
    print(f"\nResult: {total_pass}/{len(results)} endpoints working")
    
    if total_pass == len(results):
        print("🎉 All Ollama endpoints are working with BOND!")
    else:
        print("⚠️  Some endpoints failed - check configuration")

if __name__ == "__main__":
    main()
