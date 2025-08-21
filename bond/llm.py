import json

def extract_json_block(text: str) -> dict:
    """
    Extract JSON from LLM output, handling various formats:
    1. Markdown code fences (```json ... ```)
    2. Plain JSON blocks
    3. Mixed text with JSON
    """
    # First try to find markdown code fences
    import re
    
    # Look for ```json ... ``` blocks
    json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block_match:
        try:
            return json.loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Fallback to original method: find first { and last }
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    
    try:
        return json.loads(text[s:e+1])
    except json.JSONDecodeError:
        return {}
