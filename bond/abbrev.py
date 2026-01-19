import os
import json
import re
from typing import Dict, List, Tuple, Optional


class AbbreviationExpander:
    """
    Loads field-scoped abbreviations from assets/abbreviations.json and applies
    alphanumeric-boundary replacements so tokens adjacent to '_' or '+' match.
    """

    def __init__(self, assets_path: str):
        self._pats: Dict[str, List[Tuple[re.Pattern[str], str]]] = {}
        abbr_path = os.path.join(assets_path, "abbreviations.json")
        if not os.path.exists(abbr_path):
            return
        try:
            with open(abbr_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        def compile_map(m: Dict[str, object]) -> List[Tuple[re.Pattern[str], str]]:
            pats: List[Tuple[re.Pattern[str], str]] = []
            for k, v in m.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, list) and v:
                    repl = str(v[0])
                elif isinstance(v, str):
                    repl = v
                else:
                    continue
                # (?<![A-Za-z0-9])token(?![A-Za-z0-9]) to avoid splitting on '_' or '+'
                pat = re.compile(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", re.IGNORECASE)
                pats.append((pat, repl))
            return pats

        if isinstance(data, dict):
            flat_entries: Dict[str, object] = {}
            for top_k, top_v in data.items():
                if isinstance(top_v, dict):
                    self._pats[top_k.lower()] = compile_map(top_v)
                else:
                    flat_entries[top_k] = top_v
            if flat_entries:
                self._pats.setdefault("global", []).extend(compile_map(flat_entries))

    def expand(self, text: str, field_name: Optional[str]) -> str:
        if not text or not self._pats:
            return text
        out = text
        keys: List[str] = []
        if field_name:
            keys.append(field_name.lower())
        keys.extend(["*", "global"])  # optional global sections
        for key in keys:
            for pat, repl in self._pats.get(key, []):
                # Skip expansion if the replacement text is already present in the query
                # This prevents "T cell" -> "t cell cell" when "t": "t cell" exists
                if repl.lower() in text.lower():
                    continue
                out = pat.sub(repl, out)
        return out

