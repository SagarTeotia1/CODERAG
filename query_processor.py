import re
from typing import List, Dict

STOPWORDS = set(
    [
        "the","is","at","which","on","and","a","an","of","to","in","for","by","with","as","from","that","this"
    ]
)

ACRONYM_MAP = {
    "ai": "artificial intelligence",
    "covid": "covid-19",
}

def clean_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[\t\r\n]+", " ", q)
    q = re.sub(r"[^\w\s\-:/]", "", q)
    return q

def simplify_query(q: str) -> str:
    tokens = [t for t in q.lower().split() if t not in STOPWORDS]
    return " ".join(tokens)

def expand_acronyms(q: str) -> str:
    def repl(m):
        t = m.group(0).lower()
        return ACRONYM_MAP.get(t, t)
    return re.sub(r"\b[A-Za-z][A-Za-z0-9\-]{1,}\b", repl, q)

def keyword_only(q: str) -> str:
    tokens = [t for t in re.split(r"\W+", q.lower()) if t and t not in STOPWORDS]
    return " ".join(tokens)

def question_reformulation(q: str) -> str:
    ql = q.lower().strip()
    if ql.startswith("what is "):
        return ql.replace("what is ", "definition of ")
    if ql.startswith("what are "):
        return ql.replace("what are ", "definition of ")
    if ql.startswith("who is "):
        return ql.replace("who is ", "biography of ")
    if ql.startswith("how does "):
        return ql.replace("how does ", "effect of ")
    if ql.endswith("?"):
        return ql[:-1]
    return q

def extract_entities(q: str) -> List[str]:
    quoted = re.findall(r'"([^"]+)"', q)
    capitals = re.findall(r"\b([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*)\b", q)
    return list({*quoted, *capitals})

def multi_query_generate(q: str, num_variants: int = 4) -> List[str]:
    base = clean_query(q)
    variants = [base]
    variants.append(simplify_query(base))
    variants.append(expand_acronyms(base))
    variants.append(question_reformulation(base))
    variants.append(keyword_only(base))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for v in variants:
        if v and v not in seen:
            out.append(v)
            seen.add(v)
        if len(out) >= max(3, num_variants):
            break
    return out

def classify_query(q: str) -> str:
    ql = q.lower()
    if any(ql.startswith(p) for p in ["what", "who", "when", "where"]):
        return "factual"
    if "difference between" in ql or "vs" in ql:
        return "comparison"
    if "affect" in ql or "impact" in ql or "relationship" in ql:
        return "multi-hop"
    if ql.startswith("define ") or ql.startswith("what is "):
        return "definition"
    return "general"

def extract_query_signals(q: str) -> Dict[str, List[str]]:
    entities = extract_entities(q)
    years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", q)
    return {"entities": entities, "years": years}


