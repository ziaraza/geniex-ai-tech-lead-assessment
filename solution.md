# Solution — M Zia

---

## Q1: Retrieval Fix

### Is increasing `top_k` from 3 to 10 effective?

No. The corpus has 8 chunks total, so `top_k=10` simply returns everything — flooding the model with irrelevant content (hardware diagnostics, access management, etc.) and increasing the chance the model anchors to the wrong material. It treats a symptom, not the cause.

### Root Cause

The bug is in `embed_query` in `retrieval.py`. It uses **brittle string-pattern matching** to produce fake embeddings instead of calling a real embedding model. There are exactly three hardcoded patterns:

```python
"vpn connection error"
"salesforce access"
"excel crash add-in"
```

The LLM will generate its own search query from the ticket — something like `"VPN error troubleshooting"` or `"VPN connectivity issue"`. Neither of these is a substring of `"vpn connection error"`, so the pattern match fails. The fallback checks for bare category keywords (`"network"`, `"software"`, etc.); if the query contains `"network"`, it returns a boosted-network embedding. But if the query is `"VPN connectivity issue"`, none of the category keywords match either, and the fallback returns a **uniform `[0.25, ...]` embedding** — which scores well below the 0.85 threshold against every chunk, so `retrieve_chunks` returns nothing.

When retrieval returns nothing, `format_results` returns the generic escape string `"No matching runbook found. Escalate to L2 support."`, and the model falls back to reasoning from the ticket text alone — producing a plausible but generic classification with no specific error-619 guidance. (If the agent does happen to query with a string that loosely matches "vpn connection error", `net-001-a` ranks marginally higher than `net-001-b` due to embedding values, so the error-code chunk is deprioritised in the summary even when it is retrieved.)

### Fix — Minimal Change

Replace the fake embedding lookup with a **hybrid retrieval**: keep cosine similarity for semantic ranking, but add a keyword-match fallback that always promotes chunks whose `content` contains specific tokens from the query.

**`retrieval.py` — change only `retrieve_chunks`:**

```python
import re

def _keyword_boost(query_text: str, chunk: RunbookChunk) -> bool:
    """
    True if any 'distinctive' token in the query (numbers, capitalised words,
    known error-code patterns) appears verbatim in the chunk content.
    Runs only as a fallback — never used to demote chunks.
    """
    tokens = re.findall(r'\b(?:\d+|[A-Z][A-Z0-9]+)\b', query_text)
    content_lower = chunk.content.lower()
    return any(t.lower() in content_lower for t in tokens)


def retrieve_chunks(query_text, top_k=3, similarity_threshold=0.85):
    query_embedding = embed_query(query_text)

    scored = []
    keyword_boosted = []

    for chunk in RUNBOOK_CHUNKS:
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score >= similarity_threshold:
            scored.append((chunk, score))
        elif _keyword_boost(query_text, chunk):
            # Chunk missed semantic threshold but contains a specific token
            # from the query (e.g. "619"). Include it at its actual score.
            keyword_boosted.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = scored[:top_k]

    # Append keyword-boosted chunks not already present, up to top_k
    existing_ids = {c.chunk_id for c, _ in results}
    for item in keyword_boosted:
        if item[0].chunk_id not in existing_ids and len(results) < top_k:
            results.append(item)

    return results
```

**Why this is the smallest fix**: it is a pure addition to `retrieve_chunks` — no other file changes, no new dependencies, no schema changes. For TKT-001 the query will contain `"619"`, which appears verbatim in `net-001-b`, so that chunk is guaranteed to be included regardless of embedding quality. For all other tickets the behaviour is unchanged.

---

## Q2: Code Review Triage

| # | Flagged Issue | Verdict | Reason |
|---|---|---|---|
| A | Conversation history shared across tickets | **Real bug — must fix** | Cross-ticket history pollutes context: ticket N's runbook results and classification influence ticket N+1's reasoning, producing non-deterministic and potentially incorrect outputs at scale. |
| B | `_prune_history` uses naive slice deletion | **Acceptable as-is** | For this single-tool agent with bounded history, naive slicing is safe and predictable; the subtle risk of orphaning a mid-turn tool call is mitigated by the small `MAX_HISTORY_MESSAGES` value and the linear tool loop. |
| C | `response.content[0].text` accessed without type check | **Real bug — must fix** | The Anthropic SDK can return `tool_use` blocks as `content[0]`; accessing `.text` on a non-text block raises `AttributeError` and crashes the batch silently without the caller knowing why. |
| D | No retry logic on `messages.create()` | **Acceptable as-is** | For a low-volume assessment harness, transient API errors are rare and the outer `try/except` in `main.py` already logs and continues; adding retry is correct for production but not a correctness bug here. |
| E | Tool dispatch uses `if` instead of a registry | **Acceptable as-is** | With a single tool the `if` branch is readable and correct; a registry adds indirection with no benefit until a second tool is added. |
| F | Tool inputs accessed via `params.get()` without Pydantic validation | **Acceptable as-is** | The only caller is the Anthropic SDK, which validates against the declared `input_schema` before dispatching, so malformed inputs are already rejected upstream; Pydantic would add safety against direct calls but is not necessary here. |
