Solution — M Zia


Q1: Retrieval Fix

Is increasing top_k from 3 to 10 effective?

No. The corpus has only 8 chunks in total, so setting top_k to 10 simply returns everything. This floods the model with irrelevant content like hardware diagnostics and access management guides, and increases the chance it anchors to the wrong material. It treats a symptom rather than the root cause.

Root Cause

The bug is in the embed_query function in retrieval.py. Instead of calling a real embedding model, it uses brittle string-pattern matching to produce fake embeddings. There are exactly three hardcoded patterns: "vpn connection error", "salesforce access", and "excel crash add-in".

The LLM generates its own search query from the ticket text — something like "VPN error troubleshooting" or "VPN connectivity issue". Neither of these is a substring of "vpn connection error", so the pattern match fails. The fallback then checks for bare category keywords like "network" or "software". If the query is "VPN connectivity issue", none of those match either, and the fallback returns a uniform [0.25, 0.25, ...] embedding. That vector scores well below the 0.85 similarity threshold against every chunk, so retrieve_chunks returns nothing.

When retrieval returns nothing, format_results returns the generic escape string "No matching runbook found. Escalate to L2 support." The model then falls back to reasoning from the ticket text alone, producing a plausible but generic classification with no specific error-619 guidance.

There is a secondary issue even when retrieval does succeed. If the agent happens to query with a string that loosely matches "vpn connection error", chunk net-001-a (generic VPN steps) ranks marginally higher than net-001-b (the error-code specific chunk containing the fix for error 619). So even when both chunks are retrieved, the model tends to lead with the generic guidance rather than the specific one.

Fix — Minimal Change

The fix is to add a keyword-match fallback inside retrieve_chunks. The semantic cosine similarity path stays unchanged. A second pass checks whether any distinctive token from the query — numbers, uppercase error codes, specific identifiers — appears verbatim in a chunk's content. If a chunk misses the similarity threshold but contains a matching token, it is added to the results anyway.

For TKT-001 the query will contain "619", which appears verbatim in net-001-b. That chunk is therefore guaranteed to be included regardless of the embedding quality. For all other tickets the behaviour is unchanged. This is a pure addition to one function — no other files change, no new dependencies, no schema changes.

The updated retrieve_chunks:

    import re

    def _keyword_boost(query_text, chunk):
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
                keyword_boosted.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        existing_ids = {c.chunk_id for c, _ in results}
        for item in keyword_boosted:
            if item[0].chunk_id not in existing_ids and len(results) < top_k:
                results.append(item)

        return results


Q2: Code Review Triage

Issue A — Conversation history shared across tickets (main.py)
Verdict: Real bug, must fix.
The single conversation_history list is passed across all tickets in the batch. Ticket N's runbook results and classification bleed into the context for ticket N+1, producing non-deterministic and potentially wrong outputs. Each ticket should get a fresh history.

Issue B — _prune_history uses naive slice deletion (agent.py)
Verdict: Acceptable as-is.
For this single-tool agent with a small MAX_HISTORY_MESSAGES bound and a linear tool loop, naive slicing is safe and predictable. The risk of orphaning a mid-turn tool-result message is real in theory but is mitigated by the small window size.

Issue C — response.content[0].text accessed without type check (agent.py)
Verdict: Real bug, must fix.
The Anthropic SDK can return a tool_use block as content[0] in certain edge cases. Accessing .text on a non-text block raises AttributeError and crashes the batch silently, with the outer try/except in main.py logging a failure but giving no useful diagnostic. The fix is to filter for blocks where type == "text" before accessing .text.

Issue D — No retry logic on messages.create() (agent.py)
Verdict: Acceptable as-is.
For a low-volume local harness, transient API errors are rare and the outer try/except in main.py already logs and continues. Retry logic is correct for production but its absence is not a correctness bug in this context.

Issue E — Tool dispatch uses if instead of a registry (tools.py)
Verdict: Acceptable as-is.
With a single tool, the if branch is readable and correct. A registry pattern adds indirection with no benefit until a second tool is added.

Issue F — Tool inputs accessed via params.get() without Pydantic validation (tools.py)
Verdict: Acceptable as-is.
The only caller is the Anthropic SDK, which validates inputs against the declared input_schema before dispatching. Malformed inputs are already rejected upstream. Pydantic validation would add safety against direct programmatic calls but is not necessary here.
