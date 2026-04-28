"""Orchestrator for the IT triage agent.

CONFIDENCE MECHANISM

needs_human_review is set to True when ANY of these conditions hold:

  1. Model-reported confidence is 'low' or 'medium'.
     Rationale: the model has seen the ticket and the runbook content; its
     self-assessment is the richest available signal. We treat 'medium' as
     insufficient for auto-classification to err on the side of caution.

  2. Retrieval returned no results ("No matching runbook found" sentinel).
     Rationale: absence of relevant runbook content is a strong structural
     signal that the ticket falls outside known patterns — the model is
     reasoning from the ticket text alone, which increases error rate.

  3. One or more required output fields fell back to defaults.
     Rationale: a parse failure means the model deviated from the expected
     format, which is itself a sign of uncertainty or an unusual ticket.

Why NOT consistency sampling (run the model twice and compare)?
  - Doubles cost and latency for every ticket.
  - Adds complexity for a signal already available on the first pass.
  - Harder to explain to a human reviewer ("the two runs disagreed" is
    less actionable than "confidence=low because the runbook didn't match").

Trade-offs:
  - 'medium' triggering review may over-flag initially; the threshold can be
    raised to 'low'-only once the team has baseline accuracy data.
  - The retrieval sentinel string is a brittle coupling to retrieval.py's
    output; a structured return value (e.g. empty list vs. string) would be
    cleaner.
"""

from config import client, MODEL, MAX_TOKENS, MAX_HISTORY_MESSAGES
from tools import TOOLS, execute_tool
from prompts import SYSTEM_PROMPT
from parser import parse_triage_response

_NO_RUNBOOK_SENTINEL = "No matching runbook found"


def _prune_history(history_list):
    """Keep only the most recent messages to prevent context exhaustion.

    NOTE: naive slice — safe here because MAX_HISTORY_MESSAGES is small and
    the loop is single-threaded, but should prune at message-pair boundaries
    in a production system to avoid orphaned tool-result messages.
    """
    if len(history_list) > MAX_HISTORY_MESSAGES:
        del history_list[:-MAX_HISTORY_MESSAGES]
    return history_list


def _retrieval_was_empty(conversation_history):
    """Return True if the last tool result was the no-runbook sentinel."""
    for message in reversed(conversation_history):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and _NO_RUNBOOK_SENTINEL in str(block.get("content", ""))
                    ):
                        return True
    return False


def _determine_human_review(result, retrieval_empty):
    """
    Combine three signals to decide whether a ticket needs human review.

    Returns (needs_human_review: bool, reasons: list[str])
    """
    reasons = []

    confidence = result.get("confidence", "low")
    if confidence in ("low", "medium"):
        reasons.append(f"model_confidence={confidence}")

    if retrieval_empty:
        reasons.append("retrieval_returned_no_runbook")

    if result.get("_parse_used_defaults"):
        reasons.append(f"parse_defaults_used={result['_parse_used_defaults']}")

    return bool(reasons), reasons


def triage_ticket(ticket_text, ticket_id, conversation_history):
    """Process a single ticket through the triage loop.

    BUG NOTE (Issue A from Q2): conversation_history is shared across tickets
    in the current main.py. This means context from ticket N bleeds into
    ticket N+1. Each ticket should receive a fresh history. This function
    itself is correct — the bug is in the caller.
    """

    conversation_history.append({"role": "user", "content": ticket_text})

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=conversation_history,
    )

    while response.stop_reason == "tool_use":
        tool_block = next(b for b in response.content if b.type == "tool_use")
        tool_result = execute_tool(tool_block.name, tool_block.input)

        conversation_history.append({"role": "assistant", "content": response.content})
        conversation_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": tool_result,
                    }
                ],
            }
        )

        _prune_history(conversation_history)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation_history,
        )

    text_blocks = [b for b in response.content if b.type == "text"]
    if not text_blocks:
        raise ValueError(
            f"No text block in final response for {ticket_id}. "
            f"Block types: {[b.type for b in response.content]}"
        )
    raw_response = text_blocks[0].text

    conversation_history.append({"role": "assistant", "content": response.content})
    _prune_history(conversation_history)

    result = parse_triage_response(raw_response)
    result["ticket_id"] = ticket_id

    retrieval_empty = _retrieval_was_empty(conversation_history)
    needs_review, review_reasons = _determine_human_review(result, retrieval_empty)

    result["needs_human_review"] = needs_review
    result["review_reasons"] = review_reasons

    result.pop("_parse_used_defaults", None)

    return result
