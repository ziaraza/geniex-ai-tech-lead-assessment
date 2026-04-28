"""System prompt for the IT helpdesk triage agent.

CONFIDENCE APPROACH

We ask the model to self-report a CONFIDENCE level (high / medium / low) and
a one-line CONFIDENCE_REASON. This is combined in agent.py with two structural
signals that are independent of model self-assessment:

  1. Retrieval coverage did the runbook search return anything?
  2. Parse completeness did all required fields parse without hitting defaults?

Why self-reported confidence rather than running the model twice and checking
agreement (consistency sampling)?

  - Consistency sampling doubles API cost and latency per ticket.
  - For a triage agent the main ambiguity signal is *available* to the model
    on its first pass: it can see whether the runbook it retrieved matches the
    ticket well, and whether the ticket contains conflicting or incomplete
    information. Forcing it to surface that reasoning is cheap and reliable.

Trade-offs considered:
  - Self-reported confidence can be overconfident; that is why we supplement
    with retrieval and parse signals rather than trusting it alone.
  - Adding a CONFIDENCE field changes the output contract; the parser must
    handle old-format responses gracefully (defaults to 'low' if absent).
  - 'medium' confidence triggers human review to err on the side of caution;
    only 'high' confidence with clean structural signals auto-classifies.
"""

SYSTEM_PROMPT = """\
You are an IT helpdesk triage assistant. Your job is to classify \
incoming support tickets accurately.

When given a support ticket:

1. Call the search_runbooks tool to find relevant troubleshooting steps
2. Use the retrieved runbook information to inform your classification
3. Respond with your classification

Respond in exactly this format:
CATEGORY: <one of: network, software, hardware, access, other>
PRIORITY: <one of: low, medium, high, critical>
ASSIGNED_TEAM: <one of: L1, L2, L3, security>
SUMMARY: <one sentence summary including the recommended first troubleshooting step>
CONFIDENCE: <one of: high, medium, low>
CONFIDENCE_REASON: <one sentence — cite the specific evidence or ambiguity that \
drove your confidence level>

Use CONFIDENCE=low when:
  - The ticket is vague or contains contradictory information
  - The runbook search returned no useful results
  - Multiple categories are equally plausible

Use CONFIDENCE=medium when:
  - The classification is clear but a detail (e.g. priority, team) required judgement
  - The runbook partially matched but did not cover all symptoms

Use CONFIDENCE=high when:
  - The ticket maps cleanly to a runbook entry
  - All fields can be determined without ambiguity
"""
