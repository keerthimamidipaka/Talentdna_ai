"""
Optional LLM Reasoning Layer
----------------------------
The rule-based inference_engine.py is the reliable baseline: fast, free,
deterministic, fully explainable. This module adds an OPTIONAL second pass
using Claude to catch things keyword matching misses — e.g. a resume that
says "orchestrated our container fleet across three regions" instead of
the literal word "Kubernetes".

Behavior:
- If ANTHROPIC_API_KEY is not set, this module is skipped entirely and the
  API falls back to rule-based-only DNA (no error, no degraded UX).
- If a key IS set, Claude reviews the resume + the rule-based scores and
  returns adjustments, each with a one-line justification. Adjustments are
  capped at +/-15 points per dimension so a single LLM call can't wildly
  override the evidence-based baseline — it can only refine it.
- Final DNA = rule-based score, nudged by any capped LLM adjustment.
  Every adjustment is logged as a reason, so explainability is preserved.
"""

import os
import json
import re
from typing import Optional

MAX_ADJUSTMENT = 15
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You review resumes for a recruiting platform. You will be given:
1. Resume text
2. A rule-based capability score (0-100) per dimension, derived from keyword matching

Your job: look for capability evidence the keyword-matcher likely MISSED because it used
different wording (e.g. "orchestrated our container fleet" implies devops/kubernetes even
without the word "Kubernetes"). Only suggest an adjustment when you see real textual evidence.

Respond with ONLY a JSON object, no other text, no markdown fences:
{
  "adjustments": [
    {"capability": "<dimension_name>", "delta": <integer between -15 and 15>, "reason": "<one short sentence citing the specific evidence>"}
  ]
}

Only include dimensions where you found clear additional evidence or a clear over-estimate.
If nothing to adjust, return {"adjustments": []}. Valid dimension names: backend, frontend,
cloud, devops, ai_ml, leadership, communication, problem_solving, system_design,
learning_agility, collaboration.
"""


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def refine_dna_with_llm(resume_text: str, rule_based_dna: dict) -> dict:
    """
    Returns {"used_llm": bool, "dna": {...refined...}, "adjustments": [...] }
    Never raises — falls back silently to the rule-based DNA on any error,
    so the API stays reliable even if the LLM call fails or times out.
    """
    if not is_available():
        return {"used_llm": False, "dna": rule_based_dna, "adjustments": []}

    try:
        import anthropic
        client = anthropic.Anthropic()

        user_content = (
            f"Rule-based scores:\n{json.dumps(rule_based_dna, indent=2)}\n\n"
            f"Resume text:\n{resume_text[:6000]}"
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw_text = "".join(block.text for block in response.content if block.type == "text")
        parsed = _extract_json(raw_text)
        if not parsed or "adjustments" not in parsed:
            return {"used_llm": True, "dna": rule_based_dna, "adjustments": [],
                     "note": "LLM response could not be parsed; used rule-based scores only."}

        refined = dict(rule_based_dna)
        applied = []
        for adj in parsed["adjustments"]:
            cap = adj.get("capability")
            delta = adj.get("delta", 0)
            reason = adj.get("reason", "")
            if cap not in refined:
                continue
            delta = max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, int(delta)))
            new_score = max(0, min(100, refined[cap] + delta))
            refined[cap] = new_score
            applied.append({"capability": cap, "delta": delta, "reason": reason, "new_score": new_score})

        return {"used_llm": True, "dna": refined, "adjustments": applied}

    except Exception as e:
        # Never let an LLM/network failure break candidate upload
        return {"used_llm": False, "dna": rule_based_dna, "adjustments": [],
                "error": f"LLM reasoning unavailable: {str(e)}"}
