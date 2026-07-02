"""
Module 2: Job Description Intelligence
Module 6: Semantic Matching Engine (cosine similarity over DNA vectors)
Module 7: Explainable AI
"""

import math
from typing import Dict, List
from inference_engine import CAPABILITY_SIGNALS, infer_capability_scores

# Weighting scheme from the brief (Module 6)
DIMENSION_WEIGHTS = {
    "backend": 0.08, "frontend": 0.08, "cloud": 0.08, "devops": 0.08, "ai_ml": 0.08,
    "system_design": 0.08,          # -> technical skills bucket ~= 0.40 combined via these 6*~0.067
    "leadership": 0.05,
    "communication": 0.05,
    "problem_solving": 0.15,        # counted partly under "experience"
    "learning_agility": 0.10,       # "learning potential"
    "collaboration": 0.05,          # "culture fit"
}
# Normalize weights to sum to 1.0 exactly
_w_sum = sum(DIMENSION_WEIGHTS.values())
DIMENSION_WEIGHTS = {k: v / _w_sum for k, v in DIMENSION_WEIGHTS.items()}


def generate_job_dna(jd_text: str, title: str = "") -> dict:
    """Module 2: turn a pasted JD into a desired capability profile."""
    raw_scores = infer_capability_scores(jd_text)
    # For a JD, "score" = how strongly/explicitly this capability is required
    dna = {cap: data["score"] for cap, data in raw_scores.items()}
    required = [cap for cap, s in dna.items() if s >= 40]
    return {
        "title": title,
        "dna": dna,
        "required_capabilities": required,
        "details": raw_scores,
    }


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    keys = vec_a.keys()
    dot = sum(vec_a[k] * vec_b.get(k, 0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def weighted_match_score(candidate_dna: Dict[str, float], job_dna: Dict[str, float]) -> float:
    """
    Weighted score: only dimensions the JD actually asks for (score >= 15)
    count toward the requirement match, with weight redistributed among
    them proportionally. Dimensions the JD doesn't mention contribute a
    small bonus for extra candidate strength, but never dilute the core
    requirement match the way an unweighted average would.
    """
    required_dims = {d: w for d, w in DIMENSION_WEIGHTS.items() if job_dna.get(d, 0) >= 15}
    bonus_dims = {d: w for d, w in DIMENSION_WEIGHTS.items() if job_dna.get(d, 0) < 15}

    core_score = 0.0
    if required_dims:
        w_sum = sum(required_dims.values())
        for dim, weight in required_dims.items():
            required = job_dna[dim]
            have = candidate_dna.get(dim, 0)
            ratio = have / required
            dim_score = min(120, ratio * 100)  # allow slight over-qualification credit
            core_score += dim_score * (weight / w_sum)
        core_score = min(100, core_score)
    else:
        core_score = 100  # JD specified nothing concrete -> don't penalize

    bonus_score = 0.0
    if bonus_dims:
        w_sum = sum(bonus_dims.values())
        for dim, weight in bonus_dims.items():
            have = candidate_dna.get(dim, 0)
            bonus_score += have * (weight / w_sum)

    # Core requirements dominate; extra unrequested strengths add a small boost
    final = core_score * 0.9 + bonus_score * 0.1
    return round(min(100, final), 1)


def explain_match(candidate_dna: Dict[str, float], job_dna: Dict[str, float],
                   candidate_details: Dict[str, dict]) -> dict:
    """Module 7: Explainable AI — evidence-backed strengths & gaps."""
    strengths = []
    gaps = []
    for dim, required in job_dna.items():
        have = candidate_dna.get(dim, 0)
        label = dim.replace("_", " ").title()
        if required >= 40:
            if have >= required:
                reasons = candidate_details.get(dim, {}).get("reasons", [])
                evidence = reasons[0] if reasons else "Meets requirement"
                strengths.append(f"{label} ({have}/100) — {evidence}")
            elif have < required * 0.6:
                gaps.append(f"{label}: JD wants {required}/100, candidate shows {have}/100")
    return {"strengths": strengths, "gaps": gaps}


def rank_candidates(candidates: List[dict], job_dna: dict) -> List[dict]:
    """Module 6 + Ranking Engine: score & sort a list of candidate DNA profiles."""
    ranked = []
    for c in candidates:
        cos = cosine_similarity(c["dna"], job_dna["dna"])
        weighted = weighted_match_score(c["dna"], job_dna["dna"])
        explanation = explain_match(c["dna"], job_dna["dna"], c.get("details", {}))
        ranked.append({
            "name": c.get("name", "Candidate"),
            "cosine_similarity": round(cos, 3),
            "match_score": weighted,
            "strengths": explanation["strengths"],
            "gaps": explanation["gaps"],
        })
    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return ranked
