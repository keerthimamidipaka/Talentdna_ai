"""
Capability Inference Engine
----------------------------
Instead of counting keywords, this engine estimates capability SCORES (0-100)
by looking for weighted "evidence" of skill, seniority, and impact in resume text.

Design:
- Each capability dimension has a set of evidence signals (keywords/phrases),
  each with a base weight and an "impact multiplier" bump (leadership verbs,
  scale words, certifications, years of experience, etc.)
- Final score = base signal coverage + impact bonuses, capped at 100,
  with a floor determined by how many signals fired.
- Every score comes with a `reasons` list (the explainability trail).
"""

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Capability dimension -> evidence keywords (case-insensitive, regex-safe)
# ---------------------------------------------------------------------------
CAPABILITY_SIGNALS: Dict[str, List[str]] = {
    "backend": [
        "backend", "back-end", "api", "rest api", "graphql", "microservice",
        "django", "flask", "fastapi", "spring", "node.js", "express",
        "database design", "server-side", "grpc", "sql", "postgresql",
        "mysql", "mongodb", "redis"
    ],
    "frontend": [
        "frontend", "front-end", "react", "vue", "angular", "next.js",
        "typescript", "javascript", "html", "css", "tailwind", "ui/ux",
        "responsive design", "redux", "webpack"
    ],
    "cloud": [
        "aws", "azure", "gcp", "google cloud", "cloud architecture",
        "ec2", "s3", "lambda", "cloudformation", "terraform", "cloud native",
        "aws certified", "azure certified", "gcp certified"
    ],
    "devops": [
        "docker", "kubernetes", "k8s", "ci/cd", "jenkins", "github actions",
        "gitlab ci", "helm", "ansible", "prometheus", "grafana",
        "infrastructure as code", "site reliability", "sre"
    ],
    "ai_ml": [
        "machine learning", "deep learning", "neural network", "pytorch",
        "tensorflow", "nlp", "computer vision", "llm", "generative ai",
        "scikit-learn", "data science", "transformer", "huggingface",
        "model training", "mlops"
    ],
    "leadership": [
        "led a team", "led team", "managed a team", "mentored", "mentor",
        "team lead", "tech lead", "supervised", "coordinated", "led 2",
        "led 3", "led 4", "led 5", "led developers", "led engineers",
        "delivered project", "cross-functional", "stakeholder"
    ],
    "communication": [
        "presented", "presentation", "documentation", "wrote docs",
        "technical writing", "collaborated with", "client-facing",
        "communicated", "public speaking", "conference talk", "blog"
    ],
    "problem_solving": [
        "optimized", "debugged", "root cause", "algorithm", "solved",
        "performance improvement", "reduced latency", "scaled system",
        "troubleshoot", "refactored", "architecture decision"
    ],
    "system_design": [
        "system design", "distributed system", "scalable architecture",
        "designed architecture", "high availability", "fault tolerant",
        "event-driven", "message queue", "kafka", "rabbitmq", "load balancing"
    ],
    "learning_agility": [
        "self-taught", "certified", "certification", "open-source",
        "open source contributor", "hackathon", "side project",
        "continuous learning", "new technology", "quickly learned",
        "adapted to"
    ],
    "collaboration": [
        "cross-functional team", "agile", "scrum", "pair programming",
        "code review", "collaborated", "worked closely with", "team player",
        "kanban"
    ],
}

# Phrases that indicate HIGH IMPACT / seniority -> bonus points when found
# near or within the same resume (applied globally, capped).
IMPACT_BONUSES: List[Tuple[str, int, str]] = [
    (r"\bled\s+\d+\s+(developers|engineers|people)\b", 12, "Led a multi-person team"),
    (r"\b(aws|azure|gcp)\s+certified\b", 10, "Holds a cloud certification"),
    (r"\bopen[- ]source contributor\b", 8, "Active open-source contributor"),
    (r"\b\d+\+?\s+years?\s+(of\s+)?experience\b", 8, "Multiple years of hands-on experience"),
    (r"\b(scaled|scalable)\b.{0,20}\b(million|thousand|users|requests)\b", 10, "Proven experience scaling systems"),
    (r"\breduced\b.{0,20}\b(\d+%|latency|cost|time)\b", 8, "Quantified performance/cost improvement"),
]


def _find_matches(text: str, phrases: List[str]) -> List[str]:
    hits = []
    lower = text.lower()
    for phrase in phrases:
        if phrase in lower:
            hits.append(phrase)
    return hits


def infer_capability_scores(resume_text: str) -> Dict[str, dict]:
    """
    Returns, for each capability dimension:
      { score: int 0-100, evidence: [...], reasons: [...] }
    """
    results = {}
    lower_text = resume_text.lower()

    # Global impact bonuses (apply once each, shared across relevant dims)
    global_bonus_hits = []
    for pattern, bonus, label in IMPACT_BONUSES:
        if re.search(pattern, lower_text):
            global_bonus_hits.append((bonus, label))

    for capability, signals in CAPABILITY_SIGNALS.items():
        hits = _find_matches(resume_text, signals)
        coverage = len(hits) / max(len(signals), 1)

        # Base score: coverage-driven, non-linear so a few strong hits still
        # register meaningfully (avoids penalizing shorter resumes).
        if hits:
            base_score = 35 + min(55, len(hits) * 9)
        else:
            base_score = 0

        # Apply relevant global bonuses (leadership bonus only boosts
        # leadership/collaboration; cert bonus boosts cloud/learning, etc.)
        reasons = [f"Found evidence: '{h}'" for h in hits[:6]]
        bonus_total = 0
        for bonus, label in global_bonus_hits:
            relevant = (
                (capability == "leadership" and "team" in label.lower()) or
                (capability in ("cloud", "learning_agility") and "certification" in label.lower()) or
                (capability == "learning_agility" and "open-source" in label.lower()) or
                (capability in ("backend", "system_design") and ("scal" in label.lower() or "performance" in label.lower())) or
                (capability == "problem_solving" and ("performance" in label.lower() or "cost" in label.lower()))
            )
            if relevant:
                bonus_total += bonus
                reasons.append(f"Bonus: {label}")

        score = min(100, base_score + bonus_total)
        if not hits and bonus_total == 0:
            score = 0
            reasons = ["No direct evidence found in resume"]

        results[capability] = {
            "score": round(score),
            "evidence_count": len(hits),
            "reasons": reasons,
        }

    return results


def generate_dna_json(resume_text: str, candidate_name: str = "") -> dict:
    """Produces the final Talent DNA JSON profile (Module 4)."""
    raw_scores = infer_capability_scores(resume_text)
    dna = {cap: data["score"] for cap, data in raw_scores.items()}
    return {
        "name": candidate_name,
        "dna": dna,
        "details": raw_scores,
    }
