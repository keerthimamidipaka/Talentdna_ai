"""Module 8: DNA Evolution Predictor — projects score gains from targeted learning."""

LEARNING_PATHS = {
    "cloud": [("AWS Certified Solutions Architect", 15), ("Terraform in production", 8)],
    "devops": [("Kubernetes (CKA) certification", 15), ("Own a CI/CD pipeline end-to-end", 8)],
    "ai_ml": [("Ship a production ML model", 15), ("Contribute to an open-source ML library", 8)],
    "leadership": [("Lead a small project team", 12), ("Mentor a junior engineer", 8)],
    "system_design": [("Design a distributed system RFC", 12), ("Study large-scale architecture case studies", 8)],
    "frontend": [("Ship a production React/Next.js app", 12), ("Contribute to a design system", 6)],
    "backend": [("Own a production API end-to-end", 10), ("Optimize a high-traffic service", 8)],
    "communication": [("Give a public tech talk", 10), ("Write technical documentation/blog", 6)],
    "learning_agility": [("Complete a relevant certification", 10)],
    "problem_solving": [("Lead a performance optimization project", 10)],
    "collaboration": [("Work cross-functionally on a launch", 8)],
}


def predict_evolution(candidate_dna: dict, weak_threshold: int = 75) -> list:
    """For capabilities below the threshold, suggest the learning path and
    the projected score after completing it."""
    predictions = []
    for dim, score in candidate_dna.items():
        if score < weak_threshold and dim in LEARNING_PATHS:
            path = LEARNING_PATHS[dim]
            projected = min(100, score + sum(bonus for _, bonus in path))
            predictions.append({
                "capability": dim,
                "current_score": score,
                "projected_score": projected,
                "recommended_actions": [action for action, _ in path],
            })
    predictions.sort(key=lambda p: p["projected_score"] - p["current_score"], reverse=True)
    return predictions
