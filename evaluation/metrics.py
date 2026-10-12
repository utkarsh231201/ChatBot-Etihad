"""Evaluation metrics: retrieval recall/MRR, LLM-judged correctness/faithfulness/citation accuracy,
refusal accuracy and clarification rate."""
import re
from statistics import mean

from etihad_bot.services import llm


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def keyword_recall(chunk_texts: list[str], keywords: list[str]) -> float:
    """Fraction of gold evidence keywords present in the retrieved chunks (chunk-size agnostic)."""
    if not keywords:
        return 1.0
    blob = _norm(" ".join(chunk_texts))
    return sum(_norm(k) in blob for k in keywords) / len(keywords)


def reciprocal_rank(chunk_texts: list[str], keywords: list[str]) -> float:
    for i, t in enumerate(chunk_texts, 1):
        if any(_norm(k) in _norm(t) for k in keywords):
            return 1.0 / i
    return 0.0


JUDGE_SYSTEM = """You are an impartial evaluator of a RAG assistant. Score the ANSWER.
- correctness (0-1): agreement with the GOLD answer on the key facts.
- faithfulness (0-1): fraction of factual claims in the answer supported by the CONTEXT passages.
- citation_accuracy (0-1): fraction of [n] citations whose passage actually supports the sentence citing it
  (1.0 if no citations are needed).
JSON schema: {"correctness": number, "faithfulness": number, "citation_accuracy": number}"""


def judge(question: str, gold: str, answer: str, chunks: list[dict]) -> dict:
    ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    r = llm.complete_json(
        JUDGE_SYSTEM, f"QUESTION: {question}\nGOLD: {gold}\nCONTEXT:\n{ctx}\nANSWER: {answer}",
        purpose="eval_judge", max_tokens=200)
    return {k: float(r.get(k, 0.0)) for k in ("correctness", "faithfulness", "citation_accuracy")}


def aggregate(rows: list[dict]) -> dict:
    def avg(key, only=lambda r: True):
        vals = [r[key] for r in rows if r.get(key) is not None and only(r)]
        return round(mean(vals), 3) if vals else None

    scored = [r for r in rows if r["behavior"] in ("answer", "refuse")]
    return {
        "n_cases": len(rows),
        "retrieval_recall": avg("recall"),
        "mrr": avg("mrr"),
        "answer_correctness": avg("correctness"),
        "faithfulness": avg("faithfulness"),
        "citation_accuracy": avg("citation_accuracy"),
        "refusal_accuracy": round(mean(r["refused"] == (r["behavior"] == "refuse") for r in scored), 3)
        if scored else None,
        "clarification_rate": round(mean(r["clarified"] for r in rows), 3) if rows else None,
        "clarification_accuracy": round(mean(r["clarified"] == (r["behavior"] == "clarify")
                                             for r in rows), 3) if rows else None,
    }
