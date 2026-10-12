"""End-to-end evaluation through the full LangGraph workflow.

    python -m evaluation.run_eval [--skip-flight] [--out evaluation/results/eval.json]
"""
import argparse
import json
import uuid
from pathlib import Path

from etihad_bot.config import ROOT
from etihad_bot.graph.workflow import build_graph
from etihad_bot.rag.pipeline import RAGPipeline
from etihad_bot.services import chat as chat_service
from etihad_bot.services import llm
from evaluation import metrics

DATASET = ROOT / "data" / "eval" / "eval_set.json"


def run(pipeline: RAGPipeline, cases: list[dict], with_judge: bool = True) -> tuple[dict, list[dict]]:
    app = build_graph(pipeline)  # no checkpointer: every case is independent
    rows = []
    for case in cases:
        res = chat_service.chat("eval", case["question"], uuid.uuid4().hex, app=app)
        refused, clarified = res["status"] == "refused", res["status"] == "clarification"
        row = {"id": case["id"], "behavior": case["behavior"], "status": res["status"],
               "refused": refused, "clarified": clarified, "answer": res["answer"],
               "recall": None, "mrr": None, "correctness": None, "faithfulness": None,
               "citation_accuracy": None}
        if case["type"] == "policy":
            texts = [c["text"] for c in res["chunks"]]
            row["recall"] = metrics.keyword_recall(texts, case["evidence_keywords"])
            row["mrr"] = metrics.reciprocal_rank(texts, case["evidence_keywords"])
        if case["behavior"] == "answer" and with_judge:
            if res["status"] == "answered":
                llm.request_ctx.set({"user_id": "eval", "conversation_id": res["conversation_id"]})
                row.update(metrics.judge(case["question"], case["gold_answer"], res["answer"],
                                         res["chunks"]))
            else:
                row["correctness"] = 0.0  # refused/clarified an answerable question
        rows.append(row)
    return metrics.aggregate(rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-flight", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "eval.json"))
    a = ap.parse_args()

    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if a.skip_flight:
        cases = [c for c in cases if not c.get("requires_flight_api")]
    summary, rows = run(RAGPipeline(), cases, with_judge=not a.no_judge)
    print(json.dumps(summary, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

