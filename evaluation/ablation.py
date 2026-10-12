"""Ablation: chunk size x reranker on/off.

Default mode is retrieval-only (no LLM cost): recall@top_n and MRR on the policy questions.
`--e2e` additionally runs the full workflow evaluation for every configuration.

    python -m evaluation.ablation [--sizes 500 1000 1500 2000] [--e2e]
"""
import argparse
import json

from etihad_bot.config import ROOT
from etihad_bot.rag.loaders import load_pdf
from etihad_bot.rag.pipeline import RAGPipeline
from evaluation import metrics
from evaluation.run_eval import DATASET, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    ap.add_argument("--overlap-ratio", type=float, default=0.2)
    ap.add_argument("--e2e", action="store_true")
    a = ap.parse_args()

    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    policy = [c for c in cases if c["type"] == "policy"]
    from etihad_bot.config import get_settings
    docs = load_pdf(get_settings().policy_pdf)

    results = []
    for size in a.sizes:
        overlap = int(size * a.overlap_ratio)
        for rerank in (False, True):
            pipe = RAGPipeline(chunk_size=size, chunk_overlap=overlap, use_reranker=rerank)
            if pipe.count() == 0:
                pipe.ingest(docs)
            rec, mrr = [], []
            for c in policy:
                texts = [ch.text for ch in pipe.retrieve(c["question"])]
                rec.append(metrics.keyword_recall(texts, c["evidence_keywords"]))
                mrr.append(metrics.reciprocal_rank(texts, c["evidence_keywords"]))
            row = {"chunk_size": size, "overlap": overlap, "reranker": rerank,
                   "recall@k": round(sum(rec) / len(rec), 3), "mrr": round(sum(mrr) / len(mrr), 3)}
            if a.e2e:
                summary, _ = run(pipe, [c for c in cases if not c.get("requires_flight_api")])
                row.update({k: v for k, v in summary.items() if k != "n_cases"})
            results.append(row)
            print(row)

    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    cols = list(results[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c)) for c in cols) + " |" for r in results]
    (out / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
