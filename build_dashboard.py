"""Dashboard generator — bakes every saved run in results/ into one
self-contained, offline HTML file (no CDN, no server).

Run:  python build_dashboard.py   ->  dashboard.html

All numbers on the page come from the saved results/*_metrics.json and
results/batch_*.csv files; nothing is recomputed differently here.
"""
import json
import math
from pathlib import Path

import pandas as pd

from sentiment import build_system_prompt

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT / "dashboard.html"
TEMPLATE = ROOT / "dashboard_template.html"

ROW_COLS = ["review_id", "rating", "truth", "pred", "correct", "title", "text",
            "text_len", "verified_purchase", "helpful_vote", "date", "raw_output",
            "llm_emotion", "lex_emotion", "lex_top", "lex_words"]


def run_title(meta: dict) -> str:
    k = len(meta["labels"])
    if meta["sampling"] == "balanced":
        per = meta["n"] // k
        return f"Balanced {per} per class · {k}-class"
    return f"Random {meta['n']} · {k}-class"


def load_runs() -> list[dict]:
    """One run per (classes, sampling, n, seed); when a run exists both with and
    without the emotion prompt, the newer emotion version is shown (its sentiment
    labels were checked to be identical — see results/step5_*)."""
    runs = []
    for mfile in sorted(RESULTS.glob("batch_*_metrics.json")):
        metrics = json.loads(mfile.read_text())
        meta = metrics.get("run")
        if not meta:
            continue
        rows = pd.read_csv(RESULTS / f"batch_{meta['run_id']}.csv")
        rows = rows[[c for c in ROW_COLS if c in rows.columns]].copy()
        rows["date"] = rows["date"].astype(str).str[:10]
        rows = rows.fillna("")
        meta.setdefault("prompt_version", "v1")
        meta.setdefault("with_emotion", False)
        runs.append({"id": meta["run_id"], "title": run_title(meta), "meta": meta,
                     "metrics": metrics, "rows": rows.to_dict("records")})
    best = {}
    for r in runs:
        m = r["meta"]
        key = (len(m["labels"]), m["sampling"], m["n"], m["seed"])
        if key not in best or (m["with_emotion"], m["prompt_version"]) > \
                (best[key]["meta"]["with_emotion"], best[key]["meta"]["prompt_version"]):
            best[key] = r
    runs = list(best.values())
    # 2-class before 3-class; within each, random (the "lopsided" view) before balanced
    runs.sort(key=lambda r: (len(r["meta"]["labels"]), r["meta"]["sampling"] != "random"))
    return runs


def dataset_summary() -> dict:
    """Star distribution of the full file. Computed from data/ when present and
    saved to results/ (committed), so the dashboard rebuilds without the raw data."""
    saved = RESULTS / "dataset_summary.json"
    pq = ROOT / "data" / "reviews.parquet"
    if pq.exists():
        counts = pd.read_parquet(pq, columns=["rating"])["rating"].value_counts().sort_index()
        summary = {"n_reviews": int(counts.sum()),
                   "rating_counts": {int(k): int(v) for k, v in counts.items()}}
        saved.write_text(json.dumps(summary, indent=2))
    return json.loads(saved.read_text())


def clean(o):
    """NaN/inf are not valid JSON (an absent class has NaN recall) -> null."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def build() -> None:
    runs = load_runs()
    prompts = {r["id"]: build_system_prompt(r["meta"]["labels"], r["meta"]["with_emotion"]) for r in runs}
    payload = {"runs": runs, "dataset": dataset_summary(), "prompts": prompts,
               "default_run": next((r["id"] for r in runs
                                    if len(r["meta"]["labels"]) == 3 and r["meta"]["sampling"] == "balanced"),
                                   runs[-1]["id"])}
    data = json.dumps(clean(payload), default=str, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    OUT.write_text(html)
    print(f"Wrote {OUT} with {len(payload['runs'])} runs "
          f"({', '.join(r['title'] for r in payload['runs'])})")


if __name__ == "__main__":
    build()
