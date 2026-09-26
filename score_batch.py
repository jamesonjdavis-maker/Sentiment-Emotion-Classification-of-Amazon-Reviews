"""Score a batch of reviews against the star rating (Steps 2, 5, 6).

Truth rules (the model never sees the rating; it is joined back only afterwards):
    2-class: rating >= 4 -> POSITIVE, else NEGATIVE
    3-class: 4–5 -> POSITIVE, 3 -> NEUTRAL, 1–2 -> NEGATIVE

Run:
    python score_batch.py                               # Step 2: random 100, 2-class, prompt v1
    python score_batch.py --balanced                    # Step 2: 50/50 balanced
    python score_batch.py --emotion [--balanced]        # Step 5: prompt v2 adds primary emotion
    python score_batch.py --classes 3 --emotion --balanced --n 150   # Step 6: ~50 per class

Outputs (results/):
    batch_<tag>.csv          every row: review, truth, prediction(s), raw reply, correct?
    batch_<tag>_wrong.csv    only the disagreements
    batch_<tag>_metrics.json headline numbers + run settings
Re-running resumes: rows already scored in batch_<tag>.csv are not re-sent.
"""
import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from data_loader import DATA_DIR, RAW_PATH, download, load_reviews
from llm_client import MODEL, SETTINGS
from sentiment import LABELS_2, LABELS_3, classify

RESULTS = Path(__file__).parent / "results"
TRUTH_RULES = {2: "rating >= 4 -> POSITIVE, else NEGATIVE",
               3: "4–5 -> POSITIVE, 3 -> NEUTRAL, 1–2 -> NEGATIVE"}


def truth_from_rating(rating: int, k: int) -> str:
    if rating >= 4:
        return "POSITIVE"
    if k == 3 and rating == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def get_reviews() -> pd.DataFrame:
    pq = DATA_DIR / "reviews.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    if not RAW_PATH.exists():
        download()
    return load_reviews()


def pick_batch(df: pd.DataFrame, n: int, seed: int, balanced: bool, labels: list) -> pd.DataFrame:
    """Random: n rows from the whole file. Balanced: n // k rows from each class,
    drawn from the whole file with a fixed seed (same set every run)."""
    df = df.assign(truth=df["rating"].apply(truth_from_rating, k=len(labels)))
    if not balanced:
        return df.sample(n, random_state=seed)
    per = n // len(labels)
    return pd.concat([df[df.truth == lab].sample(per, random_state=seed) for lab in labels])


def score(batch: pd.DataFrame, out_csv: Path, labels: list, with_emotion: bool) -> pd.DataFrame:
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    done_ids = set(done["review_id"]) if len(done) else set()
    todo = batch[~batch["review_id"].isin(done_ids)]
    print(f"{len(done_ids)} already scored, {len(todo)} to go")

    new_rows = []
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            # Only title + text go to the model.
            out = classify(r.title, r.text, labels, with_emotion)
        except Exception as e:  # network / rate-limit: record and move on
            out = {"pred": "ERROR", "llm_emotion": "", "raw_output": repr(e)}
        new_rows.append({"review_id": r.review_id, **out})
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
            _save(batch, done, new_rows, out_csv)   # checkpoint
    return _save(batch, done, new_rows, out_csv)


PRED_COLS = ["review_id", "pred", "llm_emotion", "raw_output"]


def _save(batch, done, new_rows, out_csv) -> pd.DataFrame:
    prev = done.reindex(columns=PRED_COLS) if len(done) else None
    preds = pd.concat([prev, pd.DataFrame(new_rows, columns=PRED_COLS)], ignore_index=True)
    cols = ["review_id", "rating", "truth", "title", "text", "text_len",
            "verified_purchase", "helpful_vote", "date"]
    out = batch[cols].merge(preds, on="review_id", how="inner")
    out["llm_emotion"] = out["llm_emotion"].fillna("")
    out["correct"] = out["pred"] == out["truth"]
    out.to_csv(out_csv, index=False)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """95% confidence interval for a proportion — matters at n=100."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def report(res: pd.DataFrame, labels: list) -> dict:
    valid = res[res.pred.isin(labels)]
    n, k = len(valid), int(valid.correct.sum())
    acc = k / n if n else float("nan")
    lo, hi = wilson(k, n)
    truth_share = valid.truth.value_counts(normalize=True)
    majority_label = truth_share.idxmax()
    majority = float(truth_share.max())          # "always say the majority class" baseline

    cm = pd.crosstab(valid.truth, valid.pred, rownames=["truth"], colnames=["pred"]) \
           .reindex(index=labels, columns=labels, fill_value=0)

    per_class = {}
    for lab in labels:
        tp = int(cm.loc[lab, lab])
        support = int(cm.loc[lab].sum())         # truly this class
        predicted = int(cm[lab].sum())           # model said this class
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
        per_class[lab] = {"support": support, "predicted": predicted, "correct": tp,
                          "recall": recall, "precision": precision, "f1": f1,
                          "recall_ci95": wilson(tp, support)}
    # Average only over classes that actually occur in the sample; an absent class
    # (e.g. no 3-star reviews in a random 100) is flagged rather than turning the mean into NaN.
    present = [l for l in labels if per_class[l]["support"] > 0]
    absent = [l for l in labels if l not in present]
    bal_acc = sum(per_class[l]["recall"] for l in present) / len(present)
    f1s = [per_class[l]["f1"] for l in present if not math.isnan(per_class[l]["f1"])]
    macro_f1 = sum(f1s) / len(f1s) if f1s else float("nan")

    by_star = valid.groupby("rating").agg(n=("correct", "size"), agree=("correct", "mean"))
    star_pred = pd.crosstab(valid.rating, valid.pred).reindex(columns=labels, fill_value=0)
    len_bucket = pd.cut(valid.text_len, [-1, 5, 20, 60, 10_000],
                        labels=["≤5 words", "6–20", "21–60", ">60"])
    by_len = valid.groupby(len_bucket, observed=True).agg(n=("correct", "size"),
                                                          agree=("correct", "mean"))

    print("\n================ RESULTS ================")
    print(f"Scored: {len(res)}   usable: {n}   unparseable/error: {len(res) - n}")
    print("Truth mix: " + " / ".join(f"{(valid.truth == l).sum()} {l}" for l in labels))
    print(f"\nAgreement with rating (accuracy): {acc:.1%}   95% CI {lo:.1%}–{hi:.1%}")
    print(f"'Always {majority_label}' baseline:  {majority:.1%}   <- accuracy must beat this")
    print(f"Balanced accuracy (mean recall):  {bal_acc:.1%}   <- skew-proof headline"
          + (f"  (over present classes; ABSENT from sample: {', '.join(absent)})" if absent else ""))
    print(f"Macro F1:                         {macro_f1:.2f}")
    print("\nConfusion matrix (rows = rating says, cols = model says):")
    print(cm.to_string())
    print("\nPer class:")
    for lab, m in per_class.items():
        rlo, rhi = m["recall_ci95"]
        print(f"  {lab:<8} n={m['support']:<3} recall={m['recall']:.1%} "
              f"(CI {rlo:.0%}–{rhi:.0%})  precision={m['precision']:.1%}  F1={m['f1']:.2f}")
    print("\nModel label by star rating:")
    print(star_pred.to_string())
    print("\nAgreement by review length:")
    print(by_len.to_string(float_format=lambda x: f"{x:.1%}" if x <= 1 else f"{x:.0f}"))

    wrong = res[~res.correct]
    print(f"\nDisagreements ({len(wrong)}):")
    for r in wrong.itertuples():
        print(f"  #{r.review_id} {r.rating}★ truth={r.truth} pred={r.pred} | "
              f"{r.title!r} — {str(r.text)[:110]!r}")

    return {"n_scored": len(res), "n_usable": n, "accuracy": acc, "accuracy_ci95": [lo, hi],
            "majority_label": majority_label, "majority_baseline": majority,
            "always_positive_baseline": float(truth_share.get("POSITIVE", 0.0)),
            "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "classes_absent": absent,
            "confusion_matrix": cm.to_dict(), "per_class": per_class,
            "by_star": by_star.reset_index().to_dict("records"),
            "star_by_pred": {int(s): row.to_dict() for s, row in star_pred.iterrows()},
            "by_length": by_len.reset_index().astype({"text_len": str}).to_dict("records")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--classes", type=int, choices=[2, 3], default=2)
    ap.add_argument("--emotion", action="store_true", help="prompt also returns the primary emotion")
    args = ap.parse_args()

    labels = LABELS_3 if args.classes == 3 else LABELS_2
    prompt_version = "v3" if args.classes == 3 else ("v2" if args.emotion else "v1")
    RESULTS.mkdir(exist_ok=True)
    tag = (f"{args.n}{'_balanced' if args.balanced else ''}_s{args.seed}"
           f"{'_c3' if args.classes == 3 else ''}{'_emo' if args.emotion else ''}")
    out_csv = RESULTS / f"batch_{tag}.csv"

    batch = pick_batch(get_reviews(), args.n, args.seed, args.balanced, labels)
    res = score(batch, out_csv, labels, args.emotion)
    metrics = report(res, labels)
    metrics["run"] = {"run_id": tag, "labels": labels, "n": args.n, "seed": args.seed,
                      "sampling": "balanced" if args.balanced else "random",
                      "truth_rule": TRUTH_RULES[args.classes], "prompt_version": prompt_version,
                      "with_emotion": args.emotion, "model": MODEL, **SETTINGS,
                      "scored_on": str(date.today())}

    res[~res.correct].to_csv(RESULTS / f"batch_{tag}_wrong.csv", index=False)
    (RESULTS / f"batch_{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved {out_csv}, *_wrong.csv, *_metrics.json")
