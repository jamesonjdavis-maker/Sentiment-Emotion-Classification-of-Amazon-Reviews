"""Audit for two standing rules, saved to results/repeatability_check.txt.

1. The model never sees the rating: every request sent to the endpoint for the
   balanced 3-class run is captured and checked — the user message must be
   exactly the title + text template, and no prompt may mention stars/ratings.
2. Results are repeatable: every saved run's review selection is re-drawn with
   its seed and compared to the saved IDs; the balanced 3-class run is then
   re-scored live and every label/emotion compared with the saved output.

Run:  python check_repeatability.py        (~150 model calls, about a minute)
"""
import json
import re
from pathlib import Path

import pandas as pd

import llm_client
import sentiment
from score_batch import get_reviews, pick_batch
from sentiment import LABELS_2, LABELS_3, build_user_message

R = Path(__file__).parent / "results"
lines = []
say = lines.append

reviews = get_reviews()

# ── 2a. sample selection is fixed by the seed ──
say("== Sample selection (re-drawn with the saved seed) ==")
for mfile in sorted(R.glob("batch_*_metrics.json")):
    meta = json.loads(mfile.read_text()).get("run")
    if not meta:
        continue
    labels = LABELS_3 if len(meta["labels"]) == 3 else LABELS_2
    redrawn = pick_batch(reviews, meta["n"], meta["seed"], meta["sampling"] == "balanced", labels)
    saved = pd.read_csv(R / f"batch_{meta['run_id']}.csv")
    same = sorted(redrawn.review_id) == sorted(saved.review_id)
    truth_same = (redrawn.set_index("review_id").truth.sort_index()
                  == saved.set_index("review_id").truth.sort_index()).all()
    say(f"{meta['run_id']:<28} {'IDENTICAL' if same else 'DIFFERENT'} review IDs; correct answers "
        f"{'identical' if truth_same else 'DIFFER'}")

# ── 1 + 2b. capture every request while re-scoring the balanced 3-class run ──
captured = []
_real_chat = llm_client.chat


def spy(system, user, **kw):
    captured.append((system, user))
    return _real_chat(system, user, **kw)


sentiment.chat = spy  # sentiment.py imported chat by name

saved = pd.read_csv(R / "batch_150_balanced_s42_c3_emo.csv")
redo = [sentiment.classify(t, x, LABELS_3, True) for t, x in zip(saved.title, saved.text)]
redo = pd.DataFrame(redo)

say("\n== The model never sees the rating (balanced 3-class run, every request captured) ==")
exact_template = all(u == build_user_message(t, x) for (_, u), t, x in zip(captured, saved.title, saved.text))
star_words = re.compile(r"\b(rating|rated|stars?|★)\b", re.IGNORECASE)
systems = {s for s, _ in captured}
sys_mentions = [m.group(0) for s in systems for m in star_words.finditer(s)]
say(f"requests captured: {len(captured)}")
say(f"user message is exactly the title+text template in every request: {exact_template}")
say(f"distinct system prompts: {len(systems)}; words about stars/ratings in them: {sys_mentions or 'none'}")
say("(review text itself may mention stars — e.g. 'Five stars' titles — that is the reviewer's own words)")

say("\n== Re-scoring the balanced 3-class run live ==")
same_pred = int((redo.pred.values == saved.pred.values).sum())
same_emo = int((redo.llm_emotion.values == saved.llm_emotion.fillna("").values).sum())
say(f"sentiment labels identical: {same_pred} of {len(saved)}")
say(f"LLM emotions identical:     {same_emo} of {len(saved)}")
diff = saved[(redo.pred.values != saved.pred.values) | (redo.llm_emotion.values != saved.llm_emotion.fillna("").values)]
for i, r in diff.iterrows():
    say(f"  changed #{r.review_id}: {r.pred}/{r.llm_emotion} -> {redo.pred[i]}/{redo.llm_emotion[i]}  ({str(r.title)[:50]!r})")
acc_redo = (redo.pred.values == saved.truth.values).mean()
say(f"accuracy on re-run: {acc_redo:.1%} (saved: {(saved.pred == saved.truth).mean():.1%})")

out = R / "repeatability_check.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved {out}")
