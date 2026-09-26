"""Recompute the supporting figures quoted in README.md that are not already in a
metrics JSON, and save them to results/report_supporting_numbers.txt so every
number in the report can be traced to a saved file.

Run:  python report_numbers.py            (add --probe to re-time the endpoint)
"""
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
R = ROOT / "results"
lines = []
say = lines.append
vc = lambda s: {k: int(v) for k, v in s.value_counts().items()}   # plain ints for printing

# ── data (Step 0) ──
df = pd.read_parquet(ROOT / "data" / "reviews.parquet")
say("== Data ==")
say(f"reviews: {len(df):,}")
say(f"date range: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
say(f"empty text: {(df['text'] == '').sum()}   empty title: {(df['title'] == '').sum()}")
say(f"median words in text: {df['text_len'].median():.0f}")
share = df["rating"].value_counts(normalize=True)
say(f"4–5★ share: {share[[4, 5]].sum():.1%}   3★: {share[3]:.1%}   1–2★: {share[[1, 2]].sum():.1%}")

# ── prompt v1 vs v2: did adding emotion change any sentiment label? ──
say("\n== Sentiment labels: prompt v1 vs v2 (same reviews) ==")
changed = total = 0
for tag in ["100_s42", "100_balanced_s42"]:
    a = pd.read_csv(R / f"batch_{tag}.csv")
    b = pd.read_csv(R / f"batch_{tag}_emo.csv")
    m = a.merge(b, on="review_id", suffixes=("_v1", "_v2"))
    c = int((m.pred_v1 != m.pred_v2).sum())
    changed += c
    total += len(m)
    say(f"{tag}: {c} of {len(m)} changed")
say(f"total: {changed} of {total} changed")

# ── random samples: which stars got called NEUTRAL; stars present ──
say("\n== Random 100, 3-class ==")
r = pd.read_csv(R / "batch_100_s42_c3_emo.csv")
say(f"stars present: {dict(sorted(Counter(r.rating).items()))}")
say(f"predicted NEUTRAL by star: {dict(sorted(vc(r[r.pred == 'NEUTRAL'].rating).items()))}")

say("\n== Balanced 2-class: 3★ reviews ==")
b2 = pd.read_csv(R / "batch_100_balanced_s42_emo.csv")
say(f"3★ reviews: {int((b2.rating == 3).sum())}, labeled: {vc(b2[b2.rating == 3].pred)}")

# ── balanced 3-class: emotion details ──
say("\n== Balanced 3-class: emotions ==")
d = pd.read_csv(R / "batch_150_balanced_s42_c3_emo.csv")
lw = d.lex_words.fillna("")
words = lw.apply(lambda s: {w.split(":")[0] for w in s.split()})
say(f"'gift' matches: {sum(s.count('gift:') for s in lw)} occurrences across {int(words.apply(lambda w: 'gift' in w).sum())} of {len(d)} reviews")
ties = d[d.lex_emotion == "tie"]
say(f"unbreakable ties: {len(ties)}; involving 'gift': {int(words[ties.index].apply(lambda w: 'gift' in w).sum())}")
say(f"most common tied sets: {dict(list(vc(ties.lex_top).items())[:3])}")
only_gift = lw.apply(lambda s: bool(s) and all(w.startswith("gift:") for w in s.split()))
say(f"reviews whose only lexicon match is 'gift': {int(only_gift.sum())}")
neg = d[d.truth == "NEGATIVE"]
say(f"negative reviews (n={len(neg)}): LLM says anger {int((neg.llm_emotion == 'anger').sum())}, "
    f"word list says anger {int((neg.lex_emotion == 'anger').sum())}")
pos_neu = d[(d.truth == "POSITIVE") & (d.pred == "NEUTRAL")]
say(f"positive reviews labeled neutral: {len(pos_neu)} -> stars {vc(pos_neu.rating)}")

# ── lexicon coverage of complaint words ──
say("\n== NRC lexicon: emotions linked to common complaint words ==")
lex = {}
for line in (ROOT / "data" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt").read_text().splitlines():
    p = line.strip().split("\t")
    if len(p) == 3 and p[2] == "1" and p[1] not in ("positive", "negative"):
        lex.setdefault(p[0], []).append(p[1])
for w in ["scam", "refund", "card", "amazon", "cheated", "useless", "ripoff", "fraud", "gift", "money"]:
    say(f"{w}: {'/'.join(lex.get(w, [])) or '(no emotion)'}")

# ── optional: endpoint behaviour with thinking on vs off ──
if "--probe" in sys.argv:
    from llm_client import MODEL, client
    say("\n== Endpoint probe: thinking on vs off ==")
    for on in (True, False):
        t = time.time()
        resp = client.chat.completions.create(
            model=MODEL, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": "Respond with exactly one word: POSITIVE or NEGATIVE."},
                      {"role": "user", "content": "Title: Great\nText: Card worked, she loved it."}],
            extra_body={"chat_template_kwargs": {"enable_thinking": on}})
        u = resp.usage
        rt = getattr(u.completion_tokens_details, "reasoning_tokens", None) if u.completion_tokens_details else None
        say(f"thinking={on}: {time.time() - t:.2f}s, completion tokens {u.completion_tokens} "
            f"(reasoning {rt}), answer {resp.choices[0].message.content.strip()!r}")

out = R / "report_supporting_numbers.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved {out}")
