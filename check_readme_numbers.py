"""'Show your work' audit: every figure the README quotes is recomputed from the
saved output and must appear in README.md exactly as formatted. Any number in the
README not covered by a check is listed for manual review.

Run:  python check_readme_numbers.py   ->  results/readme_number_check.txt
"""
import ast
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
R = ROOT / "results"
readme = (ROOT / "README.md").read_text()
M = lambda tag: json.loads((R / f"batch_{tag}_metrics.json").read_text())
p1 = lambda x: f"{x * 100:.1f}%"
p0 = lambda x: f"{x * 100:.0f}%"
checks = []          # (what, string that must appear in README)


def need(what, s):
    checks.append((what, str(s)))


# ── data ──
ds = json.loads((R / "dataset_summary.json").read_text())
N, rc = ds["n_reviews"], {int(k): v for k, v in ds["rating_counts"].items()}
need("total reviews", f"{N:,}")
for s in range(1, 6):
    need(f"{s}★ count", f"{rc[s]:,}")
    need(f"{s}★ share", p1(rc[s] / N))
need("4–5★ share", p1((rc[4] + rc[5]) / N))
sup = (R / "report_supporting_numbers.txt").read_text()
d0, d1 = re.search(r"date range: (\S+) -> (\S+)", sup).groups()
need("first date", d0); need("last date", d1)
need("empty text", re.search(r"empty text: (\d+)", sup).group(1) + " reviews have empty text")
need("median words", "median " + re.search(r"median words in text: (\d+)", sup).group(1) + " words")

# ── runs table ──
for tag, label in [("100_s42_emo", "2-class random"), ("100_balanced_s42_emo", "2-class balanced"),
                   ("100_s42_c3_emo", "3-class random"), ("150_balanced_s42_c3_emo", "3-class balanced")]:
    m = M(tag)
    lo, hi = m["accuracy_ci95"]
    need(f"{label} accuracy+CI", f"**{p1(m['accuracy'])}** ({p1(lo)[:-1]}–{p1(hi)})")
    need(f"{label} baseline", p1(m["majority_baseline"]))
    need(f"{label} balanced acc", p1(m["balanced_accuracy"]))

# ── Q1 ──
m = M("100_s42_emo")
need("random: positive count", f"{m['per_class']['POSITIVE']['support']} positive")
need("random: negative count", f"{m['per_class']['NEGATIVE']['support']} negative")
nr = m["per_class"]["NEGATIVE"]
need("random: negative recall", p1(nr["recall"]))
need("random: negative recall CI", f"{p0(nr['recall_ci95'][0])[:-1]}–{p0(nr['recall_ci95'][1])}")
need("random: lift over baseline", f"{(m['accuracy'] - m['majority_baseline']) * 100:.0f} points")
mb = M("100_balanced_s42_emo")
need("balanced 2-class negatives caught", f"{mb['per_class']['NEGATIVE']['correct']}/{mb['per_class']['NEGATIVE']['support']}")
m3 = M("100_s42_c3_emo")
need("random 3-class false neutrals", f"{m3['per_class']['NEUTRAL']['predicted']} positive reviews")
neu_by_star = m3["star_by_pred"]
need("…of which 5★", f"{neu_by_star['5']['NEUTRAL']} five-star")
need("…of which 4★", f"{neu_by_star['4']['NEUTRAL']} four-star")
v12 = re.search(r"total: (\d+) of (\d+) changed", sup).groups()
need("prompt v1→v2 changes", f"{v12[0]} of {v12[1]}")

# ── Q2: balanced 3-class matrix ──
m = M("150_balanced_s42_c3_emo")
cm, pc = m["confusion_matrix"], m["per_class"]
for t in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
    row = " | ".join((f"**{cm[p][t]}**" if p == t else str(cm[p][t])) for p in ["POSITIVE", "NEUTRAL", "NEGATIVE"])
    need(f"matrix row {t}", f"| **{t.capitalize()}** | {row} | {pc[t]['support']} |")
need("neutral→negative", f"{cm['NEGATIVE']['NEUTRAL']} of {pc['NEUTRAL']['support']} three-star reviews ({p0(cm['NEGATIVE']['NEUTRAL'] / 50)})")
need("negative→neutral", f"only {cm['NEUTRAL']['NEGATIVE']} of {pc['NEGATIVE']['support']} negative reviews ({p0(cm['NEUTRAL']['NEGATIVE'] / 50)})")
need("negative predicted", f"{pc['NEGATIVE']['predicted']} labels vs. {pc['NEGATIVE']['support']} true negatives")
need("negative precision", f"**{p1(pc['NEGATIVE']['precision'])}**")
need("negative recall", f"**{p1(pc['NEGATIVE']['recall'])}**")
need("neutral predicted", f"{pc['NEUTRAL']['predicted']} labels vs. {pc['NEUTRAL']['support']}")
need("positive recall", f"**{p1(pc['POSITIVE']['recall'])}**")
nlo, nhi = pc["NEUTRAL"]["recall_ci95"]
need("neutral recall+CI", f"**{p1(pc['NEUTRAL']['recall'])}** ({pc['NEUTRAL']['correct']}/50; 95% range {p0(nlo)[:-1]}–{p0(nhi)})")
need("positive→neutral misses", f"its {cm['NEUTRAL']['POSITIVE']} misses")
mb2 = pd.read_csv(R / "batch_100_balanced_s42_emo.csv")
need("3★ in 2-class balanced", f"all {int((mb2.rating == 3).sum())} three-star reviews")

# ── Q3: emotions ──
e = m["emotion"]
need("exact agreement", f"{p1(e['agreement'])} exact")
need("lenient agreement", p1(e["agreement_lenient"]))
for k in ["anger", "joy", "trust", "disgust"]:
    need(f"LLM {k}", f"{k} {e['llm_counts'][k]}")
for k in ["joy", "anticipation", "trust"]:
    need(f"word list {k}", f"{k} {e['lex_counts'][k]}")
need("word list none", f"**none {e['n_lexicon_none']}**")
need("word list tie", f"**tie {e['n_lexicon_unbreakable_tie']}**")
bt = e["by_truth_class"]
need("positive exact", f"{p0(bt['POSITIVE']['agree'])} for positive")
need("neutral exact", f"**{p0(bt['NEUTRAL']['agree'])} for neutral")
need("negative exact", f"{p0(bt['NEGATIVE']['agree'])} for negative**")
anger = re.search(r"LLM says anger (\d+), word list says anger (\d+)", sup).groups()
need("negatives: LLM anger", f"*anger* {anger[0]} times")
need("negatives: word-list anger", f"*anger* twice" if anger[1] == "2" else f"*anger* {anger[1]}")
g = re.search(r"'gift' matches: (\d+) occurrences across (\d+) of (\d+)", sup).groups()
need("gift occurrences", f"matched {g[0]} times across {g[1]} of the {g[2]} reviews")
tg = re.search(r"unbreakable ties: (\d+); involving 'gift': (\d+)", sup).groups()
need("ties involving gift", f"{tg[1]} of the {tg[0]} ties")
need("gift-set ties", re.search(r"'anticipation/joy/surprise': (\d+)", sup).group(1) + " ties are exactly")
need("no-match count", f"{e['n_lexicon_none']} of 150 matched nothing")

# ── within-class star mix of the balanced runs ──
mix = re.search(r"150_balanced_s42_c3_emo: (\{.*\})", sup).group(1)
mix = ast.literal_eval(mix)  # the dict printed by report_numbers.py
need("positives: five-star", f"{mix['POSITIVE'][5]} five-star and {mix['POSITIVE'][4]} four-star")
need("negatives: one-star", f"{mix['NEGATIVE'][1]} of 50")
m2 = ast.literal_eval(re.search(r"100_balanced_s42_emo: (\{.*\})", sup).group(1))
need("2-class: 3★ among negatives", f"{m2['NEGATIVE'][3]} three-star in the 2-class run")

# ── Q2/Q3 evidence ──
nn = ast.literal_eval(re.search(r"3★ reviews labeled negative \(n=\d+\): LLM emotions (\{.*\})", sup).group(1))
need("neutral→negative emotions", f"anger {nn['anger']}, disgust {nn['disgust']}, sadness {nn['sadness']}")
xl = re.search(r"LLM emotion x model sentiment: (.*)", sup).group(1)
xd = {e.split(": ")[0]: dict((kv.rsplit(" ", 1)[0], int(kv.rsplit(" ", 1)[1])) for kv in e.split(": ")[1].split(", ")) for e in xl.split("; ")}
need("joy only positive", f"only on reviews it called positive ({xd['joy']['POSITIVE']})")
need("negative emotions", f"({xd['anger']['NEGATIVE']}, {xd['disgust']['NEGATIVE']}, {xd['sadness']['NEGATIVE']}, {xd['fear']['NEGATIVE']})")
need("trust split", f"neutral ({xd['trust']['NEUTRAL']}) and positive ({xd['trust']['POSITIVE']})")
need("anger-but-no-match count", re.search(r"matches nothing: (\d+)", sup).group(1) + " of the 50 negative reviews")
need("real example present", "#4383" in sup and "ZERO BALANCE!!!")

# ── verification + probe ──
chk = (R / "dashboard_check.txt").read_text()
need("dashboard checks", re.search(r"(\d+) checks, 0 mismatches", chk).group(1) + " checks, 0 mismatches")
g = re.findall(r"(\d+)px: (\d+) checks, 0 failures", chk)
need("geometry checks", " / ".join(n for _, n in g) + " checks")
probe = re.search(r"thinking=True: ([\d.]+)s.*reasoning (\d+).*\nthinking=False: ([\d.]+)s", sup).groups()
need("thinking tokens", f"{probe[1]} hidden tokens")
need("thinking latency", f"{probe[0]} s to {probe[2]} s")
rep = (R / "repeatability_check.txt").read_text()
need("re-run labels identical", re.search(r"sentiment labels identical: (\d+ of \d+)", rep).group(1))
need("re-run emotions identical", re.search(r"LLM emotions identical:\s+(\d+ of \d+)", rep).group(1))
need("re-run accuracy", re.search(r"accuracy on re-run: ([\d.]+%)", rep).group(1))

# ── report ──
lines, bad = [], []
for what, s in checks:
    ok = s in readme
    lines.append(f"{'OK     ' if ok else 'MISSING'} {what}: {s!r}")
    if not ok:
        bad.append(what)
covered = set(re.findall(r"\d[\d,.]*", " ".join(s for _, s in checks)))
prose = re.sub(r"```.*?```|<!--.*?-->|\(screenshots/[^)]*\)|https?://\S+|`[^`]*`", " ", readme, flags=re.S)
uncovered = sorted({n.rstrip(".,") for n in re.findall(r"\d[\d,.]*%?", prose)} - covered - {c + "%" for c in covered},
                   key=lambda x: (len(x), x))
summary = f"{len(checks)} README figures recomputed from saved output: {len(checks) - len(bad)} match, {len(bad)} missing"
out = [summary, *lines, "", "Numbers in the README not covered by a check (review by hand):", " ".join(uncovered)]
(R / "readme_number_check.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
