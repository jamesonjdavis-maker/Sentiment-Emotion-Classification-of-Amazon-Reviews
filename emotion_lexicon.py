"""Step 5 (part 2) — word-list emotion, no model calls.

Scores each review's words against the NRC Emotion Lexicon (EmoLex v0.92,
Mohammad & Turney 2013), adds up the hits per emotion, and takes the highest.
Runs over the saved predictions in results/batch_*_emo.csv and adds columns:
    lex_emotion  highest-scoring emotion ("none" if no word matched)
    lex_scores   per-emotion hit counts (JSON)
    lex_words    which words matched, e.g. "gift:anticipation/joy/surprise"
    lex_top      all emotions sharing the top score, e.g. "anticipation/joy/surprise"
    lex_tied     True if two or more emotions shared the top score
and writes an "emotion" block into the run's metrics JSON comparing it to the
LLM's emotion.

Decisions (the brief leaves these open):
  * Title + text are scored together — the same input the LLM sees.
  * Every occurrence counts (a word repeated three times scores three times).
  * No negation handling or stemming: the lexicon is used as published.
  * Ties are broken by whichever tied emotion is triggered EARLIEST in the review.
    If the SAME word triggered all the tied emotions (e.g. "gift" alone ->
    anticipation/joy/surprise), there is no honest way to pick one, so the answer
    is "tie" (a first version silently fell back to alphabetical order, which
    handed ~20% of reviews to "anticipation"). Agreement is reported strictly
    and leniently (LLM emotion is among the tied top set).
  * Zero hits -> "none" (kept as its own category, never forced into an emotion).

The lexicon is downloaded from the author's site on first run and kept in data/
(which is git-ignored — its terms ask that it not be redistributed).

Run:  python emotion_lexicon.py
"""
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from sentiment import EMOTIONS

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
LEX_PATH = ROOT / "data" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
LEX_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
LEX_MEMBER = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
LEX_SHA256 = "02c661544f4f12ae0c14f9576a10959e8d39a151bb091e455a71a08dcaa2535a"

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def ensure_lexicon() -> Path:
    if not LEX_PATH.exists():
        print(f"Downloading {LEX_URL} ...")
        with urllib.request.urlopen(LEX_URL) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        LEX_PATH.parent.mkdir(exist_ok=True)
        LEX_PATH.write_bytes(zf.read(LEX_MEMBER))   # extract only the one text file
    digest = hashlib.sha256(LEX_PATH.read_bytes()).hexdigest()
    if digest != LEX_SHA256:
        raise SystemExit(f"Lexicon checksum mismatch ({digest}); refusing to use it.")
    return LEX_PATH


def load_lexicon() -> dict[str, list[str]]:
    """word -> list of the 8 emotions it is linked to (positive/negative dropped)."""
    lex: dict[str, list[str]] = {}
    for line in ensure_lexicon().read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3 and parts[2] == "1" and parts[1] in EMOTIONS:
            lex.setdefault(parts[0], []).append(parts[1])
    return lex


def tokenize(title: str, text: str) -> list[str]:
    s = f"{title or ''} {text or ''}".replace("<br />", " ").lower()
    return TOKEN_RE.findall(s)


def score_review(title: str, text: str, lex: dict) -> dict:
    scores = Counter({e: 0 for e in EMOTIONS})
    first_seen: dict[str, int] = {}
    hits = []
    for pos, tok in enumerate(tokenize(title, text)):
        emos = lex.get(tok)
        if not emos:
            continue
        hits.append(f"{tok}:{'/'.join(emos)}")
        for e in emos:
            scores[e] += 1
            first_seen.setdefault(e, pos)
    top = max(scores.values())
    if top == 0:
        return {"lex_emotion": "none", "lex_top": "", "lex_scores": json.dumps(dict(scores)),
                "lex_words": "", "lex_tied": False}
    tied = [e for e in EMOTIONS if scores[e] == top]
    earliest = min(first_seen[e] for e in tied)
    firsts = [e for e in tied if first_seen[e] == earliest]
    winner = firsts[0] if len(firsts) == 1 else "tie"
    return {"lex_emotion": winner, "lex_top": "/".join(tied), "lex_scores": json.dumps(dict(scores)),
            "lex_words": " ".join(hits), "lex_tied": len(tied) > 1}


def compare(df: pd.DataFrame) -> dict:
    llm_ok = df[df.llm_emotion.isin(EMOTIONS)]
    both = llm_ok[llm_ok.lex_emotion != "none"]
    agree_all = float((llm_ok.llm_emotion == llm_ok.lex_emotion).mean())
    agree_hits = float((both.llm_emotion == both.lex_emotion).mean()) if len(both) else float("nan")
    in_top = llm_ok.apply(lambda r: r.llm_emotion in str(r.lex_top).split("/"), axis=1)
    cats = EMOTIONS + ["tie", "none"]
    xtab = pd.crosstab(llm_ok.llm_emotion, llm_ok.lex_emotion) \
             .reindex(index=EMOTIONS, columns=cats, fill_value=0)
    by_truth = {t: {"n": int(len(g)), "agree": float((g.llm_emotion == g.lex_emotion).mean()),
                    "agree_lenient": float(in_top[g.index].mean()),
                    "llm_top": g.llm_emotion.value_counts().idxmax(),
                    "lex_top": g.lex_emotion.value_counts().idxmax()}
                for t, g in llm_ok.groupby("truth")}
    words = Counter(w.split(":")[0] for ws in df.lex_words.fillna("") for w in ws.split())
    return {
        "n": int(len(llm_ok)),
        "agreement": agree_all,
        "agreement_when_lexicon_hit": agree_hits,
        "agreement_lenient": float(in_top.mean()),
        "n_lexicon_none": int((df.lex_emotion == "none").sum()),
        "n_lexicon_unbreakable_tie": int((df.lex_emotion == "tie").sum()),
        "n_lexicon_tied": int(df.lex_tied.sum()),
        "llm_counts": {e: int((llm_ok.llm_emotion == e).sum()) for e in EMOTIONS},
        "lex_counts": {e: int((df.lex_emotion == e).sum()) for e in cats},
        "crosstab_llm_by_lex": {e: {c: int(xtab.loc[e, c]) for c in cats} for e in EMOTIONS},
        "by_truth_class": by_truth,
        "top_lexicon_words": words.most_common(15),
    }


def main() -> None:
    lex = load_lexicon()
    print(f"Lexicon: {len(lex):,} words linked to at least one of the 8 emotions")
    for csv in sorted(RESULTS.glob("batch_*_emo.csv")):
        df = pd.read_csv(csv)
        df = df.drop(columns=[c for c in df.columns if c.startswith("lex_")])
        lexcols = pd.DataFrame([score_review(t, x, lex) for t, x in zip(df.title, df.text)])
        df = pd.concat([df, lexcols], axis=1)
        df.to_csv(csv, index=False)

        mfile = csv.with_name(csv.stem + "_metrics.json")
        metrics = json.loads(mfile.read_text())
        metrics["emotion"] = cmp = compare(df)
        mfile.write_text(json.dumps(metrics, indent=2, default=str))

        print(f"\n== {csv.name} ==")
        print(f"LLM vs word-list agreement: {cmp['agreement']:.1%} of {cmp['n']} "
              f"({cmp['agreement_when_lexicon_hit']:.1%} when the word list found any word)")
        print(f"  lenient (LLM emotion among word-list's tied top set): {cmp['agreement_lenient']:.1%}")
        print(f"Word list found no emotion words: {cmp['n_lexicon_none']}   ties: {cmp['n_lexicon_tied']} "
              f"(unbreakable -> 'tie': {cmp['n_lexicon_unbreakable_tie']})")
        for t, b in cmp["by_truth_class"].items():
            print(f"  {t:<8} n={b['n']:<3} strict {b['agree']:.0%}  lenient {b['agree_lenient']:.0%}  "
                  f"LLM mostly {b['llm_top']}, word list mostly {b['lex_top']}")
        print("LLM  :", {k: v for k, v in cmp["llm_counts"].items() if v})
        print("Lexic:", {k: v for k, v in cmp["lex_counts"].items() if v})
        print("Most-hit lexicon words:", cmp["top_lexicon_words"][:10])


if __name__ == "__main__":
    main()
