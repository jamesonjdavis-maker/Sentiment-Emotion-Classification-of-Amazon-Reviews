"""Step 0 — download (once) and read the Amazon Reviews '23 Gift Cards file.

Run:  python data_loader.py
Prints row count, fields, rating distribution and a few sample reviews so you
can confirm the file is readable before building anything on top of it.
"""
import gzip
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
       "review_categories/Gift_Cards.jsonl.gz")
DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "Gift_Cards.jsonl.gz"   # ~12 MB

# Metadata worth keeping: rating (ground truth for Step 3+), text fields for the
# prompt, and a few columns useful for slicing results on the dashboard.
KEEP = ["rating", "title", "text", "verified_purchase", "helpful_vote",
        "timestamp", "asin", "parent_asin", "user_id"]


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if force or not RAW_PATH.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, RAW_PATH)
    return RAW_PATH


def load_reviews(path: Path = RAW_PATH) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            row = {k: r.get(k) for k in KEEP}
            row["n_images"] = len(r.get("images") or [])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["rating"] = df["rating"].astype(int)
    df["title"] = df["title"].fillna("").str.strip()
    df["text"] = df["text"].fillna("").str.strip()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["text_len"] = df["text"].str.split().str.len()
    df = df.reset_index(names="review_id")
    return df


if __name__ == "__main__":
    download()
    df = load_reviews()
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}")
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())
    print(f"\nVerified purchase share: {df['verified_purchase'].mean():.1%}")
    print(f"Empty text: {(df['text'] == '').sum()}   Empty title: {(df['title'] == '').sum()}")
    print(f"Median words in text: {df['text_len'].median():.0f}")
    print("\nSample reviews:")
    for _, r in df.sample(5, random_state=1).iterrows():
        print(f"  [{r.rating}★] {r.title!r} — {r.text[:120]!r}")
    df.to_parquet(DATA_DIR / "reviews.parquet", index=False)
    print(f"\nSaved {DATA_DIR / 'reviews.parquet'}")
