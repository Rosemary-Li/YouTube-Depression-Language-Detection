import json
import re
import pathlib

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT        = pathlib.Path(__file__).resolve().parent.parent
INPUT_FILE  = ROOT / "data" / "raw" / "videos.json"
OUTPUT_FILE = ROOT / "data" / "processed" / "video_features.csv"

CATEGORIES = {
    "mental_health",
    "fitness_wellness",
    "music_entertainment",
    "news_current_events",
    "personal_vlog",
}

vader = SentimentIntensityAnalyzer()

_URL_RE  = re.compile(r"https?://\S+")
_TAG_RE  = re.compile(r"#\w+")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def clean_description(text: str) -> str:
    if not text:
        return ""
    text = _URL_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = _TIME_RE.sub("", text)
    return text.strip()


def text_stats(text: str) -> dict:
    tokens = text.lower().split()
    wc = len(tokens)
    return {
        "word_count":        wc,
        "lexical_diversity": round(len(set(tokens)) / wc, 4) if wc > 0 else 0.0,
    }


def extract_features(video: dict) -> dict:
    title    = video.get("title", "") or ""
    desc_raw = video.get("description", "") or ""
    desc     = clean_description(desc_raw)

    blob  = TextBlob(desc)
    stats = text_stats(desc)

    return {
        "video_id":        video["video_id"],
        "category":        video["category"],
        "category_label":  video.get("category_label", ""),
        "title":           title,
        "channel":         video.get("channel", ""),
        "view_count":      int(video.get("view_count", 0)),
        "like_count":      int(video.get("like_count", 0)),
        "comment_count":   int(video.get("comment_count", 0)),

        "title_sentiment":   round(vader.polarity_scores(title)["compound"], 4),
        "desc_vader":        round(vader.polarity_scores(desc)["compound"], 4),
        "desc_polarity":     round(blob.sentiment.polarity, 4),
        "desc_subjectivity": round(blob.sentiment.subjectivity, 4),

        "desc_word_count":        stats["word_count"],
        "desc_lexical_diversity": stats["lexical_diversity"],
    }


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        videos = json.load(f)

    rows, skipped = [], 0
    for v in videos:
        if v.get("category") not in CATEGORIES:
            skipped += 1
            continue
        rows.append(extract_features(v))

    df = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} videos → {OUTPUT_FILE}  (skipped {skipped})")
    print()
    print(df.groupby("category")[
        ["title_sentiment", "desc_vader", "desc_polarity",
         "desc_subjectivity", "desc_word_count", "desc_lexical_diversity"]
    ].mean().round(3).to_string())
    print()
    print("Per-category counts:")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
