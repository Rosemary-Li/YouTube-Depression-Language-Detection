"""
Step 1 — Video-Level Feature Extraction
Scope: news_current_events and personal_vlog categories only
Output: video_features_KL.csv (14 columns, same schema as video_features.csv)
"""

import json
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Load data ──────────────────────────────────────────────────────────────────
with open('/Users/keithleung/Downloads/videos.json') as f:
    videos = json.load(f)

# ── Filter to target categories (exact strings from videos.json) ───────────────
TARGET_CATEGORIES = {'news_current_events', 'personal_vlog'}
videos = [v for v in videos if v['category'] in TARGET_CATEGORIES]
print(f"Videos to process: {len(videos)}")

# ── Initialise VADER once (avoids re-loading the lexicon per call) ─────────────
vader = SentimentIntensityAnalyzer()

# ── Feature extraction helpers ─────────────────────────────────────────────────

def vader_compound(text):
    """VADER compound score; returns 0.0 for empty/missing text."""
    if not text or not text.strip():
        return 0.0
    return vader.polarity_scores(text)['compound']


def textblob_polarity(text):
    """TextBlob polarity; returns 0.0 for empty/missing text."""
    if not text or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity


def textblob_subjectivity(text):
    """TextBlob subjectivity; returns 0.0 for empty/missing text."""
    if not text or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.subjectivity


def word_count(text):
    """Token count via .split(); returns 0 for empty/missing text."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def lexical_diversity(text):
    """Unique tokens / total tokens via .split(); returns 0.0 when empty."""
    if not text or not text.strip():
        return 0.0
    words = text.split()
    total = len(words)
    if total == 0:
        return 0.0
    return len(set(words)) / total


# ── Build rows ─────────────────────────────────────────────────────────────────
rows = []
for v in videos:
    title = v.get('title', '') or ''
    desc  = v.get('description', '') or ''

    row = {
        'video_id':            v['video_id'],
        'category':            v['category'],
        'category_label':      v['category_label'],
        'title':               title,
        'channel':             v.get('channel', ''),
        'view_count':          int(v.get('view_count', 0)),
        'like_count':          int(v.get('like_count', 0)),
        'comment_count':       int(v.get('comment_count', 0)),
        # sentiment features — rounded to 4 d.p. to match video_features.csv
        'title_sentiment':     round(vader_compound(title),       4),
        'desc_vader':          round(vader_compound(desc),        4),
        'desc_polarity':       round(textblob_polarity(desc),     4),
        'desc_subjectivity':   round(textblob_subjectivity(desc), 4),
        'desc_word_count':     word_count(desc),
        'desc_lexical_diversity': round(lexical_diversity(desc),  4),
    }
    rows.append(row)

# ── Assemble dataframe with exact column order ─────────────────────────────────
COLUMNS = [
    'video_id', 'category', 'category_label', 'title', 'channel',
    'view_count', 'like_count', 'comment_count',
    'title_sentiment', 'desc_vader', 'desc_polarity', 'desc_subjectivity',
    'desc_word_count', 'desc_lexical_diversity',
]
df = pd.DataFrame(rows, columns=COLUMNS)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = '/Users/keithleung/Downloads/video_features_KL.csv'
df.to_csv(out_path, index=False)

print(f"\nWrote {len(df)} rows → {out_path}")
print(df[['category', 'video_id']].groupby('category').count().rename(columns={'video_id': 'count'}))
print("\nFirst two rows:")
print(df.head(2).to_string())
