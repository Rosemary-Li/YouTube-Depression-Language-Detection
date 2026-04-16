# YouTube Depression Signal Detection

> Investigating whether video-level contextual framing moderates depression-indicative linguistic patterns in YouTube comments.

We analyze YouTube videos and their comments to detect depression-related linguistic signals. Rather than treating comments in isolation, we treat each comment as **contextually grounded** in its source video — and examine how video-level framing (clinical, motivational, personal story, etc.) shapes the language users produce.

This project does **not** attempt to diagnose depression. We detect *depression-indicative linguistic signals* as a proxy for self-disclosed depression-related experiences.

---

## Research Questions

1. Can linguistic signals in YouTube comments identify users who self-disclose depression-related experiences?
2. To what extent does video-level contextual framing moderate comment-level linguistic patterns?

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw API responses (videos + comments)
│   ├── processed/            # Cleaned and feature-engineered data
│   └── outputs/              # Model outputs, cluster labels, figures
│
├── notebooks/                # Exploratory analysis and visualization
│
├── src/
│   ├── collect.py            # Step 0: YouTube Data API crawler
│   ├── video_analysis.py     # Step 1: Video-level sentiment, depression score, topic
│   ├── comment_analysis.py   # Step 2: Comment embedding, clustering, linguistic features
│   ├── linking.py            # Step 3: Context-comment linking and statistical tests
│   └── modeling.py           # Step 4: Supervised classification (optional)
│
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 — Data collection | `collect.py` | YouTube API key, category list | `data/raw/videos.json`, `data/raw/comments.json` |
| 1 — Video analysis | `video_analysis.py` | `videos.json` | `data/processed/video_features.csv` |
| 2 — Comment analysis | `comment_analysis.py` | `comments.json` | `data/processed/comment_features.csv` |
| 3 — Context linking | `linking.py` | both feature CSVs | `data/outputs/heatmap.png`, `data/outputs/umap.png` |
| 4 — Modeling | `modeling.py` | both feature CSVs | `data/outputs/model_results.json` |

---

## Setup

```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

You will need a YouTube Data API v3 key. Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_key_here
```

---

## Usage

Run each step in order:

```bash
# Step 0: collect data
python src/collect.py

# Step 1: video-level analysis
python src/video_analysis.py

# Step 2: comment-level analysis
python src/comment_analysis.py

# Step 3: context-comment linking
python src/linking.py

# Step 4: modeling (optional)
python src/modeling.py
```

---

## Data Collection Strategy

- Videos are crawled across general YouTube categories and filtered by **view count only** — no depression-related keywords are used at collection time.
- Depression relevance is determined *post-hoc* in Step 1 via a depression intensity score.
- For each video: `title`, `description`, `tags`, `view_count`, `like_count`, `comment_count` are collected.
- For each video, the top-N comments sorted by likes are collected, stored with `video_id` as a foreign key.

---

## Models and Tools

| Task | Model / Tool |
|------|-------------|
| Sentiment analysis | `cardiffnlp/twitter-roberta-base-sentiment` |
| Topic classification | `facebook/bart-large-mnli` (zero-shot) |
| Comment embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Clustering | `HDBSCAN` |
| Dimensionality reduction | `UMAP` |
| Visualization | `seaborn`, `matplotlib` |

---

## Limitations

- **Comment ≠ commenter**: commenting under a depression-related video does not imply the commenter has depression.
- **Top-comment bias**: sorting by likes favors resonant, emotionally salient language.
- **Selection bias**: high view count ≠ representative sample of all mental health discourse.
- **No clinical ground truth**: all labels are proxies based on linguistic and behavioral signals.

Results should be interpreted as detecting *depression-indicative linguistic patterns*, not as clinical or diagnostic claims.

---

## Team

<!-- Add names / roles here -->

---

## License

For academic use only. Data collected via YouTube Data API is subject to [YouTube's Terms of Service](https://www.youtube.com/t/terms).
