![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/NLP-Transformers-green)
![Status](https://img.shields.io/badge/Status-Active-success)

# YouTube Depression Signal Detection

> Measuring the natural prevalence and linguistic patterns of depression-indicative language across diverse high-traffic YouTube video categories.

We collect comments from mainstream, high-engagement YouTube videos spanning multiple content categories — and use NLP to detect depression-related linguistic signals that appear organically in everyday comment sections. Rather than searching for depression content directly, we measure how often and in what form these signals surface in ordinary public discourse.

This project does **not** attempt to diagnose depression. We detect *depression-indicative linguistic signals* as a proxy for spontaneous, self-disclosed depression-related expression in the wild.

---

## Research Questions

1. Across different YouTube content categories, how does the natural prevalence of depression-indicative language vary?
2. To what extent does video-level contextual framing shape the linguistic patterns of depression-related self-disclosure in comments?

---

## Design Rationale

Most prior work collects data from depression-specific communities (e.g. r/depression), which introduces **self-selection bias** — the users found there have already self-identified as part of a depression community. This project takes a different approach: we sample from high-traffic general-interest videos across multiple categories, then scan the comment sections for depression-indicative language. This better reflects the true **prevalence** of depression-related expression in everyday online discourse, and allows us to compare how that expression differs across content contexts.

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

Videos are sampled from five predefined content categories, each representing a distinct type of everyday YouTube content. Within each category, videos are ranked by view count and the top-N are selected, ensuring high engagement and broad audience reach — without targeting depression content directly.

| Category | Rationale |
|----------|-----------|
| Mental health | Baseline — expected higher depression signal rate |
| Fitness / wellness | Adjacent to mental health; motivational framing |
| Music / entertainment | High-traffic mainstream content; general audience |
| News / current events | Emotionally charged but non-personal framing |
| Personal vlog | First-person narrative; potential for self-disclosure |

**Per-video collection:** top 50 comments sorted by likes, stored with `video_id` as a foreign key. Target: ~40–50 videos per category, ~200–250 videos total, ~10,000–12,500 comments.

No depression-related keywords are used to filter or rank videos within a category. Depression relevance is determined entirely *post-hoc* at the comment level via NLP in Steps 1 and 2.

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

- **Comment ≠ commenter**: the presence of depression-indicative language does not imply the commenter has depression.
- **Top-comment bias**: sorting by likes favors emotionally resonant language over quieter, less-engaged expression.
- **Category sampling**: the five categories were defined by the research team and do not exhaustively represent all YouTube content.
- **No clinical ground truth**: all labels are proxies based on linguistic and behavioral signals, not clinical assessment.

Results should be interpreted as measuring *the natural prevalence of depression-indicative linguistic patterns in public comment sections*, not as clinical or diagnostic claims.

---

## Team

<!-- Add names / roles here -->

---

## License

For academic use only. Data collected via YouTube Data API is subject to [YouTube's Terms of Service](https://www.youtube.com/t/terms).
