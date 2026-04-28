# YouTube Depression Signal Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![VADER](https://img.shields.io/badge/NLP-VADER-orange)
![TextBlob](https://img.shields.io/badge/NLP-TextBlob-orange)
![AFINN](https://img.shields.io/badge/NLP-AFINN-orange)
![LIWC](https://img.shields.io/badge/Lexicon-LIWC-purple)
![GMM](https://img.shields.io/badge/Clustering-GMM-teal)
![HDBSCAN](https://img.shields.io/badge/Clustering-HDBSCAN-teal)
![tSNE](https://img.shields.io/badge/Visualization-PCA--tSNE-9cf)
![sklearn](https://img.shields.io/badge/Model-sklearn--Pipeline-red)
![YouTube](https://img.shields.io/badge/Data-YouTube%20API%20v3-ff0000)
![Status](https://img.shields.io/badge/Status-Active-success)

> Measuring the natural prevalence and linguistic patterns of depression-indicative language across diverse high-traffic YouTube video categories.

We collect comments from mainstream, high-engagement YouTube videos spanning five content categories — and use NLP to detect depression-related linguistic signals that appear organically in everyday comment sections. Rather than searching for depression content directly, we measure how often and in what form these signals surface in ordinary public discourse.

This project does **not** attempt to diagnose depression. We detect _depression-indicative linguistic signals_ as a proxy for spontaneous, self-disclosed depression-related expression in the wild.

---

## Research Questions

1. Across different YouTube content categories, how does the natural prevalence of depression-indicative language vary?
2. To what extent does video-level contextual framing shape the linguistic patterns of depression-related self-disclosure in comments?

---

## Design Rationale

Most prior work collects data from depression-specific communities (e.g. r/depression), which introduces **self-selection bias** — the users found there have already self-identified as part of a depression community. This project takes a different approach: we sample from high-traffic general-interest videos across five categories, then scan every comment for depression-indicative language using a three-layer detection method. This better reflects the true **prevalence** of depression-related expression in everyday online discourse, and allows us to compare how that expression differs across content contexts.

The analysis unit is the **comment**, not the video. A music video has no depression relevance — but a comment underneath it saying _"I've been struggling so much lately, this is the only thing keeping me going"_ does. We are measuring the signal at the comment level, across video contexts.

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
│   ├── video_analysis.py     # Step 1: Video-level NLP feature extraction
│   ├── comment_analysis.py   # Step 2: Three-layer depression signal detection
│   ├── linking.py            # Step 3: Clustering, visualization, cross-category comparison
│   └── modeling.py           # Step 4: Supervised classification with cross-validation
│
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

| Step | Script | Input | Output |
| --- | --- | --- | --- |
| 0 — Data collection | `collect.py` | API key, category list | `data/raw/videos.json`, `data/raw/comments.json` |
| 1 — Video analysis | `video_analysis.py` | `videos.json` | `data/processed/video_features.csv` |
| 2 — Comment analysis | `comment_analysis.py` | `comments.json` | `data/processed/comment_features.csv` |
| 3 — Cross-category linking | `linking.py` | both feature CSVs | `data/outputs/heatmap.png`, `data/outputs/tsne.png` |
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

# Step 1: video-level feature extraction
python src/video_analysis.py

# Step 2: comment-level depression signal detection
python src/comment_analysis.py

# Step 3: cross-category comparison and visualization
python src/linking.py

# Step 4: supervised classification with cross-validation
python src/modeling.py
```

---

## Data Collection Strategy

Videos are sampled from five predefined content categories, each representing a distinct type of everyday YouTube content. Within each category, videos are ranked by view count and the top-N are selected. No depression-related keywords are used to filter or rank videos — depression relevance is determined entirely post-hoc at the **comment level** in Step 2.

| Category | Role in study |
| --- | --- |
| Mental health | Baseline — expected higher depression signal rate |
| Fitness / wellness | Adjacent to mental health; motivational framing |
| Music / entertainment | High-traffic mainstream content; general audience |
| News / current events | Emotionally charged but non-personal framing |
| Personal vlog | First-person narrative; higher self-disclosure potential |

**Per-video collection:** top 50 comments sorted by likes, stored with `video_id` as a foreign key. Target: ~40–50 videos per category, ~200–250 videos total, ~10,000–12,500 comments.

---

## Step 1 — Video-Level Feature Extraction

Since video categories are already assigned at collection time (Step 0), Step 1 focuses on extracting NLP features from video metadata (title and description) that characterize each video's tone and framing. These features are joined with comment-level data in Step 3 to examine whether video framing influences comment-level depression signal patterns.

**Features extracted per video:**

| Feature | Method | Purpose |
| --- | --- | --- |
| `title_sentiment` | VADER compound score | Tone of video title |
| `desc_sentiment_vader` | VADER compound score | Tone of video description |
| `desc_sentiment_textblob` | TextBlob polarity | Secondary sentiment signal |
| `desc_subjectivity` | TextBlob subjectivity | Degree of personal / opinionated framing |
| `desc_word_count` | Token count | Description length |
| `desc_lexical_diversity` | unique tokens / total tokens | Vocabulary richness of description |

Output: `data/processed/video_features.csv` with one row per video.

---

## Step 2 — Depression Signal Detection (Three Layers)

Depression signal detection operates at the comment level. Sentiment analysis alone is insufficient — negative sentiment does not imply depression (e.g. "this video is garbage" is negative but not depression-indicative). We use three layers of increasing specificity:

**Layer 1 — Multi-tool sentiment scoring** (broad signal)

Rather than relying on a single model, each comment is scored by three independent tools to produce a more robust sentiment signal:

| Tool | Score | Notes |
| --- | --- | --- |
| VADER | compound ∈ [−1, 1] | Optimized for short social media text |
| TextBlob | polarity ∈ [−1, 1] | Lexicon-based, stable baseline |
| TextBlob | subjectivity ∈ [0, 1] | High subjectivity correlates with personal self-disclosure |
| AFINN | integer sum | Word-level valence scoring |

**Layer 2 — TF-IDF weighted depression lexicon matching** (domain-specific)

Comments are matched against a mental health lexicon (LIWC categories: negative emotion, sadness, anxiety, health) and a depression-specific wordlist. Rather than binary presence/absence matching, TF-IDF weighting is applied with the vocabulary locked to the depression lexicon. This captures how prominently depression-relevant terms appear relative to the rest of the comment, and extends to bigrams (e.g. "feel hopeless", "can't sleep") for richer coverage.

**Layer 3 — Self-disclosure pattern detection** (high-precision)

First-person pronoun + depression-indicative vocabulary combinations are identified using part-of-speech tagging (spaCy). This captures genuine personal disclosure (e.g. "I feel hopeless", "I've been struggling", "I can't do this anymore") rather than reactions to video content, and carries the strongest research signal.

**Additional text features** extracted alongside the three layers:

| Feature | Formula | Rationale |
| --- | --- | --- |
| `lexical_diversity` | unique tokens / total tokens | Depressive language tends to be more repetitive |
| `word_count` | token count | Very short comments (<5 words) are less likely to be genuine self-disclosure |

Each comment receives a composite **depression signal score** aggregating all three layers. Cross-category comparison is performed on this score.

---

## Step 3 — Cross-Category Comparison and Visualization

**Clustering**

Comment embeddings (`sentence-transformers/all-MiniLM-L6-v2`) are clustered using two complementary methods:

- **HDBSCAN** — density-based, does not require specifying the number of clusters, handles noise naturally.
- **Gaussian Mixture Models (GMM)** — soft clustering; each comment receives a probability of belonging to each cluster rather than a hard label. More appropriate for emotionally ambiguous comments. Optimal number of components selected by AIC/BIC.

Both clustering results are reported and compared for interpretability.

**Dimensionality reduction and visualization**

A two-step pipeline is used:

1. **PCA** reduces embeddings from high dimensions to 50 components, removing noise while preserving global structure.
2. **t-SNE** further reduces to 2D, preserving local neighbor relationships for visualization.

This pipeline produces cleaner cluster separation than direct 2D projection, consistent with established practice for high-dimensional text embeddings.

**Statistical tests**

Differences in depression signal rates across the five categories are tested for statistical significance using ANOVA and KS-test.

---

## Step 4 — Supervised Classification (Optional)

If labeled examples are available, a supervised classifier can be trained to predict whether a comment contains a depression signal. Three models are compared using a standardized Pipeline with TF-IDF feature extraction and 5-fold cross-validation:

| Model | Notes |
| --- | --- |
| Logistic Regression | Strong baseline for text classification |
| Naive Bayes | Computationally efficient; well-suited for sparse TF-IDF features |
| SVM (linear kernel) | Strong performance on high-dimensional text |

5-fold cross-validation is used throughout to ensure stable estimates that are not sensitive to a single train/test split. Models are evaluated on F1-macro to account for potential class imbalance between depression-signal and non-signal comments.

---

## Models and Tools

| Task | Model / Tool |
| --- | --- |
| Sentiment analysis (Layer 1) | VADER, TextBlob, AFINN |
| Depression lexicon matching (Layer 2) | LIWC + TF-IDF (bigrams, locked vocabulary) |
| Self-disclosure detection (Layer 3) | spaCy POS tagging + first-person pronoun patterns |
| Comment embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Clustering | HDBSCAN + GMM (AIC/BIC selection) |
| Dimensionality reduction | PCA (50D) → t-SNE (2D) |
| Supervised classification | Logistic Regression, Naive Bayes, SVM via sklearn Pipeline |
| Cross-validation | 5-fold, F1-macro, sklearn `cross_val_score` |
| Visualization | `seaborn`, `matplotlib` |

---

## Expected Core Result

The primary output is a cross-category comparison of depression signal prevalence:

| Category | Depression signal rate | Avg sentiment | Self-disclosure rate |
| --- | --- | --- | --- |
| Mental health | (measured) | (measured) | (measured) |
| Personal vlog | (measured) | (measured) | (measured) |
| Fitness / wellness | (measured) | (measured) | (measured) |
| News / current events | (measured) | (measured) | (measured) |
| Music / entertainment | (measured) | (measured) | (measured) |

Differences across categories are tested for statistical significance (ANOVA / KS-test).

---

## Limitations

- **Comment ≠ commenter**: the presence of depression-indicative language does not imply the commenter has depression.
- **Sentiment ≠ depression**: negative sentiment is a necessary but not sufficient signal — the three-layer method is designed to address this, but false positives remain possible.
- **Top-comment bias**: sorting by likes favors emotionally resonant language; quieter expressions of distress may be underrepresented.
- **Category sampling**: the five categories were defined by the research team and do not exhaustively represent all YouTube content.
- **No clinical ground truth**: all labels are proxies based on linguistic and behavioral signals, not clinical assessment.

Results should be interpreted as measuring _the natural prevalence of depression-indicative linguistic patterns in public comment sections_, not as clinical or diagnostic claims.

---

## Team

<!-- Add names / roles here -->

---

## License

For academic use only. Data collected via YouTube Data API is subject to [YouTube's Terms of Service](https://www.youtube.com/t/terms).
