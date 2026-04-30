# YouTube Depression Signal Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![VADER](https://img.shields.io/badge/NLP-VADER-orange)
![TextBlob](https://img.shields.io/badge/NLP-TextBlob-orange)
![AFINN](https://img.shields.io/badge/NLP-AFINN-orange)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09a3d5)
![SBERT](https://img.shields.io/badge/Embedding-MiniLM--L6-blueviolet)
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

| Step                       | Script                | Input                  | Output                                              |
| -------------------------- | --------------------- | ---------------------- | --------------------------------------------------- |
| 0 — Data collection        | `collect.py`          | API key, category list | `data/raw/videos.json`, `data/raw/comments.json`    |
| 1 — Video analysis         | `video_analysis.py`   | `videos.json`          | `data/processed/video_features.csv`                 |
| 2 — Comment analysis       | `comment_analysis.py` | `comments.json`        | `data/processed/comment_features.csv`               |
| 3 — Cross-category linking | `linking.py`          | both feature CSVs      | `data/outputs/heatmap.png`, `data/outputs/tsne.png` |
| 4 — Modeling               | `modeling.py`         | both feature CSVs      | `data/outputs/model_results.json`                   |

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

| Category              | Role in study                                            |
| --------------------- | -------------------------------------------------------- |
| Mental health         | Baseline — expected higher depression signal rate        |
| Fitness / wellness    | Adjacent to mental health; motivational framing          |
| Music / entertainment | High-traffic mainstream content; general audience        |
| News / current events | Emotionally charged but non-personal framing             |
| Personal vlog         | First-person narrative; higher self-disclosure potential |

**Per-video collection:** top 50 comments sorted by likes, stored with `video_id` as a foreign key. Target: ~40–50 videos per category, ~200–250 videos total, ~10,000–12,500 comments.

---

## Step 1 — Video-Level Feature Extraction

Since video categories are already assigned at collection time (Step 0), Step 1 focuses on extracting NLP features from video metadata (title and description) that characterize each video's tone and framing. These features are joined with comment-level data in Step 3 to examine whether video framing influences comment-level depression signal patterns.

**Features extracted per video:**

| Feature                   | Method                       | Purpose                                  |
| ------------------------- | ---------------------------- | ---------------------------------------- |
| `title_sentiment`         | VADER compound score         | Tone of video title                      |
| `desc_sentiment_vader`    | VADER compound score         | Tone of video description                |
| `desc_sentiment_textblob` | TextBlob polarity            | Secondary sentiment signal               |
| `desc_subjectivity`       | TextBlob subjectivity        | Degree of personal / opinionated framing |
| `desc_word_count`         | Token count                  | Description length                       |
| `desc_lexical_diversity`  | unique tokens / total tokens | Vocabulary richness of description       |

Output: `data/processed/video_features.csv` with one row per video.

---

## Step 2 — Depression Signal Detection (Three Layers)

Depression signal detection operates at the comment level. Sentiment analysis alone is insufficient — negative sentiment does not imply depression (e.g. "this video is garbage" is negative but not depression-indicative). We use three layers of increasing specificity:

**Layer 1 — Multi-tool sentiment scoring** (broad signal)

Rather than relying on a single model, each comment is scored by three independent tools to produce a more robust sentiment signal:

| Tool     | Score                 | Notes                                                      |
| -------- | --------------------- | ---------------------------------------------------------- |
| VADER    | compound ∈ [−1, 1]    | Optimized for short social media text                      |
| TextBlob | polarity ∈ [−1, 1]    | Lexicon-based, stable baseline                             |
| TextBlob | subjectivity ∈ [0, 1] | High subjectivity correlates with personal self-disclosure |
| AFINN    | integer sum           | Word-level valence scoring                                 |

**Layer 2 — TF-IDF weighted depression lexicon matching** (domain-specific)

Comments are matched against a mental health lexicon (LIWC categories: negative emotion, sadness, anxiety, health) and a depression-specific wordlist. Rather than binary presence/absence matching, TF-IDF weighting is applied with the vocabulary locked to the depression lexicon. This captures how prominently depression-relevant terms appear relative to the rest of the comment, and extends to bigrams (e.g. "feel hopeless", "can't sleep") for richer coverage.

**Layer 3 — Self-disclosure pattern detection** (high-precision)

First-person pronoun + depression-indicative vocabulary combinations are identified using part-of-speech tagging (spaCy). This captures genuine personal disclosure (e.g. "I feel hopeless", "I've been struggling", "I can't do this anymore") rather than reactions to video content, and carries the strongest research signal.

**Additional text features** extracted alongside the three layers:

| Feature             | Formula                      | Rationale                                                                    |
| ------------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| `lexical_diversity` | unique tokens / total tokens | Depressive language tends to be more repetitive                              |
| `word_count`        | token count                  | Very short comments (<5 words) are less likely to be genuine self-disclosure |

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

## Step 4 — Supervised Classification

If labeled examples are available, a supervised classifier can be trained to predict whether a comment contains a depression signal. Three models are compared using a standardized Pipeline with TF-IDF feature extraction and 5-fold cross-validation:

| Model               | Notes                                                             |
| ------------------- | ----------------------------------------------------------------- |
| Logistic Regression | Strong baseline for text classification                           |
| Naive Bayes         | Computationally efficient; well-suited for sparse TF-IDF features |
| SVM (linear kernel) | Strong performance on high-dimensional text                       |

5-fold cross-validation is used throughout to ensure stable estimates that are not sensitive to a single train/test split. Models are evaluated on F1-macro to account for potential class imbalance between depression-signal and non-signal comments.

---

## Models and Tools

| Task                                  | Model / Tool                                                  |
| ------------------------------------- | ------------------------------------------------------------- |
| Sentiment analysis (Layer 1)          | VADER, TextBlob, AFINN                                        |
| Depression lexicon matching (Layer 2) | LIWC-aligned categories + TF-IDF (bigrams, locked vocabulary) |
| Self-disclosure detection (Layer 3)   | spaCy POS tagging + first-person pronoun patterns             |
| Comment embedding                     | `sentence-transformers/all-MiniLM-L6-v2`                      |
| Clustering                            | HDBSCAN + GMM (AIC/BIC selection)                             |
| Dimensionality reduction              | PCA (50D) → t-SNE (2D)                                        |
| Supervised classification             | Logistic Regression, Naive Bayes, SVM via sklearn Pipeline    |
| Cross-validation                      | 5-fold, F1-macro, sklearn `cross_val_score`                   |
| Visualization                         | `seaborn`, `matplotlib`                                       |

---

## Methodology Note

**Standard / baseline methods.** The following techniques form the established
baseline for the pipeline: VADER, TextBlob, AFINN, spaCy POS tagging, TF-IDF
(sklearn `TfidfVectorizer`), PCA, t-SNE, HDBSCAN, GaussianMixture (GMM),
LogisticRegression, MultinomialNB, and `cross_val_score`-based cross-validation.

**Extensions beyond the baseline.** A few additions go beyond the standard
toolkit. Each is a well-established practice adopted to address a specific
limitation of the baseline:

- **LinearSVM as a third classifier** alongside LogReg / NB. SVM is the best
  performer of the three (F1-macro = 0.829 vs 0.776 LogReg / 0.484 NB), and
  including it makes the model comparison non-trivial.
- **`StratifiedKFold` instead of plain `cross_val_score` splits.** With a 6.9%
  positive class, plain k-fold can produce folds with very few positives and
  high variance; stratification preserves the class balance per fold.
- **Statistical significance testing** (`scipy.stats.f_oneway`,
  `kruskal`, `ks_2samp` with Bonferroni correction) for cross-category
  differences. We use ANOVA + Kruskal-Wallis + pairwise KS as the standard
  combination for comparing distributions across groups.
- **Sentence-transformers (`all-MiniLM-L6-v2`) for comment embeddings.** The
  PCA / t-SNE / HDBSCAN / GMM pipeline operates on a high-dimensional vector
  input; we use SBERT to produce 384-d sentence embeddings that capture
  semantic similarity, since TF-IDF embeddings would lose most of the local
  semantic structure the clustering step is designed to find.
- **spaCy `lemma_` and `dep_` in addition to `pos_`.** Beyond POS tagging, we
  also use lemma-based vocabulary matching (so "struggling" and "struggled"
  both match `struggle`) and dependency tags to verify that the first-person
  pronoun is the syntactic subject (`nsubj` / `nsubjpass`), reducing false
  positives in Layer 3.

---

## Core Result

Cross-category comparison of depression signal prevalence
(n = 10,867 comments from 230 high-engagement videos across 5 categories):

| Category              |     n | Depression signal rate¹ | Avg neg-sentiment² | Self-disclosure rate³ | Lex hit rate⁴ |
| --------------------- | ----: | ----------------------: | -----------------: | --------------------: | ------------: |
| Mental health         | 2,499 |               **28.5%** |          **0.142** |             **21.1%** |     **44.9%** |
| Fitness / wellness    | 2,399 |                    6.2% |              0.082 |                  4.7% |          9.9% |
| News / current events | 2,450 |                    5.1% |              0.141 |                  1.2% |          8.1% |
| Personal vlog         | 2,300 |                    3.2% |              0.071 |                  2.6% |          6.7% |
| Music / entertainment | 1,219 |                    2.2% |              0.095 |                  1.5% |          5.1% |

¹ Share of comments in the top decile of the global composite L1+L2+L3 signal score.
² Mean of the aggregated VADER + TextBlob + AFINN negativity score, range [0, 1] (higher = more negative).
³ Share of comments matching first-person pronoun + depression-vocabulary via spaCy POS tagging.
⁴ Share of comments with at least one TF-IDF lexicon hit (any LIWC-aligned depression term).

**Statistical significance.** Differences in the composite signal score across
the five categories are highly significant: one-way ANOVA _F_ = 586.4,
_p_ < 0.001; Kruskal-Wallis _H_ = 1491.8, _p_ < 0.001. All 10 pairwise category
contrasts remain significant after Bonferroni correction (two-sample KS-test,
α = 0.05).

### Video-level framing (RQ2)

Joining each comment to its parent video's metadata lets us compare the
_context_ that comments are written under — does negative video framing produce
more depression-related self-disclosure?

| Category              | Title sentiment | Desc sentiment | Desc subjectivity | Desc length |
| --------------------- | --------------: | -------------: | ----------------: | ----------: |
| Mental health         |           −0.07 |           0.36 |              0.31 |         106 |
| Personal vlog         |           +0.11 |           0.23 |              0.15 |          45 |
| Fitness / wellness    |           +0.12 |           0.56 |              0.37 |         173 |
| News / current events |       **−0.32** |           0.06 |              0.40 |         131 |
| Music / entertainment |           +0.01 |           0.39 |              0.36 |         192 |

Title / description sentiment from VADER compound ∈ [−1, +1]; subjectivity from
TextBlob ∈ [0, 1]; length in tokens. All values are comment-weighted means
(each video is averaged proportionally to the number of comments it received).

### Interpretation

**RQ1 — natural prevalence varies sharply by category.** Mental health stands
apart on every comment-level dimension: its depression signal rate,
self-disclosure rate, and lexicon hit rate (44.9%) are 4–9× higher than any
other category. This is consistent with the hypothesis that mental-health
comment sections host disproportionately more spontaneous self-disclosure
relative to other types of mainstream content.

**RQ2 — video framing does _not_ drive comment-level self-disclosure.** News
has the most negative video framing (title sentiment **−0.32**, the most
negative of any category) yet the _lowest_ self-disclosure rate (**1.2%**). By
contrast, mental health has only mildly negative titles (−0.07) but the
**17×** higher self-disclosure rate. The decoupling shows that video tone is
a poor predictor of personal disclosure: News negativity is content-driven
(politics, conflict) rather than personal, and it doesn't elicit personal
self-disclosure in the comment section. **Topic domain, not video sentiment,
appears to drive whether commenters use mainstream comment sections to talk
about their own mental state.**

**Supervised classification.** Predicting `self_disclosure_flag` from raw
comment text with TF-IDF (1–2 grams) + 5-fold CV: LinearSVM achieves
**F1-macro = 0.829 ± 0.014** (best of three models). Top positive features
(`lost`, `crying`, `depression`, `anxiety`, `me`, `tears`, `cried`, `my`)
are exactly the depression-vocabulary + first-person tokens that Layer 3 was
designed to detect, validating the labeling pipeline from a different angle.

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
