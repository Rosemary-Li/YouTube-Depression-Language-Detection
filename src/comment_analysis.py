"""
comment_analysis.py — Step 2: Comment-level depression signal detection
═══════════════════════════════════════════════════════════════════════════════

Three-layer detection at the comment level:

  Layer 1 — Sentiment (broad)
      cardiffnlp/twitter-roberta-base-sentiment → P(neg/neu/pos).
      Baseline only. Negative sentiment ≠ depression.

  Layer 2 — Depression lexicon matching (domain-specific)
      LIWC-inspired categories (sadness, anxiety, neg-emotion, health) plus a
      depression-specific wordlist (hopeless, worthless, exhausted, numb,
      empty, can't go on, …). LIWC is proprietary; the wordlists here are
      curated public-domain seeds drawn from prior depression-detection
      literature.

  Layer 3 — Self-disclosure (high-precision)
      First-person pronoun co-occurring with depression-indicative vocabulary
      within a short token window, plus regex patterns for canonical
      disclosure constructions ("I feel hopeless", "I've been struggling",
      "I can't do this", …). The strongest research signal.

Plus:
  • Sentence embeddings — sentence-transformers/all-MiniLM-L6-v2.
  • Clustering — HDBSCAN on UMAP-reduced embeddings (noise-aware, no fixed k).
  • Composite depression signal score — weighted aggregate of all three layers.

Input  : data/raw/comments.json
Output : data/processed/comment_features.csv
         data/processed/comment_embeddings.npy

Usage  : python src/comment_analysis.py [--sample N] [--no-cluster]
"""

import argparse
import html
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
COMMENTS_FILE = ROOT / "data" / "raw" / "comments.json"
OUT_DIR       = ROOT / "data" / "processed"
OUT_FILE      = OUT_DIR / "comment_features.csv"
EMB_FILE      = OUT_DIR / "comment_embeddings.npy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("comment_analysis")

# ── Models / device ───────────────────────────────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Lexicons — LIWC-inspired categories + depression-specific terms
# LIWC dictionaries are proprietary; these seeds draw on public-domain word
# lists used in depression-detection literature (Pennebaker et al., 2015;
# De Choudhury et al., 2013; Coppersmith et al., 2014).
# ─────────────────────────────────────────────────────────────────────────────
DEPRESSION_LEXICON = {
    "sadness": {
        "sad", "sadness", "cry", "crying", "cried", "tears", "tearful",
        "weep", "weeping", "grief", "grieving", "mourn", "mourning",
        "sorrow", "miserable", "misery", "heartbroken", "depressed",
        "depressing", "depression", "lonely", "loneliness", "alone",
        "isolated", "isolation", "unhappy", "gloomy", "melancholy",
        "despair", "despondent",
    },
    "anxiety": {
        "anxious", "anxiety", "worried", "worry", "worrying", "nervous",
        "panic", "panicked", "panicking", "afraid", "scared", "fear",
        "fearful", "terrified", "stress", "stressed", "stressful",
        "overwhelm", "overwhelmed", "overwhelming", "tense", "uneasy",
        "restless", "dread", "dreading",
    },
    "negative_emotion": {
        "hate", "hated", "angry", "anger", "rage", "frustrated",
        "frustration", "annoyed", "irritated", "bitter", "resentful",
        "guilty", "guilt", "ashamed", "shame", "embarrassed", "humiliated",
        "regret", "regretful", "disgusted", "disgust", "awful", "terrible",
        "horrible", "pain", "painful", "hurt", "hurting", "suffering",
        "suffer",
    },
    "health": {
        "sick", "ill", "illness", "diagnosed", "diagnosis", "therapy",
        "therapist", "psychiatrist", "psychologist", "counselor",
        "counseling", "medication", "meds", "antidepressant",
        "antidepressants", "prozac", "zoloft", "lexapro", "ssri",
        "prescription", "hospital", "hospitalized", "treatment", "psych",
        "ward", "clinical",
    },
    "hopelessness": {
        "hopeless", "hopelessness", "worthless", "worthlessness",
        "pointless", "meaningless", "useless", "helpless", "trapped",
        "stuck", "lost", "broken", "defeated",
    },
    "exhaustion": {
        "exhausted", "exhaustion", "tired", "drained", "burnout",
        "fatigue", "fatigued", "weary", "spent", "depleted",
    },
    "numbness": {
        "numb", "numbness", "empty", "emptiness", "hollow", "void",
        "detached", "disconnected", "dissociate", "dissociated", "blank",
    },
    "suicidal": {
        "suicide", "suicidal", "kms", "kys",
    },
}

# Multi-word phrases (matched as substring on lowered text)
DEPRESSION_PHRASES = {
    "give up", "gave up", "giving up", "burned out", "burnt out",
    "ending it", "end it all", "end my life", "kill myself",
    "can't go on", "cant go on", "can't do this", "cant do this",
    "no point", "no future", "no hope", "want to die", "wanna die",
    "tired of living", "sick of living", "falling apart",
}

# Vocabulary that, combined with first-person pronouns, indicates self-
# disclosure. We deliberately exclude very generic negative-emotion words
# ("hate", "angry") — those produce many false positives ("I hate Mondays").
DISCLOSURE_VOCAB = (
    DEPRESSION_LEXICON["sadness"]
    | DEPRESSION_LEXICON["anxiety"]
    | DEPRESSION_LEXICON["hopelessness"]
    | DEPRESSION_LEXICON["exhaustion"]
    | DEPRESSION_LEXICON["numbness"]
    | DEPRESSION_LEXICON["suicidal"]
    | {"struggle", "struggled", "struggling", "struggles", "cope", "coping"}
)

FIRST_PERSON = {
    "i", "im", "i'm", "ive", "i've", "me", "my", "mine", "myself",
}

# Self-disclosure regex patterns — canonical first-person constructions
DISCLOSURE_PATTERNS = [
    re.compile(r"\bi(?:'ve|\s+have)\s+been\s+(?:so\s+|really\s+|very\s+|feeling\s+)?\w+", re.I),
    re.compile(r"\bi\s+(?:feel|felt|am|was|get|got)\s+(?:so\s+|really\s+|very\s+|just\s+)?\w+", re.I),
    re.compile(r"\bi\s+can(?:'t|not|\s+not)\s+(?:do|go|take|stand|handle|cope|keep)\b", re.I),
    re.compile(r"\bi(?:'m|\s+am)\s+(?:so\s+|really\s+|just\s+|always\s+)?(?:tired|exhausted|done|broken|empty|numb|lost|alone|lonely|sad|depressed|hopeless|worthless)\b", re.I),
    re.compile(r"\bi\s+(?:want|wanna|need)\s+to\s+(?:die|disappear|stop|leave|end)\b", re.I),
    re.compile(r"\bmy\s+(?:depression|anxiety|mental\s+health|therapist|psychiatrist|meds)\b", re.I),
]

TOKEN_RE = re.compile(r"[a-zA-Z']+")
URL_RE   = re.compile(r"http\S+|www\.\S+")


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(s: str) -> str:
    """Decode HTML entities, strip URLs, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Depression lexicon matching
# ─────────────────────────────────────────────────────────────────────────────
def lexicon_features(text: str) -> dict:
    tokens = tokenize(text)
    n_tokens = max(len(tokens), 1)
    token_set = set(tokens)

    counts = {cat: 0 for cat in DEPRESSION_LEXICON}
    matched = []
    total_hits = 0

    for cat, words in DEPRESSION_LEXICON.items():
        for w in words:
            if " " in w:
                continue
            if w in token_set:
                counts[cat] += 1
                total_hits += 1
                matched.append(w)

    text_lower = text.lower()
    for phrase in DEPRESSION_PHRASES:
        if phrase in text_lower:
            total_hits += 1
            matched.append(phrase)
            # phrases cross multiple categories; bucket them under hopelessness
            counts["hopelessness"] += 1

    return {
        "lex_total_hits": total_hits,
        "lex_match_rate": total_hits / n_tokens,
        "lex_unique_terms": ";".join(sorted(set(matched))[:8]),
        **{f"lex_{cat}": cnt for cat, cnt in counts.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Self-disclosure detection
# ─────────────────────────────────────────────────────────────────────────────
def self_disclosure_features(text: str) -> dict:
    tokens = tokenize(text)
    if not tokens:
        return {
            "self_disclosure_flag": 0,
            "self_disclosure_score": 0.0,
            "self_disclosure_reason": "",
        }

    fp_idx    = [i for i, t in enumerate(tokens) if t in FIRST_PERSON]
    vocab_idx = [i for i, t in enumerate(tokens) if t in DISCLOSURE_VOCAB]

    # Co-occurrence: first-person pronoun within 6 tokens of disclosure vocab
    co_occur = False
    co_occur_terms = []
    if fp_idx and vocab_idx:
        for i in fp_idx:
            for j in vocab_idx:
                if abs(i - j) <= 6:
                    co_occur = True
                    co_occur_terms.append(tokens[j])
                    break
            if co_occur:
                break

    # Canonical disclosure constructions
    matched_patterns = []
    for pat in DISCLOSURE_PATTERNS:
        m = pat.search(text)
        if m:
            matched_patterns.append(m.group(0).strip())
            if len(matched_patterns) >= 3:
                break

    flag = int(co_occur or bool(matched_patterns))
    score = min(1.0, 0.5 * int(co_occur) + 0.25 * len(matched_patterns))

    reason_parts = []
    if co_occur_terms:
        reason_parts.append(f"co-occur:{co_occur_terms[0]}")
    reason_parts.extend(matched_patterns[:2])

    return {
        "self_disclosure_flag": flag,
        "self_disclosure_score": score,
        "self_disclosure_reason": " | ".join(reason_parts)[:200],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Sentiment (RoBERTa)
# ─────────────────────────────────────────────────────────────────────────────
def run_sentiment(texts: list[str], batch_size: int = 32) -> list[dict]:
    log.info(f"Loading sentiment model: {SENTIMENT_MODEL} ({DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(DEVICE)
    model.eval()
    # cardiffnlp twitter-roberta: id2label = {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE}
    id2label = ["negative", "neutral", "positive"]

    rows = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="layer1-sentiment"):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(DEVICE)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                rows.append({
                    "sent_neg":   float(p[0]),
                    "sent_neu":   float(p[1]),
                    "sent_pos":   float(p[2]),
                    "sent_label": id2label[int(p.argmax())],
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings + clustering
# ─────────────────────────────────────────────────────────────────────────────
def run_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    log.info(f"Loading embedding model: {EMBEDDING_MODEL} ({DEVICE})")
    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb


def cluster_embeddings(emb: np.ndarray, min_cluster_size: int = 25):
    import hdbscan
    import umap

    log.info("UMAP → 5-D for clustering")
    reducer = umap.UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(emb)

    log.info(f"HDBSCAN clustering (min_cluster_size={min_cluster_size})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    log.info("UMAP → 2-D for visualization")
    viz = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    ).fit_transform(emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(f"Clusters: {n_clusters} | noise points: {n_noise}/{len(labels)}")
    return labels, viz


# ─────────────────────────────────────────────────────────────────────────────
# Composite signal score
# ─────────────────────────────────────────────────────────────────────────────
def composite_score(row) -> float:
    """Weighted aggregate reflecting the increasing specificity of the layers.

       L1 (sentiment, broad)        weight 0.20
       L2 (lexicon, domain)         weight 0.30
       L3 (self-disclosure, precise) weight 0.50
    """
    sent_neg = float(row["sent_neg"])
    lex_norm = min(1.0, float(row["lex_total_hits"]) / 3.0)
    disc     = float(row["self_disclosure_score"])
    return 0.20 * sent_neg + 0.30 * lex_norm + 0.50 * disc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(sample: int | None = None, do_cluster: bool = True) -> None:
    log.info(f"Loading comments from {COMMENTS_FILE}")
    with open(COMMENTS_FILE) as f:
        raw = json.load(f)
    log.info(f"Loaded {len(raw)} raw comments")

    df = pd.DataFrame(raw)
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    log.info(f"After cleaning: {len(df)} non-empty comments")

    if sample:
        df = df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)
        log.info(f"Sampled {len(df)} for testing")

    texts = df["text_clean"].tolist()

    # Layer 2 — lexicon
    log.info("Layer 2 — depression lexicon matching")
    lex_df = pd.DataFrame([lexicon_features(t) for t in tqdm(texts, desc="layer2-lexicon")])
    df = pd.concat([df, lex_df], axis=1)

    # Layer 3 — self-disclosure
    log.info("Layer 3 — self-disclosure detection")
    disc_df = pd.DataFrame([self_disclosure_features(t) for t in tqdm(texts, desc="layer3-disclosure")])
    df = pd.concat([df, disc_df], axis=1)

    # Layer 1 — sentiment
    log.info("Layer 1 — RoBERTa sentiment scoring")
    sent_df = pd.DataFrame(run_sentiment(texts))
    df = pd.concat([df, sent_df], axis=1)

    # Composite
    df["depression_signal_score"] = df.apply(composite_score, axis=1)

    # Embeddings + clustering
    if do_cluster:
        log.info("Computing sentence embeddings")
        emb = run_embeddings(texts)
        np.save(EMB_FILE, emb)
        log.info(f"Embeddings → {EMB_FILE}  shape={emb.shape}")

        labels, viz = cluster_embeddings(emb)
        df["cluster_label"] = labels
        df["umap_x"] = viz[:, 0]
        df["umap_y"] = viz[:, 1]
    else:
        df["cluster_label"] = -1
        df["umap_x"] = np.nan
        df["umap_y"] = np.nan

    # ── Output schema ────────────────────────────────────────────────────────
    cols = [
        "comment_id", "video_id", "category", "category_label",
        "text", "like_count", "published_at",
        # Layer 1
        "sent_label", "sent_neg", "sent_neu", "sent_pos",
        # Layer 2
        "lex_total_hits", "lex_match_rate", "lex_unique_terms",
        *[f"lex_{cat}" for cat in DEPRESSION_LEXICON],
        # Layer 3
        "self_disclosure_flag", "self_disclosure_score", "self_disclosure_reason",
        # Embedding-derived
        "cluster_label", "umap_x", "umap_y",
        # Composite
        "depression_signal_score",
    ]
    df_out = df[cols]
    df_out.to_csv(OUT_FILE, index=False)
    log.info(f"Wrote {len(df_out)} rows → {OUT_FILE}")

    # ── Summary by category ──────────────────────────────────────────────────
    log.info("─" * 70)
    log.info("Per-category summary")
    summary = df.groupby("category").agg(
        n=("comment_id", "count"),
        avg_sent_neg=("sent_neg", "mean"),
        lex_hit_rate=("lex_total_hits", lambda s: (s > 0).mean()),
        self_disclosure_rate=("self_disclosure_flag", "mean"),
        avg_signal=("depression_signal_score", "mean"),
    ).round(4)
    log.info("\n" + summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N randomly sampled comments (debugging)")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip embeddings + HDBSCAN (faster smoke test)")
    args = parser.parse_args()
    main(sample=args.sample, do_cluster=not args.no_cluster)
