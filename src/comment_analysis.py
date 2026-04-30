import argparse
import html
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from afinn import Afinn
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT          = Path(__file__).resolve().parents[1]
COMMENTS_FILE = ROOT / "data" / "raw" / "comments.json"
OUT_DIR       = ROOT / "data" / "processed"
OUT_FILE      = OUT_DIR / "comment_features.csv"
EMB_FILE      = OUT_DIR / "comment_embeddings.npy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("comment_analysis")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL     = "en_core_web_sm"

# Depression lexicon — LIWC-aligned categories plus project-specific extensions.
# Each category contains both unigrams and multi-grams; multi-grams are picked
# up at the TF-IDF n-gram level (ngram_range=(1, 3)).
DEPRESSION_LEXICON: dict[str, set[str]] = {
    "sadness": {
        "sad", "sadness", "cry", "crying", "cried", "tears", "tearful",
        "weep", "weeping", "grief", "grieving", "mourn", "mourning",
        "sorrow", "miserable", "misery", "heartbroken", "depressed",
        "depressing", "depression", "lonely", "loneliness", "alone",
        "isolated", "isolation", "unhappy", "gloomy", "melancholy",
        "despair", "despondent",
        "feel sad", "feel lonely", "feeling lonely", "deeply sad",
    },
    "anxiety": {
        "anxious", "anxiety", "worried", "worry", "worrying", "nervous",
        "panic", "panicked", "panicking", "afraid", "scared", "fear",
        "fearful", "terrified", "stress", "stressed", "stressful",
        "overwhelm", "overwhelmed", "overwhelming", "tense", "uneasy",
        "restless", "dread", "dreading",
        "panic attack", "anxiety attack", "can't breathe", "cant breathe",
    },
    "negative_emotion": {
        "hate", "hated", "angry", "anger", "rage", "frustrated",
        "frustration", "annoyed", "irritated", "bitter", "resentful",
        "guilty", "guilt", "ashamed", "shame", "embarrassed", "humiliated",
        "regret", "regretful", "disgusted", "disgust", "awful", "terrible",
        "horrible", "pain", "painful", "hurt", "hurting", "suffering",
        "suffer",
        "hate myself", "self hate",
    },
    "health": {
        "sick", "ill", "illness", "diagnosed", "diagnosis", "therapy",
        "therapist", "psychiatrist", "psychologist", "counselor",
        "counseling", "medication", "meds", "antidepressant",
        "antidepressants", "prozac", "zoloft", "lexapro", "ssri",
        "prescription", "hospital", "hospitalized", "treatment", "psych",
        "ward", "clinical",
        "mental health", "mental illness",
    },
    "hopelessness": {
        "hopeless", "hopelessness", "worthless", "worthlessness",
        "pointless", "meaningless", "useless", "helpless", "trapped",
        "stuck", "lost", "broken", "defeated",
        "feel hopeless", "feel worthless", "no point", "no future",
        "no hope", "give up", "gave up", "giving up", "falling apart",
        "can't go on", "cant go on", "can't do this", "cant do this",
    },
    "exhaustion": {
        "exhausted", "exhaustion", "tired", "drained", "burnout",
        "fatigue", "fatigued", "weary", "spent", "depleted",
        "burned out", "burnt out", "no energy", "can't sleep", "cant sleep",
    },
    "numbness": {
        "numb", "numbness", "empty", "emptiness", "hollow", "void",
        "detached", "disconnected", "dissociate", "dissociated", "blank",
        "feel numb", "feel empty", "feel nothing",
    },
    "suicidal": {
        "suicide", "suicidal", "kms", "kys",
        "kill myself", "end my life", "end it all", "ending it",
        "want to die", "wanna die", "tired of living", "sick of living",
    },
}

# spaCy lemmatizes "me" → "I", so the lemma set covers I/me uniformly.
FIRST_PERSON_LEMMAS = {"i", "my", "myself", "mine"}

# Unigram set used for L3 first-person co-occurrence scan.
DISCLOSURE_VOCAB = {
    w for cat in ("sadness", "anxiety", "hopelessness",
                  "exhaustion", "numbness", "suicidal")
    for w in DEPRESSION_LEXICON[cat] if " " not in w
} | {"struggle", "struggled", "struggling", "struggles", "cope", "coping"}

CONTENT_POS = {"ADJ", "VERB", "NOUN"}

TOKEN_RE = re.compile(r"[a-zA-Z']+")
URL_RE   = re.compile(r"http\S+|www\.\S+")


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


# Layer 1 — Multi-tool sentiment scoring (VADER + TextBlob + AFINN)

def sentiment_features(
    text: str,
    vader: SentimentIntensityAnalyzer,
    afinn: Afinn,
) -> dict:
    if not text:
        return {
            "vader_compound": 0.0,
            "textblob_polarity": 0.0,
            "textblob_subjectivity": 0.0,
            "afinn_score": 0.0,
            "afinn_normalized": 0.0,
            "sent_neg_score": 0.0,
        }

    vc   = vader.polarity_scores(text)["compound"]
    blob = TextBlob(text)
    pol  = float(blob.sentiment.polarity)
    sub  = float(blob.sentiment.subjectivity)
    af   = float(afinn.score(text))
    wc   = max(len(tokenize(text)), 1)
    af_n = af / wc

    vader_neg = max(0.0, -vc)
    blob_neg  = max(0.0, -pol)
    afinn_neg = min(1.0, max(0.0, -af_n))
    sent_neg  = min(1.0, (vader_neg + blob_neg + afinn_neg + 0.5 * sub) / 3.5)

    return {
        "vader_compound":        round(vc, 4),
        "textblob_polarity":     round(pol, 4),
        "textblob_subjectivity": round(sub, 4),
        "afinn_score":           round(af, 4),
        "afinn_normalized":      round(af_n, 4),
        "sent_neg_score":        round(sent_neg, 4),
    }


# Layer 2 — TF-IDF weighted depression lexicon matching (locked vocab)

def build_lex_vocab() -> tuple[list[str], dict[str, str]]:
    """Flatten DEPRESSION_LEXICON into a TF-IDF vocabulary list and
    a term→category map for per-category aggregation."""
    vocab: list[str] = []
    term_cat: dict[str, str] = {}
    for cat, terms in DEPRESSION_LEXICON.items():
        for t in terms:
            if t not in term_cat:
                vocab.append(t)
                term_cat[t] = cat
    return vocab, term_cat


def _tfidf_tokenizer(s: str) -> list[str]:
    return TOKEN_RE.findall(s)


def fit_tfidf_lexicon(texts: list[str]):
    vocab, term_cat = build_lex_vocab()
    vectorizer = TfidfVectorizer(
        vocabulary=vocab,
        tokenizer=_tfidf_tokenizer,
        ngram_range=(1, 3),
        lowercase=True,
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts)
    return X, vocab, term_cat


def lexicon_features_from_tfidf(X, vocab: list[str], term_cat: dict[str, str]) -> list[dict]:
    cats = sorted(set(term_cat.values()))
    cat_idx = {
        c: np.array([i for i, w in enumerate(vocab) if term_cat[w] == c])
        for c in cats
    }
    X_arr = X.toarray()

    rows = []
    for row in X_arr:
        nz = row > 0
        total_tfidf = float(row.sum())
        max_tfidf   = float(row.max()) if row.size else 0.0
        n_unique    = int(nz.sum())
        if n_unique > 0:
            top = np.argsort(-row)[:8]
            top_terms = ";".join(vocab[j] for j in top if row[j] > 0)
        else:
            top_terms = ""

        feat = {
            "lex_tfidf_sum":    round(total_tfidf, 4),
            "lex_tfidf_max":    round(max_tfidf, 4),
            "lex_unique_terms": n_unique,
            "lex_top_terms":    top_terms,
        }
        for c in cats:
            idx = cat_idx[c]
            feat[f"lex_{c}"] = round(float(row[idx].sum()), 4) if idx.size else 0.0
        rows.append(feat)
    return rows


# Layer 3 — Self-disclosure detection (spaCy POS tagging)

def self_disclosure_from_doc(doc) -> dict:
    """First-person pronoun + depression-vocabulary co-occurrence,
    using spaCy POS + dependency tags so the pronoun is a genuine PRON
    (not e.g. capitalization noise) and the predicate is content-bearing
    (ADJ / VERB / NOUN — filters out "I think this video is great")."""
    fp_idx, fp_subj = [], False
    vocab_idx, vocab_terms = [], []

    for i, tok in enumerate(doc):
        if tok.pos_ == "PRON" and tok.lemma_.lower() in FIRST_PERSON_LEMMAS:
            fp_idx.append(i)
            if tok.dep_ in ("nsubj", "nsubjpass"):
                fp_subj = True
        if tok.pos_ in CONTENT_POS:
            lemma = tok.lemma_.lower()
            text  = tok.text.lower()
            if lemma in DISCLOSURE_VOCAB or text in DISCLOSURE_VOCAB:
                vocab_idx.append(i)
                vocab_terms.append(lemma)

    co_occur, co_occur_term = False, ""
    for i in fp_idx:
        for k, j in enumerate(vocab_idx):
            if abs(i - j) <= 6:
                co_occur, co_occur_term = True, vocab_terms[k]
                break
        if co_occur:
            break

    flag  = int(co_occur)
    score = (
        0.5 * int(co_occur)
        + 0.3 * int(fp_subj and co_occur)
        + 0.2 * int(len(vocab_idx) >= 2)
    )
    score = min(1.0, score)

    if fp_subj and co_occur:
        reason = f"PRON-subj + {co_occur_term}"
    elif co_occur:
        reason = f"PRON + {co_occur_term}"
    else:
        reason = ""

    return {
        "self_disclosure_flag":   flag,
        "self_disclosure_score":  round(score, 4),
        "self_disclosure_reason": reason,
    }



def text_stats(text: str) -> dict:
    tokens = tokenize(text)
    wc = len(tokens)
    return {
        "word_count":        wc,
        "lexical_diversity": round(len(set(tokens)) / wc, 4) if wc else 0.0,
    }

def run_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    log.info(f"Loading embedding model: {EMBEDDING_MODEL} ({device})")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

def composite_score(row) -> float:
    """L1 (broad) 0.20 + L2 (domain) 0.30 + L3 (precise) 0.50."""
    sent     = float(row["sent_neg_score"])
    lex_norm = min(1.0, float(row["lex_tfidf_sum"]))
    disc     = float(row["self_disclosure_score"])
    return 0.20 * sent + 0.30 * lex_norm + 0.50 * disc



def load_spacy() -> "spacy.language.Language":
    try:
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as e:
        raise SystemExit(
            f"spaCy model '{SPACY_MODEL}' not found. "
            f"Install with:  python -m spacy download {SPACY_MODEL}"
        ) from e


def main(sample: int | None = None, do_embed: bool = True) -> None:
    log.info(f"Loading comments from {COMMENTS_FILE}")
    with open(COMMENTS_FILE, encoding="utf-8") as f:
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

    # Layer 1 — multi-tool sentiment
    log.info("Layer 1 — multi-tool sentiment (VADER + TextBlob + AFINN)")
    vader = SentimentIntensityAnalyzer()
    afinn = Afinn()
    sent_rows = [
        sentiment_features(t, vader, afinn)
        for t in tqdm(texts, desc="layer1-sentiment")
    ]
    df = pd.concat([df, pd.DataFrame(sent_rows)], axis=1)

    # Layer 2 — TF-IDF lexicon (locked vocab, n-grams 1–3)
    log.info("Layer 2 — TF-IDF lexicon matching (locked vocab, n-grams 1–3)")
    X, vocab, term_cat = fit_tfidf_lexicon(texts)
    lex_rows = lexicon_features_from_tfidf(X, vocab, term_cat)
    df = pd.concat([df, pd.DataFrame(lex_rows)], axis=1)

    # Layer 3 — self-disclosure (spaCy POS)
    log.info(f"Layer 3 — self-disclosure detection (spaCy {SPACY_MODEL})")
    nlp = load_spacy()
    disc_rows = []
    for doc in tqdm(nlp.pipe(texts, batch_size=64),
                    total=len(texts), desc="layer3-disclosure"):
        disc_rows.append(self_disclosure_from_doc(doc))
    df = pd.concat([df, pd.DataFrame(disc_rows)], axis=1)

    # Additional text features
    df = pd.concat([df, pd.DataFrame([text_stats(t) for t in texts])], axis=1)

    # Composite
    df["depression_signal_score"] = df.apply(composite_score, axis=1)

    # Embeddings (clustering deferred to Step 3 / linking.py)
    if do_embed:
        log.info("Computing sentence embeddings (clustering done in linking.py)")
        emb = run_embeddings(texts)
        np.save(EMB_FILE, emb)
        log.info(f"Embeddings → {EMB_FILE}  shape={emb.shape}")

    cat_cols = [f"lex_{c}" for c in DEPRESSION_LEXICON]
    cols = [
        "comment_id", "video_id", "category", "category_label",
        "text", "like_count", "published_at",
        # Layer 1
        "vader_compound", "textblob_polarity", "textblob_subjectivity",
        "afinn_score", "afinn_normalized", "sent_neg_score",
        # Layer 2
        "lex_tfidf_sum", "lex_tfidf_max", "lex_unique_terms", "lex_top_terms",
        *cat_cols,
        # Layer 3
        "self_disclosure_flag", "self_disclosure_score", "self_disclosure_reason",
        # Additional text features
        "word_count", "lexical_diversity",
        # Composite
        "depression_signal_score",
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols]
    df_out.to_csv(OUT_FILE, index=False)
    log.info(f"Wrote {len(df_out)} rows → {OUT_FILE}")

    log.info("─" * 70)
    log.info("Per-category summary")
    summary = df.groupby("category").agg(
        n=("comment_id", "count"),
        avg_vader=("vader_compound", "mean"),
        lex_hit_rate=("lex_tfidf_sum", lambda s: (s > 0).mean()),
        self_disclosure_rate=("self_disclosure_flag", "mean"),
        avg_signal=("depression_signal_score", "mean"),
    ).round(4)
    log.info("\n" + summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2 — comment-level depression signal detection"
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N randomly sampled comments (debugging)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip sentence embeddings (faster smoke test)")
    args = parser.parse_args()
    main(sample=args.sample, do_embed=not args.no_embed)