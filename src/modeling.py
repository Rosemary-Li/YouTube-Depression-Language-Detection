from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT          = Path(__file__).resolve().parents[1]
COMMENTS_CSV  = ROOT / "data" / "processed" / "comment_features.csv"
OUT_DIR       = ROOT / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("modeling")

RANDOM_STATE = 42
N_FOLDS = 5


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "NaiveBayes": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", MultinomialNB()),
        ]),
        "LinearSVM": Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LinearSVC(
                class_weight="balanced", random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_model(name: str, pipe: Pipeline, X, y, cv) -> dict:

    log.info("[%s] running %d-fold CV ...", name, N_FOLDS)
    f1_scores  = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    prec = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, digits=3, zero_division=0)

    log.info("[%s] f1_macro = %.4f ± %.4f | accuracy = %.4f",
             name, f1_scores.mean(), f1_scores.std(), acc_scores.mean())
    log.info("[%s] positive class -- precision=%.3f recall=%.3f f1=%.3f",
             name, prec, rec, f1_pos)

    return {
        "model": name,
        "f1_macro_mean":   float(f1_scores.mean()),
        "f1_macro_std":    float(f1_scores.std()),
        "f1_macro_folds":  [float(s) for s in f1_scores],
        "accuracy_mean":   float(acc_scores.mean()),
        "accuracy_std":    float(acc_scores.std()),
        "precision_pos":   float(prec),
        "recall_pos":      float(rec),
        "f1_pos":          float(f1_pos),
        "confusion_matrix": cm,
        "classification_report": report,
        "cv_predictions":  y_pred.tolist(),
    }


def plot_model_comparison(results: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["model"] for r in results]
    means = [r["f1_macro_mean"] for r in results]
    stds  = [r["f1_macro_std"]  for r in results]
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=["#7f77dd", "#ef9f27", "#3aa66a"])
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                f"{m:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("F1-macro (5-fold CV)")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison: depression-signal classification")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  -> %s", out_path)


def plot_confusion_matrices(results: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
            ax=ax, cbar=False,
        )
        ax.set_title(f"{r['model']}\nF1-macro={r['f1_macro_mean']:.3f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  -> %s", out_path)


def export_top_features(pipe: Pipeline, X, y, out_path: Path, top_n: int = 20) -> None:
    pipe.fit(X, y)
    vec  = pipe.named_steps["tfidf"]
    clf  = pipe.named_steps["clf"]
    feats = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()
    order = np.argsort(coefs)
    top_neg = feats[order[:top_n]]
    top_pos = feats[order[-top_n:][::-1]]
    df = pd.DataFrame({
        "rank": range(1, top_n + 1),
        "top_positive_term": top_pos,
        "pos_coef": coefs[order[-top_n:][::-1]],
        "top_negative_term": top_neg,
        "neg_coef": coefs[order[:top_n]],
    })
    df.to_csv(out_path, index=False)
    log.info("  -> %s", out_path)


def main() -> None:
    sns.set_context("notebook"); sns.set_style("whitegrid")

    log.info("Loading %s", COMMENTS_CSV)
    df = pd.read_csv(COMMENTS_CSV)
    df = df.dropna(subset=["text", "self_disclosure_flag"]).reset_index(drop=True)
    log.info("  %d rows", len(df))

    X = df["text"].astype(str).values
    y = df["self_disclosure_flag"].astype(int).values
    pos_rate = y.mean()
    log.info("Label = self_disclosure_flag | positives = %d / %d (%.2f%%)",
             y.sum(), len(y), pos_rate * 100)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    results: list[dict] = []
    for name, pipe in pipelines.items():
        results.append(evaluate_model(name, pipe, X, y, cv))

    #save outputs
    plot_model_comparison(results, OUT_DIR / "model_comparison.png")
    plot_confusion_matrices(results, OUT_DIR / "confusion_matrices.png")

    #top features
    log.info("Extracting top features from LogisticRegression ...")
    export_top_features(pipelines["LogisticRegression"], X, y,
                        OUT_DIR / "top_features.csv")

    # write JSON
    json_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "cv_predictions"}
        json_results.append(r_copy)
    summary = {
        "n_samples":       int(len(y)),
        "positive_rate":   float(pos_rate),
        "label":           "self_disclosure_flag",
        "n_folds":         N_FOLDS,
        "scoring":         "f1_macro",
        "models":          json_results,
    }
    with open(OUT_DIR / "model_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("  -> %s", OUT_DIR / "model_results.json")

if __name__ == "__main__":
    main()
