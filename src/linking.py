from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


ROOT             = Path(__file__).resolve().parents[1]
PROCESSED_DIR    = ROOT / "data" / "processed"
COMMENTS_FEATS   = PROCESSED_DIR / "comment_features.csv"
COMMENTS_EMB     = PROCESSED_DIR / "comment_embeddings.npy"
VIDEO_FEATS_A    = PROCESSED_DIR / "video_features.csv"
VIDEO_FEATS_B    = PROCESSED_DIR / "video_features_KL.csv"

OUT_DIR          = ROOT / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("linking")

CATEGORY_ORDER = [
    "mental_health",
    "personal_vlog",
    "fitness_wellness",
    "news_current_events",
    "music_entertainment",
]
CATEGORY_PALETTE = {
    "mental_health":       "#7f77dd",
    "personal_vlog":       "#ef9f27",
    "fitness_wellness":    "#3aa66a",
    "news_current_events": "#d85a30",
    "music_entertainment": "#4a90c2",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Loading comment features: %s", COMMENTS_FEATS)
    comments = pd.read_csv(COMMENTS_FEATS)
    log.info("  %d comments × %d columns", *comments.shape)

    log.info("Loading video features (both teammates' files)")
    v_a = pd.read_csv(VIDEO_FEATS_A)
    v_b = pd.read_csv(VIDEO_FEATS_B)
    videos = pd.concat([v_a, v_b], ignore_index=True).drop_duplicates("video_id")
    log.info("  %d videos × %d columns", *videos.shape)
    cats_present = sorted(videos["category"].unique())
    log.info("  video categories: %s", cats_present)
    return comments, videos


def merge_video_context(comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "video_id", "title_sentiment", "desc_vader", "desc_polarity",
        "desc_subjectivity", "desc_word_count", "desc_lexical_diversity",
    ]
    merged = comments.merge(videos[keep], on="video_id", how="left", suffixes=("", "_v"))
    log.info("Merged → %d rows × %d columns", *merged.shape)
    return merged

def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    global_top10 = df["depression_signal_score"].quantile(0.90)
    g = df.groupby("category")
    summary = pd.DataFrame({
        "n_comments":             g.size(),
        "avg_signal":             g["depression_signal_score"].mean(),
        "median_signal":          g["depression_signal_score"].median(),
        "signal_rate_top10":      g["depression_signal_score"].apply(
                                       lambda s: (s >= global_top10).mean()),
        "avg_sent_neg":           g["sent_neg"].mean(),
        "avg_sent_pos":           g["sent_pos"].mean(),
        "lex_hit_rate":           g["lex_total_hits"].apply(lambda s: (s > 0).mean()),
        "self_disclosure_rate":   g["self_disclosure_flag"].mean(),
        "avg_sad":                g["lex_sadness"].mean(),
        "avg_anxiety":            g["lex_anxiety"].mean(),
        "avg_hopelessness":       g["lex_hopelessness"].mean(),
    }).round(4)
    summary = summary.reindex([c for c in CATEGORY_ORDER if c in summary.index])
    return summary


def plot_category_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    plot_cols = [
        "avg_signal", "signal_rate_top10",
        "avg_sent_neg", "lex_hit_rate", "self_disclosure_rate",
        "avg_sad", "avg_anxiety", "avg_hopelessness",
    ]
    data = summary[plot_cols]
    # Per-column z-score so colour scale is comparable across heterogeneous metrics
    z = (data - data.mean()) / data.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        z, annot=data.round(3), fmt=".3f", cmap="RdBu_r", center=0,
        cbar_kws={"label": "z-score across categories"}, ax=ax,
    )
    ax.set_title("Depression-signal metrics by video category (annot = raw value)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  → %s", out_path)


def plot_signal_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    for cat in cats:
        sns.kdeplot(
            df.loc[df["category"] == cat, "depression_signal_score"],
            ax=axes[0], label=cat, color=CATEGORY_PALETTE[cat],
            fill=True, alpha=0.18, linewidth=1.6, clip=(0, 1),
        )
    axes[0].set_title("Depression signal score — KDE by category")
    axes[0].set_xlabel("depression_signal_score")
    axes[0].legend(fontsize=8)

    sns.boxplot(
        data=df, x="category", y="depression_signal_score",
        order=cats, hue="category", legend=False,
        palette=[CATEGORY_PALETTE[c] for c in cats],
        ax=axes[1], showfliers=False,
    )
    axes[1].set_title("Depression signal score — boxplot by category")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_xlabel("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  → %s", out_path)


def run_statistical_tests(df: pd.DataFrame) -> dict:
    cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    groups = [df.loc[df["category"] == c, "depression_signal_score"].values for c in cats]

    # One-way ANOVA across all 5 categories
    f_stat, anova_p = stats.f_oneway(*groups)
    # Kruskal-Wallis as non-parametric robustness check (signal scores are skewed)
    h_stat, kw_p = stats.kruskal(*groups)

    # Pairwise Kolmogorov-Smirnov, two-sided
    ks_pairs = []
    for i, ci in enumerate(cats):
        for cj in cats[i + 1:]:
            ks_stat, ks_p = stats.ks_2samp(
                df.loc[df["category"] == ci, "depression_signal_score"],
                df.loc[df["category"] == cj, "depression_signal_score"],
            )
            ks_pairs.append({
                "cat_a": ci, "cat_b": cj,
                "ks_stat": float(ks_stat), "p_value": float(ks_p),
            })

    # Bonferroni correction for KS pairs
    m = len(ks_pairs)
    for r in ks_pairs:
        r["p_bonferroni"] = min(1.0, r["p_value"] * m)
        r["significant_alpha_0.05"] = bool(r["p_bonferroni"] < 0.05)

    return {
        "categories":      cats,
        "n_per_category":  {c: int((df["category"] == c).sum()) for c in cats},
        "anova":           {"F": float(f_stat), "p_value": float(anova_p)},
        "kruskal_wallis":  {"H": float(h_stat), "p_value": float(kw_p)},
        "pairwise_ks":     ks_pairs,
    }



# PCA → t-SNE
def reduce_pca_tsne(emb: np.ndarray, pca_dim: int = 50, seed: int = 42) -> np.ndarray:
    pca_dim = min(pca_dim, emb.shape[1], emb.shape[0])
    log.info("PCA → %d dims", pca_dim)
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(emb)
    log.info("  PCA explained variance (cumulative): %.3f", pca.explained_variance_ratio_.sum())

    log.info("t-SNE → 2D (perplexity=30)")
    perplexity = max(5.0, min(30.0, (len(emb) - 1) / 3))
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate="auto",
        init="pca", random_state=seed,
    )
    return tsne.fit_transform(reduced)


def plot_tsne(coords: np.ndarray, labels: pd.Series, out_path: Path,
              title: str, palette: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels.values})

    if palette is None:
        unique = sorted(pd.unique(df["label"]))
        cmap = sns.color_palette("tab20", n_colors=len(unique))
        palette = {lbl: cmap[i] for i, lbl in enumerate(unique)}

    for lbl in df["label"].drop_duplicates():
        mask = df["label"] == lbl
        ax.scatter(
            df.loc[mask, "x"], df.loc[mask, "y"],
            s=6, alpha=0.55, label=str(lbl),
            color=palette.get(lbl, "#888888"), linewidths=0,
        )
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  → %s", out_path)



# GMM with AIC/BIC selection
def fit_gmm_with_selection(reduced: np.ndarray,
                           k_range: range = range(2, 13),
                           seed: int = 42) -> tuple[GaussianMixture, pd.DataFrame]:
    log.info("GMM model selection over k ∈ [%d, %d]", k_range.start, k_range.stop - 1)
    rows = []
    best = (None, np.inf)
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=seed, max_iter=200, n_init=2,
        )
        gmm.fit(reduced)
        aic, bic = gmm.aic(reduced), gmm.bic(reduced)
        rows.append({"k": k, "AIC": aic, "BIC": bic})
        if bic < best[1]:
            best = (gmm, bic)
        log.info("  k=%2d  AIC=%.0f  BIC=%.0f", k, aic, bic)

    selection = pd.DataFrame(rows)
    return best[0], selection


def plot_gmm_selection(selection: pd.DataFrame, chosen_k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(selection["k"], selection["AIC"], "o-", label="AIC", color="#7f77dd")
    ax.plot(selection["k"], selection["BIC"], "s-", label="BIC", color="#d85a30")
    ax.axvline(chosen_k, ls="--", color="grey", alpha=0.6, label=f"chosen k = {chosen_k}")
    ax.set_xlabel("n_components"); ax.set_ylabel("information criterion")
    ax.set_title("GMM model selection"); ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  → %s", out_path)



# Cluster × category contingency
def plot_cluster_category(df: pd.DataFrame, cluster_col: str, out_path: Path,
                          title: str) -> None:
    cats = [c for c in CATEGORY_ORDER if c in df["category"].unique()]
    ct = pd.crosstab(df[cluster_col], df["category"])[cats]
    ct_norm = ct.div(ct.sum(axis=1), axis=0)  # row-normalize

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ct_norm))))
    sns.heatmap(
        ct_norm, annot=ct.values, fmt="d", cmap="Blues",
        cbar_kws={"label": "row-normalized share"}, ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("category"); ax.set_ylabel(cluster_col)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("  → %s", out_path)



# Main
def main(sample: int | None = None, do_embeddings: bool = True) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    comments, videos = load_data()
    df = merge_video_context(comments, videos)

    # 1. Descriptive comparison
    log.info("─" * 60)
    log.info("Descriptive cross-category summary")
    summary = category_summary(df)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "category_summary.csv")
    log.info("  → %s", OUT_DIR / "category_summary.csv")

    plot_category_heatmap(summary, OUT_DIR / "category_heatmap.png")
    plot_signal_distribution(df, OUT_DIR / "signal_distribution.png")

    # 2. Statistical tests
    log.info("─" * 60)
    log.info("Statistical tests on depression_signal_score across categories")
    test_results = run_statistical_tests(df)
    log.info("  ANOVA           F=%.3f  p=%.4g",
             test_results["anova"]["F"], test_results["anova"]["p_value"])
    log.info("  Kruskal-Wallis  H=%.3f  p=%.4g",
             test_results["kruskal_wallis"]["H"], test_results["kruskal_wallis"]["p_value"])
    n_sig = sum(r["significant_alpha_0.05"] for r in test_results["pairwise_ks"])
    log.info("  Pairwise KS: %d/%d pairs significant after Bonferroni",
             n_sig, len(test_results["pairwise_ks"]))
    with open(OUT_DIR / "stats_tests.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    log.info("  → %s", OUT_DIR / "stats_tests.json")

    # 3. Embedding-based: PCA→t-SNE + GMM
    if not do_embeddings:
        log.info("--no-embeddings flag set; skipping clustering / t-SNE.")
        return

    if not COMMENTS_EMB.exists():
        log.warning("Embeddings file not found: %s — skipping.", COMMENTS_EMB)
        return

    log.info("─" * 60)
    log.info("Loading embeddings: %s", COMMENTS_EMB)
    emb = np.load(COMMENTS_EMB)
    log.info("  shape=%s", emb.shape)

    work_df = df.reset_index(drop=True)
    if len(work_df) != len(emb):
        log.warning("Comments (%d) and embeddings (%d) length mismatch — "
                    "truncating to min.", len(work_df), len(emb))
        n = min(len(work_df), len(emb))
        work_df = work_df.iloc[:n].copy()
        emb = emb[:n]

    if sample and sample < len(work_df):
        idx = work_df.sample(sample, random_state=42).index.values
        work_df = work_df.loc[idx].reset_index(drop=True)
        emb = emb[idx]
        log.info("Sampled %d comments for clustering / t-SNE", len(work_df))

    # PCA to t-SNE
    coords = reduce_pca_tsne(emb)
    work_df["tsne_x"] = coords[:, 0]
    work_df["tsne_y"] = coords[:, 1]

    plot_tsne(
        coords, work_df["category"],
        OUT_DIR / "tsne_by_category.png",
        title="PCA(50) → t-SNE(2): comments by category",
        palette=CATEGORY_PALETTE,
    )

    # HDBSCAN labels    
    if "cluster_label" in work_df.columns:
        plot_tsne(
            coords, work_df["cluster_label"].astype(str),
            OUT_DIR / "tsne_by_cluster_hdbscan.png",
            title="PCA(50) → t-SNE(2): HDBSCAN clusters (-1 = noise)",
        )
        plot_cluster_category(
            work_df, "cluster_label",
            OUT_DIR / "cluster_category_heatmap.png",
            title="HDBSCAN cluster × category (annot = raw count, colour = row share)",
        )

    # GMM with AIC/BIC selection on the 50-D PCA space
    pca_dim = min(50, emb.shape[1], emb.shape[0])
    pca_reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(emb)
    gmm, selection = fit_gmm_with_selection(pca_reduced)
    chosen_k = int(gmm.n_components)
    log.info("Selected GMM k=%d (lowest BIC)", chosen_k)
    plot_gmm_selection(selection, chosen_k, OUT_DIR / "gmm_model_selection.png")
    selection.to_csv(OUT_DIR / "gmm_model_selection.csv", index=False)

    gmm_labels = gmm.predict(pca_reduced)
    work_df["gmm_label"] = gmm_labels
    # Keep the highest soft-assignment probability for inspection
    work_df["gmm_max_prob"] = gmm.predict_proba(pca_reduced).max(axis=1)

    plot_tsne(
        coords, pd.Series([f"g{l}" for l in gmm_labels]),
        OUT_DIR / "tsne_by_cluster_gmm.png",
        title=f"PCA(50) → t-SNE(2): GMM clusters (k={chosen_k}, BIC-selected)",
    )
    plot_cluster_category(
        work_df, "gmm_label",
        OUT_DIR / "gmm_category_heatmap.png",
        title="GMM cluster and category (annot = raw count, colour = row share)",
    )

    # Per-comment output
    keep = [
        "comment_id", "video_id", "category", "depression_signal_score",
        "self_disclosure_flag", "lex_total_hits", "sent_neg",
        "cluster_label", "gmm_label", "gmm_max_prob", "tsne_x", "tsne_y",
    ]
    keep = [c for c in keep if c in work_df.columns]
    work_df[keep].to_csv(OUT_DIR / "comment_with_clusters.csv", index=False)
    log.info("  → %s", OUT_DIR / "comment_with_clusters.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N comments before clustering / t-SNE (debug).")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip embedding-based steps (descriptive + stats only).")
    args = parser.parse_args()
    main(sample=args.sample, do_embeddings=not args.no_embeddings)
