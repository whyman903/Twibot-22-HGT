"""
Exploratory data analysis pipeline for the TwiBot-22 dataset.

The script samples users, edges, and tweets to keep memory usage reasonable,
computes descriptive statistics, generates visualizations, and produces
tweet embeddings using a transformer model for downstream modelling insights.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ijson
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas import DataFrame
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

matplotlib.use("Agg")

DATA_DIR = Path("TwiBot-22")
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Reference snapshot of data collection; close to the dataset timestamp.
DATA_CUTOFF = datetime(2022, 7, 1, tzinfo=timezone.utc)


@dataclass
class SampledDataset:
    data: DataFrame
    counts: Counter
    extras: Dict[str, Counter]


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_labels() -> Tuple[DataFrame, Dict[str, str]]:
    label_path = DATA_DIR / "label.csv"
    labels_df = pd.read_csv(label_path)
    labels_df = labels_df[labels_df["label"].isin(["human", "bot"])]
    label_map = dict(zip(labels_df["id"], labels_df["label"]))
    return labels_df, label_map


def write_table(df: DataFrame, path: Path) -> Path:
    """Persist dataframe using parquet when available; otherwise fall back to CSV."""
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ModuleNotFoundError, ValueError):
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(
            f"Parquet engine unavailable; wrote fallback CSV to {fallback.name}",
            flush=True,
        )
        return fallback


def process_user_data(
    label_map: Dict[str, str],
    sample_size: int = 20000,
) -> SampledDataset:
    user_path = DATA_DIR / "user.json"
    counts: Counter = Counter()
    extras: Dict[str, Counter] = {
        "verified": Counter(),
        "location": Counter(),
        "protected": Counter(),
        "has_url": Counter(),
    }
    reservoir: List[dict] = []

    with user_path.open("rb") as fh:
        iterator = ijson.items(fh, "item")
        for idx, user in enumerate(iterator):
            user_id = user.get("id")
            label = label_map.get(user_id)
            if label is None:
                continue

            counts[label] += 1
            public_metrics = user.get("public_metrics") or {}

            followers = public_metrics.get("followers_count")
            following = public_metrics.get("following_count")
            tweets = public_metrics.get("tweet_count")
            listed = public_metrics.get("listed_count")

            created_at = _parse_datetime(user.get("created_at"))
            account_age_days: float | None = None
            if created_at is not None:
                delta = DATA_CUTOFF - created_at
                account_age_days = max(delta.days, 0)

            has_url = bool(user.get("url")) or bool(
                (user.get("entities") or {}).get("url")
            )

            extras["verified"][label] += int(user.get("verified", False))
            extras["location"][label] += int(bool(user.get("location")))
            extras["protected"][label] += int(bool(user.get("protected")))
            extras["has_url"][label] += int(has_url)

            features = {
                "user_id": user_id,
                "label": label,
                "followers_count": followers,
                "following_count": following,
                "tweet_count": tweets,
                "listed_count": listed,
                "account_age_days": account_age_days,
                "description_length": len(user.get("description") or ""),
                "name_length": len(user.get("name") or ""),
                "username_length": len(user.get("username") or ""),
                "verified": bool(user.get("verified", False)),
                "protected": bool(user.get("protected", False)),
                "has_location": bool(user.get("location")),
                "has_url": has_url,
            }

            if following not in (None, 0):
                features["followers_to_following"] = (
                    followers / following if followers is not None else np.nan
                )
            else:
                features["followers_to_following"] = np.nan

            if account_age_days and account_age_days > 0:
                features["tweets_per_day"] = (
                    tweets / account_age_days if tweets is not None else np.nan
                )
            else:
                features["tweets_per_day"] = np.nan

            if len(reservoir) < sample_size:
                reservoir.append(features)
            else:
                j = random.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = features

    user_df = pd.DataFrame(reservoir)
    return SampledDataset(user_df, counts, extras)


def generate_user_figures(user_df: DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=user_df, x="label", ax=ax, palette="Set2")
    ax.set_title("Sampled User Count by Label")
    ax.set_xlabel("")
    ax.set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "label_distribution.png", dpi=200)
    plt.close(fig)

    melt_cols = [
        "followers_count",
        "following_count",
        "tweet_count",
        "account_age_days",
    ]
    long_df = user_df.melt(
        id_vars=["label"], value_vars=melt_cols, var_name="metric", value_name="value"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxenplot(
        data=long_df,
        x="metric",
        y=np.log10(long_df["value"] + 1),
        hue="label",
        ax=ax,
    )
    ax.set_ylabel("log10(value + 1)")
    ax.set_xlabel("User Metric")
    ax.set_title("User Activity Metrics (log scale)")
    ax.legend(title="Label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "user_metrics_log.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=user_df,
        x="tweets_per_day",
        hue="label",
        ax=ax,
        element="step",
        stat="density",
        common_norm=False,
        bins=40,
    )
    ax.set_xlim(0, np.nanquantile(user_df["tweets_per_day"], 0.99))
    ax.set_title("Tweet Frequency Distribution")
    ax.set_xlabel("Tweets per day")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tweet_frequency_density.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=user_df,
        x="label",
        y="description_length",
        ax=ax,
        inner="quart",
        palette="Set3",
    )
    ax.set_title("Profile Description Length by Label")
    ax.set_xlabel("")
    ax.set_ylabel("Description characters")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "description_length_violin.png", dpi=200)
    plt.close(fig)


def sample_edges(label_map: Dict[str, str], nrows: int = 1_000_000) -> DataFrame:
    edge_path = DATA_DIR / "edge.csv"
    usecols = ["source_id", "relation", "target_id"]
    edge_df = pd.read_csv(edge_path, usecols=usecols, nrows=nrows)
    edge_df = edge_df.dropna(subset=["source_id", "target_id"])
    edge_df["source_label"] = edge_df["source_id"].map(label_map)
    edge_df["target_label"] = edge_df["target_id"].map(label_map)
    mask = edge_df["source_label"].notna() & edge_df["target_label"].notna()
    return edge_df[mask]


def generate_edge_figures(edge_df: DataFrame) -> dict:
    result = {}
    sns.set_theme(style="whitegrid")

    relation_counts = edge_df["relation"].value_counts()
    result["relation_counts"] = relation_counts.to_dict()

    fig, ax = plt.subplots(figsize=(6, 4))
    relation_counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Edge Relation Types (sample)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "edge_relation_counts.png", dpi=200)
    plt.close(fig)

    out_degree = edge_df.groupby(["source_id", "source_label"]).size().reset_index(
        name="out_degree"
    )
    in_degree = edge_df.groupby(["target_id", "target_label"]).size().reset_index(
        name="in_degree"
    )

    edge_summary = {
        "sample_edges": int(len(edge_df)),
        "unique_sources": int(out_degree["source_id"].nunique()),
        "unique_targets": int(in_degree["target_id"].nunique()),
        "source_label_counts": out_degree["source_label"].value_counts().to_dict(),
        "target_label_counts": in_degree["target_label"].value_counts().to_dict(),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.ecdfplot(
        data=out_degree,
        x=np.log10(out_degree["out_degree"] + 1),
        hue="source_label",
        ax=ax,
    )
    ax.set_xlabel("log10(out-degree + 1)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Out-degree distribution by label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "out_degree_ecdf.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.ecdfplot(
        data=in_degree,
        x=np.log10(in_degree["in_degree"] + 1),
        hue="target_label",
        ax=ax,
    )
    ax.set_xlabel("log10(in-degree + 1)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("In-degree distribution by label")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "in_degree_ecdf.png", dpi=200)
    plt.close(fig)

    pair_counts = (
        edge_df.groupby(["source_label", "target_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    edge_summary["label_pair_counts"] = pair_counts.to_dict(orient="records")

    pivot = pair_counts.pivot(
        index="source_label", columns="target_label", values="count"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.log10(pivot + 1),
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "log10(count + 1)"},
        ax=ax,
    )
    ax.set_title("Connection patterns between labels (log scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "edge_label_heatmap.png", dpi=200)
    plt.close(fig)

    result.update(edge_summary)
    return result


def sample_tweets(
    label_map: Dict[str, str],
    sample_size: int = 12000,
    embedding_size: int = 600,
) -> Tuple[SampledDataset, DataFrame]:
    tweet_files = sorted(DATA_DIR.glob("tweet_*.json"))
    counts = Counter()
    reservoir: List[dict] = []
    embed_candidates: List[dict] = []
    lang_counts = Counter()

    total_seen = 0

    for tweet_file in tweet_files:
        with tweet_file.open("rb") as fh:
            iterator = ijson.items(fh, "item")
            for tweet in iterator:
                total_seen += 1
                author_id = tweet.get("author_id")
                if author_id is None:
                    continue
                user_key = f"u{author_id}"
                label = label_map.get(user_key)
                if label is None:
                    continue

                counts[label] += 1

                text = tweet.get("text") or ""
                entities = tweet.get("entities") or {}
                referenced = tweet.get("referenced_tweets") or []
                metrics = tweet.get("public_metrics") or {}

                created_at = _parse_datetime(tweet.get("created_at"))
                days_since = None
                if created_at is not None:
                    delta = DATA_CUTOFF - created_at
                    days_since = max(delta.days, 0)

                has_hashtags = bool(entities.get("hashtags"))
                has_url = bool(entities.get("urls"))
                has_media = bool(entities.get("media"))

                ref_types = {ref.get("type") for ref in referenced if ref}
                is_retweet = "retweeted" in ref_types or text.startswith("RT ")
                is_reply = "replied_to" in ref_types or tweet.get("in_reply_to_user_id")

                lang = tweet.get("lang") or "und"
                lang_counts[lang] += 1

                record = {
                    "tweet_id": tweet.get("id"),
                    "author_id": user_key,
                    "label": label,
                    "text": text,
                    "text_length": len(text),
                    "language": lang,
                    "has_hashtag": has_hashtags,
                    "has_url": has_url,
                    "has_media": has_media,
                    "is_retweet": is_retweet,
                    "is_reply": bool(is_reply),
                    "like_count": metrics.get("like_count"),
                    "retweet_count": metrics.get("retweet_count"),
                    "quote_count": metrics.get("quote_count"),
                    "reply_count": metrics.get("reply_count"),
                    "days_since_post": days_since,
                }

                if len(reservoir) < sample_size:
                    reservoir.append(record)
                else:
                    j = random.randint(0, total_seen - 1)
                    if j < sample_size:
                        reservoir[j] = record

                if len(embed_candidates) < embedding_size:
                    embed_candidates.append({"text": text, "label": label})
                else:
                    j = random.randint(0, total_seen - 1)
                    if j < embedding_size:
                        embed_candidates[j] = {"text": text, "label": label}

    tweet_df = pd.DataFrame(reservoir)
    extras = {"language": lang_counts}
    return SampledDataset(tweet_df, counts, extras), pd.DataFrame(embed_candidates)


def generate_tweet_figures(tweet_df: DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=tweet_df,
        x="text_length",
        hue="label",
        bins=60,
        element="step",
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_xlim(0, np.nanquantile(tweet_df["text_length"], 0.995))
    ax.set_title("Tweet text length distribution")
    ax.set_xlabel("Characters")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tweet_length_distribution.png", dpi=200)
    plt.close(fig)

    top_langs = (
        tweet_df["language"].value_counts().head(10).index.tolist()
        if not tweet_df.empty
        else []
    )
    if top_langs:
        filtered = tweet_df[tweet_df["language"].isin(top_langs)]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered,
            y="language",
            hue="label",
            order=top_langs,
            ax=ax,
        )
        ax.set_title("Top tweet languages in sample")
        ax.set_xlabel("Tweets")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "tweet_language_counts.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    engagement_cols = ["like_count", "retweet_count", "quote_count", "reply_count"]
    for label, marker, color in [
        ("human", "o", "#4C72B0"),
        ("bot", "s", "#DD8452"),
    ]:
        subset = tweet_df[tweet_df["label"] == label]
        if subset.empty:
            continue
        metrics = subset[engagement_cols].fillna(0)
        point = metrics.mean()
        ax.scatter(
            [point["like_count"]],
            [point["retweet_count"]],
            s=100,
            marker=marker,
            color=color,
            label=label,
        )
    ax.set_xlabel("Average likes")
    ax.set_ylabel("Average retweets")
    ax.set_title("Mean engagement per tweet by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tweet_engagement_scatter.png", dpi=200)
    plt.close(fig)


def compute_embeddings(embed_df: DataFrame) -> DataFrame:
    if embed_df.empty:
        return embed_df

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    texts = embed_df["text"].tolist()
    batch_size = 16
    embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            outputs = model(**encoded)
            pooled = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    embedding_matrix = np.vstack(embeddings)
    embed_df = embed_df.copy()
    embed_df["embedding"] = list(embedding_matrix)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(embedding_matrix)
    embed_df["pc1"] = coords[:, 0]
    embed_df["pc2"] = coords[:, 1]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"human": "#4C72B0", "bot": "#DD8452"}
    for label, color in palette.items():
        subset = embed_df[embed_df["label"] == label]
        ax.scatter(
            subset["pc1"],
            subset["pc2"],
            s=20,
            alpha=0.7,
            color=color,
            label=label,
        )
    ax.set_title("Tweet embeddings (DistilBERT + PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tweet_embedding_pca.png", dpi=200)
    plt.close(fig)

    return embed_df


def summarize_user_sample(sample: SampledDataset) -> dict:
    df = sample.data
    summary = {
        "sample_size": int(len(df)),
        "label_counts_sample": sample.data["label"].value_counts().to_dict(),
        "global_counts": sample.counts,
        "verified_rate": {
            label: sample.extras["verified"][label] / sample.counts[label]
            if sample.counts[label]
            else 0.0
            for label in sample.counts
        },
        "location_rate": {
            label: sample.extras["location"][label] / sample.counts[label]
            if sample.counts[label]
            else 0.0
            for label in sample.counts
        },
        "protected_rate": {
            label: sample.extras["protected"][label] / sample.counts[label]
            if sample.counts[label]
            else 0.0
            for label in sample.counts
        },
        "has_url_rate": {
            label: sample.extras["has_url"][label] / sample.counts[label]
            if sample.counts[label]
            else 0.0
            for label in sample.counts
        },
    }

    for metric in [
        "followers_count",
        "following_count",
        "tweet_count",
        "account_age_days",
        "tweets_per_day",
        "followers_to_following",
    ]:
        metric_stats = (
            df.groupby("label")[metric]
            .describe(percentiles=[0.5, 0.75, 0.9, 0.99])
            .to_dict()
        )
        summary.setdefault("metric_describe", {})[metric] = metric_stats
    return summary


def summarize_tweet_sample(sample: SampledDataset) -> dict:
    df = sample.data
    summary = {
        "sample_size": int(len(df)),
        "label_counts_sample": df["label"].value_counts().to_dict(),
        "language_top10": sample.extras["language"].most_common(10),
        "engagement_mean": df.groupby("label")[
            ["like_count", "retweet_count", "quote_count", "reply_count"]
        ]
        .mean()
        .fillna(0)
        .round(2)
        .to_dict(orient="index"),
    }
    for metric in ["text_length", "days_since_post"]:
        metric_stats = (
            df.groupby("label")[metric]
            .describe(percentiles=[0.5, 0.75, 0.9, 0.99])
            .to_dict()
        )
        summary.setdefault("metric_describe", {})[metric] = metric_stats
    return summary


def main() -> None:
    labels_df, label_map = load_labels()
    labels_df.to_csv(OUTPUT_DIR / "labels_clean.csv", index=False)

    user_sample = process_user_data(label_map=label_map, sample_size=20000)
    write_table(user_sample.data, OUTPUT_DIR / "user_sample.parquet")
    generate_user_figures(user_sample.data)

    edge_sample_df = sample_edges(label_map=label_map, nrows=1_000_000)
    write_table(edge_sample_df, OUTPUT_DIR / "edge_sample.parquet")
    edge_summary = generate_edge_figures(edge_sample_df)

    tweet_sample, embed_candidates = sample_tweets(
        label_map=label_map,
        sample_size=12000,
        embedding_size=600,
    )
    write_table(tweet_sample.data, OUTPUT_DIR / "tweet_sample.parquet")
    generate_tweet_figures(tweet_sample.data)

    embedding_df = compute_embeddings(embed_candidates)
    if not embedding_df.empty:
        write_table(embedding_df, OUTPUT_DIR / "tweet_embeddings.parquet")

    summary = {
        "label_counts": labels_df["label"].value_counts().to_dict(),
        "user_summary": summarize_user_sample(user_sample),
        "edge_summary": edge_summary,
        "tweet_summary": summarize_tweet_sample(tweet_sample),
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
