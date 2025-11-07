"""Utilities for preparing the TwiBot-22 heterogeneous graph dataset."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import ijson
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

# Mapping between raw relation names and (source_node_type, relation_name, target_node_type)
RELATION_SCHEMA: Mapping[str, Tuple[str, str, str]] = {
    "followers": ("user", "followers", "user"),
    "following": ("user", "following", "user"),
    "followed": ("list", "followed", "user"),
    "membership": ("list", "membership", "user"),
    "own": ("user", "own", "list"),
    "pinned": ("user", "pinned", "tweet"),
    "post": ("user", "post", "tweet"),
    "contain": ("list", "contain", "tweet"),
    "discuss": ("tweet", "discuss", "hashtag"),
    "mentioned": ("tweet", "mentioned", "user"),
    "like": ("user", "like", "tweet"),
    "replied_to": ("tweet", "replied_to", "tweet"),
    "retweeted": ("tweet", "retweeted", "tweet"),
    "quoted": ("tweet", "quoted", "tweet"),
}

NODE_PREFIX_TO_TYPE = {
    "u": "user",
    "t": "tweet",
    "h": "hashtag",
    "l": "list",
}

RAW_USER_FILENAME = "user.json"
RAW_HASHTAG_FILENAME = "hashtag.json"
RAW_LIST_FILENAME = "list.json"
RAW_TWEET_PATTERN = "tweet_{}.json"
RAW_SPLIT_FILENAME = "split.csv"
RAW_LABEL_FILENAME = "label.csv"
RAW_EDGE_FILENAME = "edge.csv"

DEFAULT_REFERENCE_DATETIME = datetime(2022, 7, 1, tzinfo=timezone.utc)


@dataclass
class PreprocessedDataPaths:
    """Helper dataclass that stores the processed artifact paths."""

    graph_pt: Path
    user_ids_json: Path
    profile_features_pt: Path
    profile_scaler_json: Path
    labels_pt: Path
    splits_pt: Path
    texts_jsonl: Path


def _iter_json_array(path: Path) -> Iterable[Mapping[str, object]]:
    """Stream items from a JSON array without loading it entirely."""

    with path.open("rb") as handle:
        for obj in ijson.items(handle, "item"):
            yield obj


def _coerce_bool(value: Optional[bool]) -> float:
    return float(bool(value))


def _safe_log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _account_age_days(created_at: Optional[str], reference: datetime = DEFAULT_REFERENCE_DATETIME) -> float:
    """Return the non-negative day difference between reference and created_at.

    Ensures both datetimes are timezone-aware (UTC). If either input is
    offset-naive, assume UTC.
    """
    if not created_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0

    # Normalize both to timezone-aware UTC for safe subtraction.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    if reference.tzinfo is None:
        ref = reference.replace(tzinfo=timezone.utc)
    else:
        ref = reference.astimezone(timezone.utc)

    delta = ref - dt
    return max(delta.total_seconds() / 86400.0, 0.0)

class TwiBot22GraphBuilder:
    """Preprocesses the raw TwiBot-22 dump into tensors consumable by PyG."""

    def __init__(
        self,
        raw_root: Path,
        processed_root: Path,
        max_tweets_per_user: int = 8,
        text_concat_separator: str = " \n",
    ) -> None:
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.max_tweets_per_user = max_tweets_per_user
        self.text_concat_separator = text_concat_separator

        self.processed_root.mkdir(parents=True, exist_ok=True)

        self._user_id_to_index: Dict[str, int] = {}
        self._tweet_id_to_index: Dict[str, int] = {}
        self._hashtag_id_to_index: Dict[str, int] = {}
        self._list_id_to_index: Dict[str, int] = {}

        self._user_numeric_features: List[List[float]] = []
        self._user_binary_features: List[List[float]] = []
        self._user_text_fragments: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
        self._user_bio: Dict[int, str] = {}
        self._user_metadata: Dict[int, Dict[str, object]] = {}

        self._labels: Dict[int, int] = {}
        self._split: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Runs the end-to-end preprocessing pipeline."""

        self._index_users()
        self._index_auxiliary_nodes()
        self._load_labels_and_splits()
        self._collect_tweets()
        texts = self._materialize_user_texts()
        profile_matrix, scaler_stats = self._finalize_profile_features()
        graph = self._build_graph()
        self._persist_artifacts(graph, texts, profile_matrix, scaler_stats)

    def get_processed_paths(self) -> PreprocessedDataPaths:
        return PreprocessedDataPaths(
            graph_pt=self.processed_root / "hetero_graph.pt",
            user_ids_json=self.processed_root / "user_ids.json",
            profile_features_pt=self.processed_root / "profile_features.pt",
            profile_scaler_json=self.processed_root / "profile_feature_scaler.json",
            labels_pt=self.processed_root / "user_labels.pt",
            splits_pt=self.processed_root / "user_splits.pt",
            texts_jsonl=self.processed_root / "user_texts.jsonl",
        )

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _index_users(self) -> None:
        user_path = self.raw_root / RAW_USER_FILENAME
        for idx, record in enumerate(_iter_json_array(user_path)):
            user_id: str = record["id"]  # type: ignore[index]
            self._user_id_to_index[user_id] = idx
            self._user_metadata[idx] = {
                "username": record.get("username"),
                "name": record.get("name"),
                "created_at": record.get("created_at"),
                "protected": bool(record.get("protected")),
                "verified": bool(record.get("verified")),
                "url": record.get("url") or "",
                "location": record.get("location") or "",
                "description": record.get("description") or "",
                "pinned_tweet_id": record.get("pinned_tweet_id"),
                "profile_image_url": record.get("profile_image_url"),
            }
            self._user_bio[idx] = str(record.get("description") or "")

            public_metrics = record.get("public_metrics") or {}
            followers_count = float(public_metrics.get("followers_count", 0))
            following_count = float(public_metrics.get("following_count", 0))
            tweet_count = float(public_metrics.get("tweet_count", 0))
            listed_count = float(public_metrics.get("listed_count", 0))
            account_age_days = _account_age_days(record.get("created_at"))
            tweets_per_day = tweet_count / account_age_days if account_age_days > 0 else 0.0
            follower_following_ratio = (
                followers_count / following_count if following_count > 0 else followers_count
            )
            bio_length = float(len(self._user_bio[idx]))

            numeric_features = [
                followers_count,
                following_count,
                tweet_count,
                listed_count,
                account_age_days,
                tweets_per_day,
                follower_following_ratio,
                bio_length,
            ]
            binary_features = [
                _coerce_bool(public_metrics.get("followers_count", 0) > 0),
                _coerce_bool(record.get("protected")),
                _coerce_bool(record.get("verified")),
                _coerce_bool(bool(record.get("url"))),
                _coerce_bool(bool(record.get("location"))),
                _coerce_bool(bool(record.get("pinned_tweet_id"))),
                _coerce_bool(bool(self._user_bio[idx])),
                _coerce_bool(record.get("profile_image_url") and "default_profile" not in record.get("profile_image_url", "")),
            ]

            self._user_numeric_features.append(numeric_features)
            self._user_binary_features.append(binary_features)

    def _index_auxiliary_nodes(self) -> None:
        hashtag_path = self.raw_root / RAW_HASHTAG_FILENAME
        for idx, record in enumerate(_iter_json_array(hashtag_path)):
            self._hashtag_id_to_index[record["id"]] = idx  # type: ignore[index]

        list_path = self.raw_root / RAW_LIST_FILENAME
        for idx, record in enumerate(_iter_json_array(list_path)):
            self._list_id_to_index[record["id"]] = idx  # type: ignore[index]

        tweet_files = sorted(self.raw_root.glob("tweet_*.json"))
        running_idx = 0
        for path in tweet_files:
            for record in _iter_json_array(path):
                self._tweet_id_to_index[record["id"]] = running_idx  # type: ignore[index]
                running_idx += 1

    def _load_labels_and_splits(self) -> None:
        label_path = self.raw_root / RAW_LABEL_FILENAME
        label_mapping = {"human": 0, "bot": 1}
        with label_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                user_id = row["id"]
                if user_id not in self._user_id_to_index:
                    continue
                label_str = row["label"].strip().lower()
                if label_str not in label_mapping:
                    continue
                self._labels[self._user_id_to_index[user_id]] = label_mapping[label_str]

        split_path = self.raw_root / RAW_SPLIT_FILENAME
        with split_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                user_id = row["id"]
                split_name = row["split"].strip().lower()
                if user_id not in self._user_id_to_index:
                    continue
                self._split[self._user_id_to_index[user_id]] = split_name

    def _collect_tweets(self) -> None:
        tweet_files = sorted(self.raw_root.glob("tweet_*.json"))
        for path in tweet_files:
            for record in _iter_json_array(path):
                author_id = record.get("author_id")
                if author_id is None or author_id not in self._user_id_to_index:
                    continue
                user_idx = self._user_id_to_index[author_id]
                created_at = record.get("created_at")
                # Use POSIX timestamp to sort tweets by recency (newest first)
                timestamp = 0.0
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        timestamp = dt.timestamp()
                    except Exception:
                        timestamp = 0.0
                text = str(record.get("text") or "")
                heap = self._user_text_fragments[user_idx]
                entry = (timestamp, text)
                if len(heap) < self.max_tweets_per_user:
                    heap.append(entry)
                    heap.sort(key=lambda x: x[0], reverse=True)
                else:
                    if entry[0] > heap[-1][0]:
                        heap[-1] = entry
                        heap.sort(key=lambda x: x[0], reverse=True)

    def _materialize_user_texts(self) -> List[str]:
        texts: List[str] = []
        for user_idx in range(len(self._user_id_to_index)):
            pieces: List[str] = []
            bio = self._user_bio.get(user_idx, "").strip()
            if bio:
                pieces.append(bio)
            tweets = [t for _, t in sorted(self._user_text_fragments.get(user_idx, []), key=lambda x: x[0], reverse=True)]
            if tweets:
                pieces.extend(tweets)
            combined = self.text_concat_separator.join(pieces)
            texts.append(combined)
        return texts

    def _finalize_profile_features(self) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        numeric = np.array(self._user_numeric_features, dtype=np.float32)
        binary = np.array(self._user_binary_features, dtype=np.float32)
        numeric_log = np.vectorize(_safe_log1p)(numeric)

        train_indices = [idx for idx, split in self._split.items() if split == "train"]
        scaler = StandardScaler()
        if train_indices:
            scaler.fit(numeric_log[train_indices, :])
        else:
            scaler.fit(numeric_log)
        numeric_scaled = scaler.transform(numeric_log)
        combined = np.concatenate([numeric_scaled, binary], axis=1)
        profile_tensor = torch.from_numpy(combined).to(torch.float32)

        scaler_stats = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
        return profile_tensor, scaler_stats

    def _build_graph(self) -> HeteroData:
        node_counts = {
            "user": len(self._user_id_to_index),
            "tweet": len(self._tweet_id_to_index),
            "hashtag": len(self._hashtag_id_to_index),
            "list": len(self._list_id_to_index),
        }
        data = HeteroData()
        for node_type, count in node_counts.items():
            data[node_type].num_nodes = count

        edge_offsets: Dict[str, int] = {rel: 0 for rel in RELATION_SCHEMA.keys()}
        edge_counts: Dict[str, int] = {rel: 0 for rel in RELATION_SCHEMA.keys()}
        with (self.raw_root / RAW_EDGE_FILENAME).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                edge_counts[row["relation"]] += 1
        # Pre-allocate arrays per relation for memory efficiency.
        edge_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for relation, count in edge_counts.items():
            edge_arrays[relation] = (
                np.empty(count, dtype=np.int64),
                np.empty(count, dtype=np.int64),
            )

        with (self.raw_root / RAW_EDGE_FILENAME).open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relation = row["relation"]
                if relation not in RELATION_SCHEMA:
                    continue
                src_type, rel_name, dst_type = RELATION_SCHEMA[relation]
                src_map = self._get_id_map_for_type(src_type)
                dst_map = self._get_id_map_for_type(dst_type)
                if row["source_id"] not in src_map or row["target_id"] not in dst_map:
                    continue
                src_idx = src_map[row["source_id"]]
                dst_idx = dst_map[row["target_id"]]
                pos = edge_offsets[relation]
                edge_arrays[relation][0][pos] = src_idx
                edge_arrays[relation][1][pos] = dst_idx
                edge_offsets[relation] += 1

        for relation, (src_type, rel_name, dst_type) in RELATION_SCHEMA.items():
            src_array, dst_array = edge_arrays[relation]
            edge_index = torch.from_numpy(
                np.stack([src_array, dst_array], axis=0)
            ).to(torch.long)
            data[(src_type, rel_name, dst_type)].edge_index = edge_index

        return data

    def _persist_artifacts(
        self,
        graph: HeteroData,
        texts: Sequence[str],
        profile_features: torch.Tensor,
        scaler_stats: Mapping[str, Sequence[float]],
    ) -> None:
        paths = self.get_processed_paths()
        torch.save(graph, paths.graph_pt)
        torch.save(profile_features, paths.profile_features_pt)

        with paths.profile_scaler_json.open("w", encoding="utf-8") as handle:
            json.dump({k: list(v) for k, v in scaler_stats.items()}, handle)

        user_ids: List[str] = [""] * len(self._user_id_to_index)
        for user_id, idx in self._user_id_to_index.items():
            user_ids[idx] = user_id
        with paths.user_ids_json.open("w", encoding="utf-8") as handle:
            json.dump(user_ids, handle)

        labels = torch.full((len(user_ids),), fill_value=-1, dtype=torch.long)
        for idx, value in self._labels.items():
            labels[idx] = value
        torch.save(labels, paths.labels_pt)

        split_tensor = torch.full((len(user_ids),), fill_value=-1, dtype=torch.long)
        # Support both 'val' and 'valid' naming conventions
        split_lookup = {"train": 0, "valid": 1, "val": 1, "test": 2}
        for idx, split_name in self._split.items():
            if split_name in split_lookup:
                split_tensor[idx] = split_lookup[split_name]
        torch.save(split_tensor, paths.splits_pt)

        with paths.texts_jsonl.open("w", encoding="utf-8") as handle:
            for user_idx, text in enumerate(texts):
                payload = {
                    "user_idx": user_idx,
                    "text": text,
                }
                handle.write(json.dumps(payload) + "\n")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_id_map_for_type(self, node_type: str) -> Mapping[str, int]:
        if node_type == "user":
            return self._user_id_to_index
        if node_type == "tweet":
            return self._tweet_id_to_index
        if node_type == "hashtag":
            return self._hashtag_id_to_index
        if node_type == "list":
            return self._list_id_to_index
        raise KeyError(f"Unknown node type: {node_type}")


class UserTextStore:
    """Stores user-level text and provides tokenization batches on demand."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        self._texts = texts
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = device

    def fetch(self, indices: Sequence[int]) -> Mapping[str, torch.Tensor]:
        batch_texts = [self._texts[idx] if idx < len(self._texts) else "" for idx in indices]
        encoding = self._tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        if self._device is not None:
            encoding = {k: v.to(self._device) for k, v in encoding.items()}
        return encoding


class ProfileFeatureStore:
    """Simple tensor-backed lookup for profile features."""

    def __init__(self, features: torch.Tensor, device: Optional[torch.device] = None) -> None:
        self._tensor = features
        self._device = device

    def fetch(self, indices: Sequence[int]) -> torch.Tensor:
        feat = self._tensor[indices]
        if self._device is not None:
            feat = feat.to(self._device)
        return feat


class TwiBot22DataModule:
    """High-level convenience wrapper for loading all processed artifacts."""

    def __init__(
        self,
        raw_root: Path,
        processed_root: Path,
        max_tweets_per_user: int = 8,
        tokenizer=None,
        text_max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.builder = TwiBot22GraphBuilder(raw_root, processed_root, max_tweets_per_user)
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.device = device

        self.graph: Optional[HeteroData] = None
        self.user_ids: Optional[List[str]] = None
        self.text_store: Optional[UserTextStore] = None
        self.profile_store: Optional[ProfileFeatureStore] = None
        self.labels: Optional[torch.Tensor] = None
        self.splits: Optional[torch.Tensor] = None

    def prepare_data(self, rebuild: bool = False) -> None:
        paths = self.builder.get_processed_paths()
        if rebuild or not paths.graph_pt.exists():
            self.builder.build()

    def setup(self) -> None:
        paths = self.builder.get_processed_paths()
        # Explicitly disable weights-only loading for application-owned artifacts
        # for compatibility with PyTorch 2.6+ where weights_only defaults to True.
        # Fall back to the old signature on older PyTorch versions.
        def _torch_load_compat(p: Path):
            try:
                return torch.load(p, weights_only=False)
            except TypeError:
                return torch.load(p)

        self.graph = _torch_load_compat(paths.graph_pt)
        with paths.user_ids_json.open("r", encoding="utf-8") as handle:
            self.user_ids = json.load(handle)
        texts = self._load_texts(paths.texts_jsonl)
        if self.tokenizer is not None:
            self.text_store = UserTextStore(texts, tokenizer=self.tokenizer, max_length=self.text_max_length, device=self.device)
        profile_features = _torch_load_compat(paths.profile_features_pt)
        self.profile_store = ProfileFeatureStore(profile_features, device=self.device)
        self.labels = _torch_load_compat(paths.labels_pt)
        self.splits = _torch_load_compat(paths.splits_pt)

    def _load_texts(self, path: Path) -> List[str]:
        texts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                idx = payload["user_idx"]
                text = payload["text"]
                if idx == len(texts):
                    texts.append(text)
                elif idx < len(texts):
                    texts[idx] = text
                else:
                    # Fill missing entries if needed.
                    texts.extend([""] * (idx - len(texts)))
                    texts.append(text)
        return texts

    # Convenience accessors -------------------------------------------------
    def get_split_indices(self, split: str) -> torch.Tensor:
        if self.splits is None:
            raise RuntimeError("Call setup() first")
        split_lookup = {"train": 0, "valid": 1, "val": 1, "test": 2}
        if split not in split_lookup:
            raise KeyError(f"Unknown split: {split}")
        split_value = split_lookup[split]
        return (self.splits == split_value).nonzero(as_tuple=False).view(-1)
