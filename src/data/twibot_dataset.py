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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import ijson
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
    text_tokens_pt: Path
    tweet_text_embeddings_pt: Path


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
    """Return the non-negative day difference between reference and created_at."""
    if not created_at:
        return 0.0
    try:
        dt = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0

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
        max_tweets_per_user: int = -1,
        text_concat_separator: str = " \n",
        tweet_text_model_name: Optional[str] = None,
        tweet_text_max_length: int = 96,
        tweet_text_batch_size: int = 512,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.max_tweets_per_user = max_tweets_per_user
        self.text_concat_separator = text_concat_separator
        self.tweet_text_model_name = tweet_text_model_name
        self.tweet_text_max_length = tweet_text_max_length
        self.tweet_text_batch_size = tweet_text_batch_size

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

        self._tweet_text: Dict[int, str] = {}
        self._tweet_author_idx: Dict[int, int] = {}
        self._tweet_created_ts: Dict[int, float] = {}
        self._pending_replies: List[Tuple[int, int, int]] = []
        self._user_reply_sims: Dict[int, List[float]] = defaultdict(list)
        self._user_reply_lat_min: Dict[int, List[float]] = defaultdict(list)
        self._tweets_for_text_features: Set[int] = set()

    def build(self) -> None:
        """Runs the end-to-end preprocessing pipeline."""
        self._index_users()
        self._index_auxiliary_nodes()
        self._load_labels_and_splits()
        self._collect_tweets()
        self._compute_reply_aggregates()
        tweet_text_embeddings = self._encode_tweet_text_features()
        texts = self._materialize_user_texts()
        profile_matrix, scaler_stats = self._finalize_profile_features()
        graph = self._build_graph()
        self._persist_artifacts(graph, texts, profile_matrix, scaler_stats, tweet_text_embeddings)

    def get_processed_paths(self) -> PreprocessedDataPaths:
        return PreprocessedDataPaths(
            graph_pt=self.processed_root / "hetero_graph.pt",
            user_ids_json=self.processed_root / "user_ids.json",
            profile_features_pt=self.processed_root / "profile_features.pt",
            profile_scaler_json=self.processed_root / "profile_feature_scaler.json",
            labels_pt=self.processed_root / "user_labels.pt",
            splits_pt=self.processed_root / "user_splits.pt",
            texts_jsonl=self.processed_root / "user_texts.jsonl",
            text_tokens_pt=self.processed_root / "user_text_tokens.pt",
            tweet_text_embeddings_pt=self.processed_root / "tweet_text_embeddings.pt",
        )

    def _index_users(self) -> None:
        user_path = self.raw_root / RAW_USER_FILENAME
        for idx, record in enumerate(_iter_json_array(user_path)):
            user_id: str = record["id"]
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
            self._hashtag_id_to_index[record["id"]] = idx

        list_path = self.raw_root / RAW_LIST_FILENAME
        for idx, record in enumerate(_iter_json_array(list_path)):
            self._list_id_to_index[record["id"]] = idx

        tweet_files = sorted(self.raw_root.glob("tweet_*.json"))
        running_idx = 0
        for path in tweet_files:
            for record in _iter_json_array(path):
                self._tweet_id_to_index[record["id"]] = running_idx
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

                t_id = record.get("id")
                if t_id is not None and t_id in self._tweet_id_to_index:
                    t_idx = self._tweet_id_to_index[t_id]
                    self._tweet_author_idx[t_idx] = user_idx
                    self._tweet_created_ts[t_idx] = timestamp
                    self._tweet_text[t_idx] = text

                self._user_text_fragments[user_idx].append((timestamp, text))

                referenced = record.get("referenced_tweets") or []
                for ref in referenced:
                    try:
                        if not ref:
                            continue
                        ref_type = ref.get("type")
                        orig_id = ref.get("id")
                        if (
                            ref_type == "replied_to"
                            and orig_id is not None
                            and orig_id in self._tweet_id_to_index
                            and t_id is not None
                            and t_id in self._tweet_id_to_index
                        ):
                            reply_idx = self._tweet_id_to_index[t_id]
                            orig_idx = self._tweet_id_to_index[orig_id]
                            self._pending_replies.append((user_idx, reply_idx, orig_idx))
                            self._tweets_for_text_features.add(reply_idx)
                            self._tweets_for_text_features.add(orig_idx)
                        if (
                            ref_type in {"retweeted", "quoted"}
                            and orig_id is not None
                            and orig_id in self._tweet_id_to_index
                        ):
                            if t_id is not None and t_id in self._tweet_id_to_index:
                                self._tweets_for_text_features.add(self._tweet_id_to_index[t_id])
                            self._tweets_for_text_features.add(self._tweet_id_to_index[orig_id])
                    except Exception:
                        continue

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return float(inter / union) if union > 0 else 0.0

    def _compute_reply_aggregates(self) -> None:
        for user_idx, reply_idx, orig_idx in self._pending_replies:
            r_text = self._tweet_text.get(reply_idx, "")
            o_text = self._tweet_text.get(orig_idx, "")
            if not r_text or not o_text:
                continue
            sim = self._jaccard_similarity(r_text, o_text)
            r_ts = self._tweet_created_ts.get(reply_idx, None)
            o_ts = self._tweet_created_ts.get(orig_idx, None)
            if r_ts is None or o_ts is None:
                continue
            lat_min = max((r_ts - o_ts) / 60.0, 0.0)
            self._user_reply_sims[user_idx].append(sim)
            self._user_reply_lat_min[user_idx].append(lat_min)

    def _encode_tweet_text_features(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.tweet_text_model_name:
            return None

        target_indices = sorted(self._tweets_for_text_features)
        if not target_indices:
            return None

        tokenizer = AutoTokenizer.from_pretrained(self.tweet_text_model_name)
        config = AutoConfig.from_pretrained(self.tweet_text_model_name)
        if hasattr(config, "add_pooling_layer"):
            config.add_pooling_layer = False
        model = AutoModel.from_pretrained(self.tweet_text_model_name, config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        hidden = model.config.hidden_size
        storage = torch.empty((len(target_indices), hidden), dtype=torch.float16)

        batch_size = self.tweet_text_batch_size
        with torch.no_grad():
            for start in range(0, len(target_indices), batch_size):
                batch_indices = target_indices[start : start + batch_size]
                texts = [self._tweet_text.get(idx, "") for idx in batch_indices]
                tokens = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.tweet_text_max_length,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = model(**tokens)
                hidden_states = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                storage[start : start + len(batch_indices)] = pooled.detach().cpu().to(torch.float16)

        return {
            "indices": torch.tensor(target_indices, dtype=torch.long),
            "embeddings": storage,
        }

    def _materialize_user_texts(self) -> List[str]:
        texts: List[str] = []
        for user_idx in range(len(self._user_id_to_index)):
            pieces: List[str] = []
            bio = self._user_bio.get(user_idx, "").strip()
            if bio:
                pieces.append(bio)
            tweet_entries = self._user_text_fragments.get(user_idx, [])
            tweet_entries_sorted = sorted(tweet_entries, key=lambda x: x[0], reverse=True)
            if self.max_tweets_per_user is not None and self.max_tweets_per_user > 0:
                tweet_entries_sorted = tweet_entries_sorted[:self.max_tweets_per_user]
            tweets = [t for _, t in tweet_entries_sorted]
            if tweets:
                pieces.extend(tweets)
            combined = self.text_concat_separator.join(pieces)
            texts.append(combined)
        return texts

    def _finalize_profile_features(self) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        numeric = np.array(self._user_numeric_features, dtype=np.float32)
        binary = np.array(self._user_binary_features, dtype=np.float32)

        num_users = numeric.shape[0]
        extras: List[List[float]] = []
        for u in range(num_users):
            sims = self._user_reply_sims.get(u, [])
            lats = self._user_reply_lat_min.get(u, [])
            if sims:
                sim_mean = float(np.mean(sims))
                sim_p50 = float(np.percentile(sims, 50))
                sim_p90 = float(np.percentile(sims, 90))
                very_sim_share = float(np.mean(np.array(sims) >= 0.9))
            else:
                sim_mean = sim_p50 = sim_p90 = very_sim_share = 0.0
            if lats:
                lat_p50 = float(np.percentile(lats, 50))
                lat_p90 = float(np.percentile(lats, 90))
                fast_reply_rate = float(np.mean(np.array(lats) <= 5.0))
            else:
                lat_p50 = lat_p90 = fast_reply_rate = 0.0
            extras.append([sim_mean, sim_p50, sim_p90, very_sim_share, lat_p50, lat_p90, fast_reply_rate])

        extras_np = np.array(extras, dtype=np.float32) if extras else np.zeros((numeric.shape[0], 7), dtype=np.float32)
        numeric_ext = np.concatenate([numeric, extras_np], axis=1)
        numeric_log = np.vectorize(_safe_log1p)(numeric_ext)

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
                relation = row["relation"]
                if relation not in RELATION_SCHEMA:
                    continue
                edge_counts[relation] += 1

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
            filled = edge_offsets[relation]
            edges_np = np.stack([src_array[:filled], dst_array[:filled]], axis=1)
            if edges_np.size > 0:
                edges_np = np.unique(edges_np, axis=0)
            if edges_np.size == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.from_numpy(edges_np.T).to(torch.long)
            data[(src_type, rel_name, dst_type)].edge_index = edge_index

            reverse_rel = f"{rel_name}_rev"
            if edges_np.size == 0:
                rev_index = torch.empty((2, 0), dtype=torch.long)
            else:
                rev_edges_np = edges_np[:, ::-1].copy()
                rev_index = torch.from_numpy(rev_edges_np.T).to(torch.long)
            data[(dst_type, reverse_rel, src_type)].edge_index = rev_index

        return data

    def _persist_artifacts(
        self,
        graph: HeteroData,
        texts: Sequence[str],
        profile_features: torch.Tensor,
        scaler_stats: Mapping[str, Sequence[float]],
        tweet_text_embeddings: Optional[Dict[str, torch.Tensor]],
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

        if tweet_text_embeddings is not None:
            artifact = {
                "indices": tweet_text_embeddings["indices"],
                "embeddings": tweet_text_embeddings["embeddings"],
                "model_name": self.tweet_text_model_name,
                "max_length": self.tweet_text_max_length,
            }
            torch.save(artifact, paths.tweet_text_embeddings_pt)

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
        encoded_tensors: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> None:
        self._texts = texts
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = device
        self._encoded = dict(encoded_tensors) if encoded_tensors is not None else None

    def fetch(self, indices: Sequence[int] | torch.Tensor) -> Mapping[str, torch.Tensor]:
        if self._encoded is not None:
            first_tensor = next(iter(self._encoded.values()))
            idx = torch.as_tensor(indices, dtype=torch.long, device=first_tensor.device)
            out: Dict[str, torch.Tensor] = {}
            for k, v in self._encoded.items():
                sel = v.index_select(0, idx)
                out[k] = sel.to(self._device) if self._device is not None else sel
            return out
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

    def fetch(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(indices, dtype=torch.long, device=self._tensor.device)
        feat = self._tensor[idx]
        if self._device is not None and feat.device != self._device:
            feat = feat.to(self._device)
        return feat


class TweetFeatureStore:
    """Lookup for tweet-level text embeddings used by the graph backbone."""

    def __init__(
        self,
        indices: torch.Tensor,
        embeddings: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        if indices.ndim != 1:
            raise ValueError("Tweet indices tensor must be 1-D")
        order = torch.argsort(indices)
        self._indices = indices[order].long()
        self._embeddings = embeddings[order]
        self._device = device

    @property
    def feature_dim(self) -> int:
        return int(self._embeddings.shape[1])

    def fetch(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        if idx_tensor.numel() == 0:
            return torch.empty((0, self.feature_dim), dtype=torch.float32, device=self._device)
        storage_device = self._indices.device
        search_indices = idx_tensor.to(storage_device)
        pos = torch.searchsorted(self._indices, search_indices)
        matched = (pos < self._indices.numel()) & (self._indices[pos] == search_indices)
        out = torch.zeros((idx_tensor.numel(), self.feature_dim), dtype=torch.float32, device=storage_device)
        if matched.any():
            out[matched] = self._embeddings[pos[matched]].to(torch.float32)
        if self._device is not None and out.device != self._device:
            out = out.to(self._device)
        return out


class TwiBot22DataModule:
    """High-level convenience wrapper for loading all processed artifacts."""

    def __init__(
        self,
        raw_root: Path,
        processed_root: Path,
        max_tweets_per_user: int = -1,
        tokenizer=None,
        text_max_length: int = 512,
        device: Optional[torch.device] = None,
        tweet_text_model_name: Optional[str] = None,
        tweet_text_max_length: int = 96,
        tweet_text_batch_size: int = 512,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.builder = TwiBot22GraphBuilder(
            raw_root,
            processed_root,
            max_tweets_per_user,
            tweet_text_model_name=tweet_text_model_name,
            tweet_text_max_length=tweet_text_max_length,
            tweet_text_batch_size=tweet_text_batch_size,
        )
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.device = device
        self.tweet_text_model_name = tweet_text_model_name

        self.graph: Optional[HeteroData] = None
        self.user_ids: Optional[List[str]] = None
        self.text_store: Optional[UserTextStore] = None
        self.profile_store: Optional[ProfileFeatureStore] = None
        self.tweet_feature_store: Optional[TweetFeatureStore] = None
        self.labels: Optional[torch.Tensor] = None
        self.splits: Optional[torch.Tensor] = None

    def prepare_data(self, rebuild: bool = False) -> None:
        paths = self.builder.get_processed_paths()
        if rebuild or not paths.graph_pt.exists():
            self.builder.build()

    def setup(self) -> None:
        paths = self.builder.get_processed_paths()

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
            encoded: Optional[Dict[str, torch.Tensor]] = None
            if paths.text_tokens_pt.exists():
                try:
                    encoded = _torch_load_compat(paths.text_tokens_pt)
                except Exception:
                    encoded = None
            if encoded is None:
                batch_size = 256
                all_input_ids: List[torch.Tensor] = []
                all_attention: List[torch.Tensor] = []
                token_type_ids: List[torch.Tensor] = []
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    enc = self.tokenizer(
                        batch,
                        padding='max_length',
                        truncation=True,
                        max_length=self.text_max_length,
                        return_tensors="pt",
                    )
                    all_input_ids.append(enc["input_ids"])
                    all_attention.append(enc["attention_mask"])
                    if "token_type_ids" in enc:
                        token_type_ids.append(enc["token_type_ids"])
                input_ids = torch.cat(all_input_ids, dim=0)
                attention_mask = torch.cat(all_attention, dim=0)
                encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids:
                    encoded["token_type_ids"] = torch.cat(token_type_ids, dim=0)
                torch.save(encoded, paths.text_tokens_pt)
            self.text_store = UserTextStore(
                texts,
                tokenizer=self.tokenizer,
                max_length=self.text_max_length,
                device=self.device,
                encoded_tensors=encoded,
            )
        profile_features = _torch_load_compat(paths.profile_features_pt)
        self.profile_store = ProfileFeatureStore(profile_features, device=self.device)
        self.labels = _torch_load_compat(paths.labels_pt)
        self.splits = _torch_load_compat(paths.splits_pt)

        if self.tweet_text_model_name is not None and paths.tweet_text_embeddings_pt.exists():
            payload = _torch_load_compat(paths.tweet_text_embeddings_pt)
            indices = payload.get("indices")
            embeddings = payload.get("embeddings")
            if isinstance(indices, torch.Tensor) and isinstance(embeddings, torch.Tensor):
                self.tweet_feature_store = TweetFeatureStore(indices, embeddings, device=self.device)
            else:
                self.tweet_feature_store = None

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
                    texts.extend([""] * (idx - len(texts)))
                    texts.append(text)
        return texts

    def get_split_indices(self, split: str) -> torch.Tensor:
        if self.splits is None:
            raise RuntimeError("Call setup() first")
        split_lookup = {"train": 0, "valid": 1, "val": 1, "test": 2}
        if split not in split_lookup:
            raise KeyError(f"Unknown split: {split}")
        split_value = split_lookup[split]
        return (self.splits == split_value).nonzero(as_tuple=False).view(-1)

    def get_node_feature_dims(self) -> Dict[str, int]:
        dims: Dict[str, int] = {}
        if self.tweet_feature_store is not None:
            dims["tweet"] = self.tweet_feature_store.feature_dim
        return dims

    def fetch_node_features(self, batch: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = {}
        if self.tweet_feature_store is not None and "tweet" in batch.node_types:
            n_id = batch["tweet"].n_id
            feats = self.tweet_feature_store.fetch(n_id)
            features["tweet"] = feats.to(device)
        return features
