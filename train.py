"""Entry point for training the TwiBot-22 RGT-light model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data import TwiBot22DataModule
from src.models import ProfileFeatureEncoder, RelationalGraphBackbone, RobertaTextEncoder, TwiBotModel, HGTBackbone
from src.training import TrainingConfig, Trainer, set_seed
from src.utils.losses import FocalLoss, WeightedCrossEntropy
from src.utils.optim import build_optimizers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TwiBot-22 relational graph transformer model")
    parser.add_argument("--raw-data", type=Path, default=Path("TwiBot-22"), help="Path to raw TwiBot-22 dataset directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"), help="Directory to store processed artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max-tweets", type=int, default=-1, help="Max tweets per user (-1 for all)")
    parser.add_argument("--text-max-length", type=int, default=256, help="Max tokens for text encoder input")
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        metavar="N",
        default=[15, 10, 5],
        help="Neighbor sample sizes per GNN layer (one value per layer)",
    )
    parser.add_argument("--loss", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--bot-weight-mult", type=float, default=1.0, help="Multiply bot class weight to push precision up")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor (0.0-0.2 recommended)")
    parser.add_argument("--target-pos-ratio", type=float, default=None, help="Target bot ratio for oversampling train indices (0-1)")
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-text", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold (prevents NaN)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--text-model", type=str, default="xlm-roberta-base")
    parser.add_argument("--text-trainable", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--enable-tweet-text", action="store_true", help="Attach tweet text embeddings to tweet nodes")
    parser.add_argument("--tweet-text-model", type=str, default=None, help="Optional HF model name for tweet embeddings (defaults to --text-model when enabled)")
    parser.add_argument("--tweet-text-max-length", type=int, default=96, help="Max tokens for tweet text encoder input")
    parser.add_argument("--tweet-text-batch-size", type=int, default=512, help="Batch size for encoding tweet texts during preprocessing")
    parser.add_argument("--use-hgt", action="store_true", help="Use HGT backbone instead of RGT")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild processed artifacts before training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Enable deterministic behavior (disables CuDNN benchmark)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Allow non-deterministic CuDNN ops")
    parser.set_defaults(mixed_precision=False, deterministic=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log_dir = args.log_dir if args.log_dir else Path("runs/latest")
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    tweet_text_model = None
    if args.enable_tweet_text:
        tweet_text_model = args.tweet_text_model or args.text_model

    cfg = TrainingConfig(
        raw_data_dir=args.raw_data,
        processed_data_dir=args.processed_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=str(device),
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        graph_heads=args.graph_heads,
        dropout=args.dropout,
        max_tweets_per_user=args.max_tweets,
        num_neighbors=tuple(args.num_neighbors),
        loss_type=args.loss,
        gamma=args.gamma,
        lr_backbone=args.lr_backbone,
        lr_text=args.lr_text,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
        log_dir=log_dir,
        mixed_precision=args.mixed_precision,
        text_model_name=args.text_model,
        text_trainable=args.text_trainable,
        use_lora=args.use_lora,
        tweet_text_model_name=tweet_text_model,
        tweet_text_max_length=args.tweet_text_max_length,
        tweet_text_batch_size=args.tweet_text_batch_size,
        text_max_length=args.text_max_length,
        resume=args.resume,
        seed=args.seed,
        deterministic=args.deterministic,
        target_pos_ratio=args.target_pos_ratio,
        bot_weight_mult=args.bot_weight_mult,
        label_smoothing=args.label_smoothing,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    data_module = TwiBot22DataModule(
        raw_root=cfg.raw_data_dir,
        processed_root=cfg.processed_data_dir,
        max_tweets_per_user=cfg.max_tweets_per_user,
        tokenizer=tokenizer,
        text_max_length=args.text_max_length,
        device=device,
        tweet_text_model_name=cfg.tweet_text_model_name,
        tweet_text_max_length=cfg.tweet_text_max_length,
        tweet_text_batch_size=cfg.tweet_text_batch_size,
    )
    data_module.prepare_data(rebuild=args.rebuild)
    data_module.setup()

    graph = data_module.graph
    if graph is None:
        raise RuntimeError("Graph data is not loaded.")

    metadata = graph.metadata()
    num_nodes_dict = graph.num_nodes_dict
    node_feature_dims = data_module.get_node_feature_dims()

    if args.use_hgt:
        graph_backbone = HGTBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
            node_feature_dims=node_feature_dims,
        )
    else:
        graph_backbone = RelationalGraphBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
            node_feature_dims=node_feature_dims,
        )

    text_encoder = RobertaTextEncoder(
        model_name=cfg.text_model_name,
        trainable=cfg.text_trainable,
        dropout=cfg.dropout,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )

    if data_module.profile_store is None:
        raise RuntimeError("Profile feature store is not initialized.")
    profile_sample = data_module.profile_store.fetch([0])
    profile_dim = profile_sample.shape[-1]
    profile_encoder = ProfileFeatureEncoder(input_dim=profile_dim, hidden_dims=(128, 64), dropout=cfg.dropout)

    # Update node_feature_dims to include user profile features for the backbone
    if "user" not in node_feature_dims:
        node_feature_dims["user"] = profile_dim

    # Re-initialize graph backbone with updated node_feature_dims
    if args.use_hgt:
        graph_backbone = HGTBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
            node_feature_dims=node_feature_dims,
        )
    else:
        graph_backbone = RelationalGraphBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
            node_feature_dims=node_feature_dims,
        )

    model = TwiBotModel(
        graph_backbone=graph_backbone,
        text_encoder=text_encoder,
        profile_encoder=profile_encoder,
        fusion_hidden_dims=(512, 256, 128),
        dropout=cfg.dropout,
    )

    train_indices = data_module.get_split_indices("train")
    train_labels = data_module.labels[train_indices]
    train_labels = train_labels[train_labels >= 0]
    if train_labels.numel() == 0:
        raise RuntimeError("No labeled training nodes found.")
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
    class_weights = (class_counts.sum() / class_counts).to(device)
    class_weights[1] = class_weights[1] * cfg.bot_weight_mult

    if cfg.loss_type == "focal":
        loss_fn = FocalLoss(gamma=cfg.gamma, alpha=class_weights, label_smoothing=cfg.label_smoothing)
    else:
        loss_fn = WeightedCrossEntropy(weight=class_weights, label_smoothing=cfg.label_smoothing)

    optimizers = build_optimizers(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_text=cfg.lr_text,
        weight_decay=cfg.weight_decay,
    )

    trainer = Trainer(cfg, model, data_module, loss_fn, optimizers)
    trainer.train()


if __name__ == "__main__":
    main()
