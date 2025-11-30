"""Evaluate a trained TwiBot-22 model checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data import TwiBot22DataModule
from src.models import ProfileFeatureEncoder, RelationalGraphBackbone, RobertaTextEncoder, TwiBotModel, HGTBackbone
from src.training import TrainingConfig, Trainer, set_seed
from src.utils.losses import FocalLoss, WeightedCrossEntropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TwiBot-22 relational graph transformer model")
    parser.add_argument("--raw-data", type=Path, default=Path("TwiBot-22"), help="Path to raw TwiBot-22 dataset directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"), help="Directory containing processed artifacts and checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (defaults to processed_dir/best_model.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max-tweets", type=int, default=-1)
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
    parser.add_argument("--text-model", type=str, default="xlm-roberta-base")
    parser.add_argument("--text-trainable", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--enable-tweet-text", action="store_true", help="Load/compute tweet text embeddings for tweet nodes")
    parser.add_argument("--tweet-text-model", type=str, default=None, help="Optional HF model for tweet embeddings (defaults to --text-model when enabled)")
    parser.add_argument("--tweet-text-max-length", type=int, default=96)
    parser.add_argument("--tweet-text-batch-size", type=int, default=512)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild processed artifacts before evaluation")
    parser.add_argument("--use-hgt", action="store_true", help="Use HGT backbone instead of RGT")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", help="Enable deterministic behavior (disables CuDNN benchmark)")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false", help="Allow non-deterministic CuDNN ops")
    parser.add_argument("--auto-config", action="store_true", default=True, help="Auto-detect training config from checkpoint (default: True)")
    parser.add_argument("--no-auto-config", dest="auto_config", action="store_false", help="Use CLI args instead of checkpoint config")
    parser.set_defaults(deterministic=True)
    return parser.parse_args()


def load_checkpoint_config(checkpoint_path: Path, args: argparse.Namespace) -> argparse.Namespace:
    """Load training configuration from checkpoint and override args if auto_config is enabled."""
    if not args.auto_config:
        return args

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint:
            saved_config = checkpoint["config"]
            print("Auto-detected training config from checkpoint:")

            if "text_trainable" in saved_config:
                if saved_config["text_trainable"] != args.text_trainable:
                    print(f"  - text_trainable: {args.text_trainable} -> {saved_config['text_trainable']}")
                args.text_trainable = saved_config["text_trainable"]

            if "use_lora" in saved_config:
                if saved_config["use_lora"] != args.use_lora:
                    print(f"  - use_lora: {args.use_lora} -> {saved_config['use_lora']}")
                args.use_lora = saved_config["use_lora"]

            if "text_model_name" in saved_config:
                if saved_config["text_model_name"] != args.text_model:
                    print(f"  - text_model: {args.text_model} -> {saved_config['text_model_name']}")
                args.text_model = saved_config["text_model_name"]

            if "hidden_dim" in saved_config:
                if saved_config["hidden_dim"] != args.hidden_dim:
                    print(f"  - hidden_dim: {args.hidden_dim} -> {saved_config['hidden_dim']}")
                args.hidden_dim = saved_config["hidden_dim"]

            if "num_layers" in saved_config:
                if saved_config["num_layers"] != args.num_layers:
                    print(f"  - num_layers: {args.num_layers} -> {saved_config['num_layers']}")
                args.num_layers = saved_config["num_layers"]

            if "graph_heads" in saved_config:
                if saved_config["graph_heads"] != args.graph_heads:
                    print(f"  - graph_heads: {args.graph_heads} -> {saved_config['graph_heads']}")
                args.graph_heads = saved_config["graph_heads"]
        else:
            print("Warning: Checkpoint does not contain config. Using CLI args.")
            print("  This may cause state_dict mismatch if training config differs.")
    except Exception as e:
        print(f"Warning: Could not load config from checkpoint: {e}")
        print("  Using CLI args instead.")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or (args.processed_dir / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    args = load_checkpoint_config(checkpoint_path, args)

    tweet_text_model = None
    if args.enable_tweet_text:
        tweet_text_model = args.tweet_text_model or args.text_model

    cfg = TrainingConfig(
        raw_data_dir=args.raw_data,
        processed_data_dir=args.processed_dir,
        batch_size=args.batch_size,
        device=str(device),
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        graph_heads=args.graph_heads,
        dropout=args.dropout,
        max_tweets_per_user=args.max_tweets,
        num_neighbors=tuple(args.num_neighbors),
        loss_type=args.loss,
        gamma=args.gamma,
        bot_weight_mult=args.bot_weight_mult,
        label_smoothing=args.label_smoothing,
        text_model_name=args.text_model,
        text_trainable=args.text_trainable,
        use_lora=args.use_lora,
        mixed_precision=False,
        epochs=0,
        patience=0,
        text_max_length=args.text_max_length,
        tweet_text_model_name=tweet_text_model,
        tweet_text_max_length=args.tweet_text_max_length,
        tweet_text_batch_size=args.tweet_text_batch_size,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    data_module = TwiBot22DataModule(
        raw_root=cfg.raw_data_dir,
        processed_root=cfg.processed_data_dir,
        max_tweets_per_user=cfg.max_tweets_per_user,
        tokenizer=tokenizer,
        text_max_length=cfg.text_max_length,
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

    # Get profile dimension and add to node_feature_dims BEFORE creating backbone
    # This is critical - the backbone needs to know user node feature dimensions
    # to create the appropriate projection layer
    if data_module.profile_store is None:
        raise RuntimeError("Profile feature store is not initialized.")
    profile_sample = data_module.profile_store.fetch([0])
    profile_dim = profile_sample.shape[-1]
    
    if "user" not in node_feature_dims:
        node_feature_dims["user"] = profile_dim

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

    profile_encoder = ProfileFeatureEncoder(input_dim=profile_dim, hidden_dims=(128, 64), dropout=cfg.dropout)

    model = TwiBotModel(
        graph_backbone=graph_backbone,
        text_encoder=text_encoder,
        profile_encoder=profile_encoder,
        fusion_hidden_dims=(512, 256, 128),
        dropout=cfg.dropout,
    )

    state = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state["model_state"])
    except RuntimeError as e:
        print(f"\nError loading state dict: {e}")
        print("\nThis usually means the model architecture doesn't match the checkpoint.")
        print("Common causes:")
        print("  1. --text-trainable and/or --use-lora flags differ from training")
        print("  2. --hidden-dim, --num-layers, or --graph-heads differ")
        print("\nTry running with --no-auto-config and matching the training args exactly.")
        raise

    train_indices = data_module.get_split_indices("train")
    train_labels = data_module.labels[train_indices]
    train_labels = train_labels[train_labels >= 0]
    if train_labels.numel() == 0:
        raise RuntimeError("No labeled training nodes found for weight computation.")
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
    class_weights = (class_counts.sum() / class_counts).to(device)
    class_weights[1] = class_weights[1] * cfg.bot_weight_mult

    if cfg.loss_type == "focal":
        criterion = FocalLoss(gamma=cfg.gamma, alpha=class_weights, label_smoothing=cfg.label_smoothing)
    else:
        criterion = WeightedCrossEntropy(weight=class_weights, label_smoothing=cfg.label_smoothing)

    trainer = Trainer(cfg, model, data_module, criterion, optimizers={})
    metrics = trainer.evaluate(split=args.split)

    print(f"\n{'='*50}")
    print(f"Evaluation Results on {args.split} split:")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
