"""Evaluate a trained TwiBot-22 model checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data import TwiBot22DataModule
from src.models import ProfileFeatureEncoder, RelationalGraphBackbone, RobertaTextEncoder, TwiBotModel, HGTBackbone
from src.training import TrainingConfig, Trainer
from src.utils.losses import FocalLoss, WeightedCrossEntropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TwiBot-22 relational graph transformer model")
    parser.add_argument("--raw-data", type=Path, default=Path("TwiBot-22"), help="Path to raw TwiBot-22 dataset directory")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"), help="Directory containing processed artifacts and checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint (defaults to processed_dir/best_model.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for evaluation (reduce if OOM)")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-tweets", type=int, default=-1)
    parser.add_argument("--text-max-length", type=int, default=256, help="Max tokens for text encoder input")
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs=2,
        metavar=("L1", "L2"),
        default=[20, 20],
        help="Neighbor sample sizes per GNN layer",
    )
    parser.add_argument("--loss", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--text-model", type=str, default="xlm-roberta-base")
    parser.add_argument("--text-trainable", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild processed artifacts before evaluation")
    parser.add_argument("--use-hgt", action="store_true", help="Use HGT backbone instead of RGT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or (args.processed_dir / "best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
        text_model_name=args.text_model,
        text_trainable=args.text_trainable,
        use_lora=args.use_lora,
        mixed_precision=False,
        epochs=0,
        patience=0,
        text_max_length=args.text_max_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    data_module = TwiBot22DataModule(
        raw_root=cfg.raw_data_dir,
        processed_root=cfg.processed_data_dir,
        max_tweets_per_user=cfg.max_tweets_per_user,
        tokenizer=tokenizer,
        text_max_length=cfg.text_max_length,
        device=device,
    )
    data_module.prepare_data(rebuild=args.rebuild)
    data_module.setup()

    graph = data_module.graph
    if graph is None:
        raise RuntimeError("Graph data is not loaded.")

    metadata = graph.metadata()
    num_nodes_dict = graph.num_nodes_dict

    if args.use_hgt:
        graph_backbone = HGTBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
        )
    else:
        graph_backbone = RelationalGraphBackbone(
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            heads=cfg.graph_heads,
            dropout=cfg.dropout,
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

    model = TwiBotModel(
        graph_backbone=graph_backbone,
        text_encoder=text_encoder,
        profile_encoder=profile_encoder,
        fusion_hidden_dims=(512, 256, 128),
        dropout=cfg.dropout,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])

    train_indices = data_module.get_split_indices("train")
    train_labels = data_module.labels[train_indices]
    train_labels = train_labels[train_labels >= 0]
    if train_labels.numel() == 0:
        raise RuntimeError("No labeled training nodes found for weight computation.")
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
    class_weights = (class_counts.sum() / class_counts).to(device)

    if cfg.loss_type == "focal":
        criterion = FocalLoss(gamma=cfg.gamma, alpha=class_weights)
    else:
        criterion = WeightedCrossEntropy(weight=class_weights)

    trainer = Trainer(cfg, model, data_module, criterion, optimizers={})
    metrics = trainer.evaluate(split=args.split)

    print(f"Evaluation on {args.split} split:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
