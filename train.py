import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.transformer import TransformerGaslightingDetector
from models.hierarchical import HierarchicalGaslightingDetector
from utils.vocab import Vocabulary
from utils.dataset import GaslightingDataset, ContextWindowDataset
from utils.metrics import compute_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


def main(args):
    config = load_config(args.config)
    set_seed(config["seed"])

    # Directories
    save_dir = config["training"]["save_dir"]
    log_dir = config["training"]["log_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data loading
    ds = load_dataset(config["data"]["dataset_name"])
    train_raw = ds["train"]
    val_raw = ds["validation"]

    # Build vocab on training set
    vocab = Vocabulary(max_vocab_size=config["data"]["vocab_size"])
    vocab.build_from_dataset(train_raw)
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    vocab.save(vocab_path)
    
    # Model type
    model_type = config["model"].get("type", "baseline")
    
    # Wrap datasets based on type of model
    if model_type == "hierarchical":
        context_size = config["data"].get("context_size", 5)
        train_dataset = ContextWindowDataset(
            train_raw, vocab,
            max_length=config["data"]["max_length"],
            context_size=context_size,
        )
        val_dataset = ContextWindowDataset(
            val_raw, vocab,
            max_length=config["data"]["max_length"],
            context_size=context_size,
        )
    else:
        train_dataset = GaslightingDataset(
            train_raw, vocab, max_length=config["data"]["max_length"]
        )
        val_dataset = GaslightingDataset(
            val_raw, vocab, max_length=config["data"]["max_length"]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    # Model
    num_classes = len(train_dataset.label2idx)
    if model_type == "hierarchical":
        max_utts = config["data"].get("context_size", 5) + 1
        model = HierarchicalGaslightingDetector(
            vocab_size=len(vocab),
            embedding_dim=config["model"]["embedding_dim"],
            num_heads=config["model"]["num_heads"],
            num_utterance_layers=config["model"].get("utterance_layers", 1),
            num_conversation_layers=config["model"].get("conversation_layers", 1),
            dim_feedforward=config["model"]["dim_feedforward"],
            num_classes=num_classes,
            dropout=config["model"]["dropout"],
            max_length=config["data"]["max_length"],
            max_utterances=max_utts,
        ).to(device)
    else:
        model = TransformerGaslightingDetector(
            vocab_size=len(vocab),
            embedding_dim=config["model"]["embedding_dim"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            num_classes=num_classes,
            dropout=config["model"]["dropout"],
            max_length=config["data"]["max_length"],
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # class-balanced loss 
    use_class_weights = config["training"].get("use_class_weights", False)
    class_weights = None
    if use_class_weights:
        from collections import Counter

        counts = Counter([item["category"] for item in train_raw])
        total = sum(counts.values())
        class_weights_list = []
        for i in range(num_classes):
            lab = train_dataset.idx2label[i]
            c = counts[lab]
            class_weights_list.append(total / c)
        class_weights = torch.tensor(class_weights_list, dtype=torch.float, device=device)
        print(f"Class weights: {class_weights_list}")
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    writer = SummaryWriter(log_dir=log_dir)
    best_metric = 0.0
    patience_counter = 0
    primary_metric = config["evaluation"]["primary_metric"]

    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=config["training"]["gradient_clip"],
        )

        print(
            f"Train - loss: {train_loss:.4f}, "
            f"{primary_metric}: {train_metrics[primary_metric]:.4f}"
        )

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Val   - loss: {val_loss:.4f}, "
            f"{primary_metric}: {val_metrics[primary_metric]:.4f}"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"Metrics/{name}", value, epoch)

        scheduler.step(val_metrics[primary_metric])

        # Checkpointing on best val metric
        current = val_metrics[primary_metric]
        if current > best_metric:
            best_metric = current
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
                "label_mapping": {
                    "label2idx": train_dataset.label2idx,
                    "idx2label": train_dataset.idx2label,
                },
            }
            ckpt_path = os.path.join(save_dir, "baseline.pt")
            torch.save(checkpoint, ckpt_path)
            print("Saved new best model.")
        else:
            patience_counter += 1

        if patience_counter >= config["training"]["patience"]:
            print("Early stopping.")
            break

    writer.close()
    print(f"\nBest val {primary_metric}: {best_metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config",
    )
    main(parser.parse_args())
