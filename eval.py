import argparse
import numpy as np
import torch

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from models.transformer import TransformerGaslightingDetector
from utils.vocab import Vocabulary
from utils.dataset import GaslightingDataset
from utils.metrics import (
    compute_metrics,
    print_classification_report,
    get_confusion_matrix,
)


def evaluate_model(checkpoint_path, split="test"):
    # checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

   
     # load dataset
    ds = load_dataset(config["data"]["dataset_name"])
    data_raw = ds[split]

    # load vocab
    vocab_path = f"{config['training']['save_dir']}/vocab.pkl"
    vocab = Vocabulary.load(vocab_path)
    # wrap dataset
    dataset = GaslightingDataset(
        data_raw, vocab, max_length=config["data"]["max_length"]
    )

   
    # ensure label mapping matches training
    dataset.label2idx = checkpoint["label_mapping"]["label2idx"]
    dataset.idx2label = checkpoint["label_mapping"]["idx2label"]

    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # build model
    
    num_classes = len(dataset.label2idx)
    model = TransformerGaslightingDetector(
        vocab_size=len(vocab),
        embedding_dim=config["model"]["embedding_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        num_classes=num_classes,
        dropout=config["model"]["dropout"],
        max_length=config["data"]["max_length"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Evaluation loop
    all_preds, all_labels = [], []

    print(f"Evaluating on split: {split}")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]

            logits = model(input_ids)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    label_names = [dataset.idx2label[i] for i in range(len(dataset.label2idx))]
    metrics = compute_metrics(all_preds, all_labels, label_names)

    print("Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report:")
    print_classification_report(all_preds, all_labels, label_names)
    
    
    cm = get_confusion_matrix(all_preds, all_labels)
    print("\nConfusion matrix:")
    print(cm)

    return metrics, all_preds, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (baseline.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Which dataset split to evaluate",
    )
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.split)
