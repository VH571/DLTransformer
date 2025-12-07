import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def compute_metrics(preds, labels, label_names=None):
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
    }

    per_class = f1_score(labels, preds, average=None)
    for i, f1 in enumerate(per_class):
        name = label_names[i] if label_names else f"class_{i}"
        metrics[f"f1_{name}"] = f1

    return metrics


def print_classification_report(preds, labels, label_names):
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(labels, preds, target_names=label_names))


def get_confusion_matrix(preds, labels):
    return confusion_matrix(labels, preds)
