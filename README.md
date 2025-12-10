# Gaslighting Detection - Hierarchical (Stage 3)

## Dataset

This project uses the **GaslightingLLM** dataset from HuggingFace: `Maxwe11y/gaslighting`

The dataset loads automatically when you run the scripts.

## Getting Started

### Step 1

```bash
python -m venv venv
venv\Scripts\activate   
```

### Step 2

```bash
pip install -r requirements.txt
```

### Step 3

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/hierarchical.yaml

```

This saves your trained model to `checkpoints/baseline.pt` and `checkpoints_hier/baseline.pt`
### Step 4

```bash
python eval.py --checkpoint checkpoints/baseline.pt --split test
python eval.py --checkpoint checkpoints_hier/baseline.pt --split test

```

This shows accuracy, F1 scores, and a confusion matrix, as well as some examples that it missed.
