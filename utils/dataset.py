import torch
from torch.utils.data import Dataset


class GaslightingDataset(Dataset):
    def __init__(self, data, vocab, max_length=50):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        labels = sorted({item["category"] for item in data})
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}

    def __len__(self):
        return len(self.data)

    def _build_text(self, item):
        utterance = item["utterance"]
        speaker = item.get("speaker_role", None)

        if speaker:
            return f"{speaker}: {utterance}"
        return utterance

    def __getitem__(self, idx):
        item = self.data[idx]

        text = self._build_text(item)
        ids = self.vocab.encode(text)

        ids = ids[:self.max_length]
        pad_len = self.max_length - len(ids)
        ids = ids + [self.vocab.word2idx[self.vocab.PAD_TOKEN]] * pad_len

        label = self.label2idx[item["category"]]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
