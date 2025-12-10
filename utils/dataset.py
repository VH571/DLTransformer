import torch
from collections import defaultdict
from torch.utils.data import Dataset


#dataset for baseline (one utterance) 
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
            "text": str(text),
            "category": str(item.get("category", "")),
            "id": str(item.get("id", "")),
            "turn": str(item.get("turn", "")),
            "speaker_role": str(item.get("speaker_role", "")),
}


#dataset for hierarchical model with the context 
# of the previous utterances context and the 
# current one
class ContextWindowDataset(Dataset):
    def __init__(self, data, vocab, max_length = 50, context_size = 5):
        self.vocab = vocab
        self.max_length = max_length
        self.context_size = context_size

        #each group by conversation id
        conversations = defaultdict(list)

        for item in data:
            conv_id = item["id"]
            conversations[conv_id].append(item)
        
        #sort thru by intex
        for con_id in conversations:
            conversations[conv_id] = sorted(conversations[con_id], key=lambda x: x["turn"])

        self.conversations = conversations

        labels = sorted({item["category"] for item in data})
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}
        #flat index 
        self.indices = []
        for cid, utterances in self.conversations.items():
            for i in range(len(utterances)):
                self.indices.append((cid, i))

    def __len__(self):
        return len(self.indices)

    def _build_text(self, item):
        speaker = item.get("speaker_role", None)
        utterance = item["utterance"]

        if speaker:
            return f"{speaker}: {utterance}"
        return utterance

    def _encode_utterance(self, text):
        ids = self.vocab.encode(text)
        ids = ids[:self.max_length]
        pad_id = self.vocab.word2idx[self.vocab.PAD_TOKEN]
        pad_len = self.max_length - len(ids)

        return ids + [pad_id] * pad_len

    def __getitem__(self, idx):
        conv_id, pos = self.indices[idx]
        conv = self.conversations[conv_id]

        start = max(0, pos - self.context_size)

        #window has the context_size utterance plus the current one
        window = conv[start: pos + 1]

        encoded = [self._encode_utterance(self._build_text(item)) for item in window]

        max_utts = self.context_size + 1
        pad_id = self.vocab.word2idx[self.vocab.PAD_TOKEN]

        #pad the ledt so that the recne tmessage is at the end
        while len(encoded) < max_utts:
            encoded.insert(0, [pad_id] * self.max_length)

        if len(encoded) > max_utts:
            encoded = encoded[-max_utts:]

        input_ids = torch.tensor(encoded, dtype=torch.long)

        target_item = conv[pos]
        label = self.label2idx[target_item["category"]]

        # human readable context texts
        target_text = self._build_text(target_item)
        context_items = conv[start:pos]
        context_texts = [self._build_text(it) for it in context_items]

        return {
            "input_ids": input_ids,
            "label": torch.tensor(label, dtype=torch.long),
            "target_text": str(target_text),
            "context_texts": "\n".join([str(t) for t in context_texts]),
            "category": str(target_item.get("category", "")),
            "id": str(target_item.get("id", "")),
            "turn": str(target_item.get("turn", "")),
            "speaker_role": str(target_item.get("speaker_role", "")),
            }

