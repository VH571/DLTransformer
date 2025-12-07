from collections import Counter
import pickle


class Vocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word_counts = Counter()

        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx2word = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

    def tokenize(self, text):
        return text.lower().split()

    def build_from_dataset(self, dataset):
        print("Building vocabulary...")

        for item in dataset:
            text = item["utterance"]
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

        most_common = self.word_counts.most_common(
            self.max_vocab_size - len(self.word2idx)
        )

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocab size: {len(self.word2idx)}")

    def encode(self, text):
        tokens = self.tokenize(text)
        return [
            self.word2idx.get(tok, self.word2idx[self.UNK_TOKEN])
            for tok in tokens
        ]

    def save(self, path):
        data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "max_vocab_size": self.max_vocab_size,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        vocab = cls(max_vocab_size=data["max_vocab_size"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab

    def __len__(self):
        return len(self.word2idx)
