import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """
    Loads pre-tokenized VQ codes produced by train/pretokenize.py.
    Returns (token_sequence, label) where token_sequence has shape (N,).
    AiM.forward() accepts this directly — no VQ-VAE encoding at training time.
    """

    def __init__(self, pt_path: str):
        data = torch.load(pt_path, map_location="cpu")
        self.tokens = data["tokens"]  # (total, N)  long
        self.labels = data["labels"]  # (total,)     long
        self.classes = data.get("classes", [])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Return shape (1, N) so AiM.forward sees len(x.shape)==3 → x.squeeze(1) → (N,)
        return self.tokens[idx].unsqueeze(0), int(self.labels[idx])
