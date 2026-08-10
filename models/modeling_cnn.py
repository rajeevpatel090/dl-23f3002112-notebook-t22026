"""
Custom TextCNN model made Hugging Face Hub compatible.
Architecture inferred from checkpoint weights:
- Embedding: (30522 vocab, 140 dim)
- 3 parallel Conv1d branches: kernel sizes 3, 4, 5 — each with 100 output channels
- Final FC layer: (300 -> 1)  [300 = 100*3 concatenated conv outputs]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class TextCNNConfig:
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 140,
        num_filters: int = 100,
        kernel_sizes: list = [3, 4, 5],
        num_classes: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes


class TextCNN(nn.Module, PyTorchModelHubMixin):
    """
    TextCNN for MCQ scoring, pushable to / loadable from the Hugging Face Hub.

    Usage:
        # Save + push
        model = TextCNN(config)
        model.load_state_dict(torch.load("original_state_dict.pt"))
        model.push_to_hub("your-username/cnn-mcq-model")

        # Load back
        model = TextCNN.from_pretrained("your-username/cnn-mcq-model")
    """

    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = {}
        cfg = TextCNNConfig(**config)
        self.config = config if config else cfg.__dict__

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.embed_dim, out_channels=cfg.num_filters, kernel_size=k)
            for k in cfg.kernel_sizes
        ])
        self.fc = nn.Linear(cfg.num_filters * len(cfg.kernel_sizes), cfg.num_classes)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                 # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        cat = torch.cat(pooled, dim=1)         # (batch, num_filters * len(kernel_sizes))
        out = self.fc(cat)                     # (batch, num_classes)
        return out
