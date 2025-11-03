import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            embed_dim, n_heads, n_layers, n_layers, dim_feedforward=512
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x, x)
        return self.fc(x)
