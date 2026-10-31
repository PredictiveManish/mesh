"""A deliberately tiny model so the prototype runs anywhere.

The point of this repository is the system around training, not the model.
120k parameters is enough to show real convergence, real compression, real
verification and real churn handling, and small enough to run on a laptop CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

VOCAB = 40
SEQ_LEN = 48


class TinyLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = 64, layers: int = 2,
                 heads: int = 2, seq: int = SEQ_LEN):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers,
                                            enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        pos = torch.arange(t).unsqueeze(0).expand(b, t)
        h = self.tok(x) + self.pos(pos)
        h = self.blocks(h)
        return self.head(self.ln(h))


def build(seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_corpus(n_tokens: int = 60_000, seed: int = 0) -> torch.Tensor:
    """A synthetic 'language' with learnable structure.

    Each document is an arithmetic progression modulo VOCAB, so a small
    transformer can learn it in a few hundred steps and we get a clean,
    fast convergence signal to compare runs against. Vectorised: this runs
    in milliseconds, which matters because every worker builds it on start.
    """
    g = torch.Generator().manual_seed(seed)
    n_docs = n_tokens // 192 + 2
    lens = torch.randint(128, 384, (n_docs,), generator=g)
    steps = torch.randint(1, VOCAB, (n_docs,), generator=g)
    starts = torch.randint(0, VOCAB, (n_docs,), generator=g)

    docs, total = [], 0
    for i in range(n_docs):
        length = int(lens[i])
        j = torch.arange(length)
        docs.append((int(starts[i]) + int(steps[i]) * j) % VOCAB)
        total += length
        if total >= n_tokens:
            break
    return torch.cat(docs)[:n_tokens]


def shard(corpus: torch.Tensor, worker: int, workers: int) -> torch.Tensor:
    """Contiguous, non-overlapping slice: every worker sees different data."""
    n = len(corpus)
    lo = (n * worker) // workers
    hi = (n * (worker + 1)) // workers
    return corpus[lo:hi]


def batch(data: torch.Tensor, bs: int, seq: int, gen: torch.Generator):
    """(x, y) pair where y is x shifted by one token."""
    ix = torch.randint(0, len(data) - seq - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + seq] for i in ix])
    y = torch.stack([data[i + 1:i + seq + 1] for i in ix])
    return x, y


def probe_batch(corpus: torch.Tensor, bs: int = 8, seq: int = SEQ_LEN) -> tuple:
    """Fixed held-out batch used to measure the global model each round."""
    g = torch.Generator().manual_seed(1234)
    return batch(corpus, bs, seq, g)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    loss = nn.functional.cross_entropy(
        model(x).reshape(-1, VOCAB), y.reshape(-1)
    )
    model.train()
    return float(loss)
