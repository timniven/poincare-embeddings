"""Utility functions."""
import torch
from torch import nn


def embeddings(embed_dim, vocab_size, requires_grad=False):
    """Embedding module.

    Embeddings will be uniform in range [-0.001, 0.001], as specified in the
    paper on p. 5 under "Training Details".

    Args:
      embed_dim: Integer.
      vocab_size: Integer.
      requires_grad: Boolean. False by default - don't think we have autodiff
        for hyperbolic geometry.

    Returns:
      torch.nn.Embeddings of shape vocab_size x embed_dim.
    """
    tensor = torch.FloatTensor(vocab_size, embed_dim).uniform_(-0.001, 0.001)
    embs = nn.Embedding(vocab_size, embed_dim)
    embs.weight = nn.Parameter(tensor, requires_grad=requires_grad)
    return embs
