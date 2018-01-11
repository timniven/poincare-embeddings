"""Models for training."""
import torch
from torch import nn
from poincare import func, util
import numpy as np
import random


class TextModel(nn.Module):
    """Model for training text hierarchies."""

    def __init__(self, emb_dim, trans_closure, num_neg=3):
        """Create a TextModel.

        Args:
          emb_dim: Integer, the embedding dimension.
          trans_closure: np.ndarray representing the transitive closure of the
            text to be modeled.
          num_neg: Integer, the number of negative samples to take for
            each positive sample in the loss calculation. Defaults to 3.
        """
        super(TextModel, self).__init__()
        self.emb_dim = emb_dim
        self.trans_closure = trans_closure
        self.num_neg = num_neg
        self.vocab_size = trans_closure.shape[0]
        self.embeddings = util.embeddings(emb_dim, self.vocab_size)

    def epoch_data(self):
        """Sample the positive and negative pairs for loss for an epoch.

        The number of negative examples to sample is controlled by the hyper-
        parameter num_neg (see constructor).

        Returns:
          data:
            nested list structure with pair and negative samples like:
              [((u, v), [(u, v'1), (u, v'2), ..., (u, v'n)]), ...]
        """
        u, v = np.nonzero(self.trans_closure)
        pos_pairs = [(a, b) for a, b in list(zip(u.tolist(), v.tolist()))
                     if a != b]
        # randomize order each epoch
        random.shuffle(pos_pairs)
        return list(zip(pos_pairs,
                        [self.neg_samples(u, v) for u, v in pos_pairs]))

    def neg_samples(self, u, v):
        """Get a list of randomly sampled negatives.

        Args:
          u: Integer, ix for the u term in the loss function.
          v: Integer, ix for the v term in the loss function.

        Returns:
          List of Integer ixs representing negative samples for the loss.
        """
        N = [v]
        pos_ixs = [ix for ix in np.nonzero(self.trans_closure[u, :])[0].tolist()
                   if ix != u]
        neg_ixs = [ix for ix in range(self.vocab_size)
                   if (ix not in pos_ixs and ix != u)]
        num_negs = min(len(neg_ixs), self.num_neg)
        if num_negs > 0:
            N += random.sample(neg_ixs, num_negs)
        return N

    def loss(self, data):
        """Calculate loss for an epoch.

        .. math::
            L(\Theta) =
                \sum_{(u, v) \in D}
                    \log \frac{e^{-d(u, v)}}
                              {\sum_{v' in N(u)} e^{-d(u, v')}}

        Args:
          data: nested list structure with pair and negative samples like:
              [((u, v), [(u, v'1), (u, v'2), ..., (u, v'n)]), ...]

        Returns:
          Float.
        """
        # TODO: figure out how to vectorize this and eliminate the for loops
        loss = 0.
        for pair, negs in data:
            u_ix, v_ix = pair
            u = self.embeddings(torch.LongTensor([u_ix]))
            v = self.embeddings(torch.LongTensor([v_ix]))
            neg_sum = 0.
            for v_prime_ix in negs:
                v_prime = self.embeddings(torch.LongTensor([v_prime_ix]))
                neg_sum += np.exp(-1 * func.poincare_distance(u, v_prime))
            loss += np.log(np.exp(-1 * func.poincare_distance(u, v)) / neg_sum)
        return loss
