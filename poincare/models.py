"""Models for training."""
import torch
from torch import nn
from poincare import func
from torch.autograd import Variable


def nickel_loss(u, v, negs):
    """Loss function a la Nickel paper.

    Seems they missed the minus sign at the front though.

    Args:
      u: Tensor, may be a 1D or 2D Tensor. Unsqueezing will happen in the
        distance calculation function if necessary.
      u: Tensor, may be a 1D or 2D Tensor. Unsqueezing will happen in the
        distance calculation function if necessary.

    Returns:
      Tensor.
    """
    return -torch.log(torch.exp(-func.poincare_distance(u, v))
                      / torch.exp(-func.poincare_distance(u, negs)))


class TextModel(nn.Module):
    """Model for training text hierarchies."""

    def __init__(self, emb_dim, vocab, loss=nickel_loss):
        """Create a TextModel.

        Args:
          emb_dim: Integer, the embedding dimension.
          vocab: Dictionary with words as keys and ids as values.
          loss: Loss function. Default is nickel_loss.
        """
        super(TextModel, self).__init__()
        self.emb_dim = emb_dim
        self.vocab_dict = vocab
        self.vocab_size = len(vocab)
        self.ix_to_word = {v: k for k, v in vocab.items()}
        self.loss = loss
        self.embeddings = torch.Tensor(self.vocab_size, self.emb_dim)
        torch.nn.init.uniform(self.embeddings, a=-0.001, b=0.001)

    def fit(self, data, n_epochs=100, lr=0.2, beta=0.01, report=True):
        """Fit the model.

        Args:
          data: poincare.data.Sampler() class exposing a new_epoch() function
            that returns a list of (u_ix, v_ix, neg_ixs) for each training
            sample.
          n_epochs: Integer, the number of epochs to train for. Default is 100.
          lr: Float, starting learning rate. Default is 0.2. Will be annealed
            by the formula (1 / (1 + beta * epoch)) * lr.
          beta: Float, learning rate decay factor (see lr above). Default 0.01.
          report: Bool, whether or not to report loss with each sample.

        Returns:
          Tensor: embeddings.
        """
        # TODO: think of a reasonable convergence condition
        for epoch in range(n_epochs):
            epoch_loss = 0.
            elr = (1 / (1 + beta * epoch)) * lr  # effective learning rate

            # Paper reduces lr for first 10 epochs to improve angular layout
            if epoch <= 10:
                elr /= 10.

            for ixs in data.new_epoch():
                loss, *vecs = self.forward(ixs)
                self.optimize(loss, vecs, ixs)
                epoch_loss += loss.data.numpy()[0][0]

            if report:
                print('%s\t%s' % (epoch, epoch_loss / float(data.n)))

        return self.embeddings

    def forward(self, ixs):
        """Forward pass.

        Args:
          ixs: Tuple of (u_ix, v_ix, neg_ixs) representing a training sample.
        """
        u_ix, v_ix, neg_ixs = ixs
        u = self.lookup([u_ix])
        v = self.lookup([v_ix])
        negs = self.lookup(neg_ixs)
        loss = self.loss(u, v, negs)
        return loss, u, v, negs

    def lookup(self, ix):
        """Embedding lookup.

        Any unsqueezing to be done downstream.

        Args:
          ix: Integer or List of Integers.

        Returns:
          Tensor: 1D vector or matrix.
        """
        return Variable(self.embeddings[ix], requires_grad=True)

    def optimize(self, loss, vecs, ixs):
        """Optimization step.

        Args:
          loss: Tensor.
          vecs: Tuple with (u, v, negs), where u and v are vector Tensors and
            negs is a Tensor matrix.
        """
        u, v, negs = vecs
        u_ix, v_ix, neg_ixs = ixs

        # This is probably redundant
        u.zero_grad()
        v.zero_grad()
        negs.zero_grad()

        loss.backward()

        self.embeddings[neg_ixs] -= self.lr * func.scale_grad(negs)
        self.embeddings[u_ix] -= self.lr * func.scale_grad(u)
        self.embeddings[v_ix] -= self.lr * func.scale_grad(v)
        self.embeddings = func.project(self.embeddings)
