"""Models for training."""
import torch
from torch import nn
from poincare import func
from torch.autograd import Variable
from data import preprocess
import numpy as np


def nan_check(model, ixs, vecs, priors):
    """For debugging"""
    if np.isnan(np.sum(model.embeddings.cpu().numpy())):
        s = sorted([(i, t) for i, t in model.vocab.word.items()],
                   key=lambda x: x[0])
        print(s)
        print(priors)
        u_ix, v_ix, neg_ixs = ixs
        print('%s -> %s ; %s' % (model.vocab[u_ix],
                                 model.vocab[v_ix],
                                 ' '.join([model.vocab[i] for i in neg_ixs])))
        u, v, negs = vecs
        print('New values')
        print(u)
        print(v)
        print(negs)
        print('Grads')
        print(u.grad)
        print(v.grad)
        print(negs.grad)
        raise Exception


class TextModel(nn.Module):
    """Model for training text hierarchies."""

    def __init__(self, dataset_name, emb_dim, vocab):
        """Create a TextModel.

        Args:
          dataset_name: String. Needed to query dataset size.
          emb_dim: Integer, the embedding dimension.
          vocab: data.preprocess.Vocab.
        """
        super(TextModel, self).__init__()
        self.data_size = preprocess.data_size(dataset_name)
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embeddings = torch.Tensor(self.vocab_size, self.emb_dim)
        torch.nn.init.uniform(self.embeddings, a=-0.001, b=0.001)

    def fit(self, data, n_epochs=100, lr=0.2, beta=0.01, report=True):
        """Fit the model.

        Args:
          data: torch.utils.data.dataloader.DataLoader.
          n_epochs: Integer, the number of epochs to train for. Default is 100.
          lr: Float, starting learning rate. Default is 0.2. Will be annealed
            by the formula (1 / (1 + beta * epoch)) * lr.
          beta: Float, learning rate decay factor (see lr above). Default 0.01.
          report: Bool, whether or not to report average loss with each epoch.

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

            for _, ixs in enumerate(data):
                # debuggung
                priors = self.embeddings.clone()
                loss, *vecs = self.forward(ixs)
                self.optimize(loss, vecs, ixs, lr)
                epoch_loss += loss.data.numpy()[0][0]
                nan_check(self, ixs, vecs, priors)

            if report:
                print('%s\t%s' % (epoch, epoch_loss / self.data_size))

            if epoch_loss / self.data_size < 0.:
                continue

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

    @staticmethod
    def loss(u, v, negs):
        """Loss function a la Nickel paper.

        Seems they missed the minus sign at the front though.

        Args:
          u: Tensor, may be a 1D or 2D Tensor. Unsqueezing will happen in the
            distance calculation function if necessary.
          v: Tensor, may be a 1D or 2D Tensor. Unsqueezing will happen in the
            distance calculation function if necessary.
          negs: Tensor, matrix.

        Returns:
          Variable.
        """
        return -torch.log(torch.exp(-func.poincare_distance(u, v))
                          / torch.exp(-func.poincare_distance(u, negs)).sum())

    def optimize(self, loss, vecs, ixs, lr):
        """Optimization step.

        Args:
          loss: Tensor.
          vecs: Tuple with (u, v, negs), where u and v are vector Tensors and
            negs is a Tensor matrix.
          ixs: Tuple with (Integer u_ix, Integer v_ix, List(Int) neg_ixs).
          lr: Float, the learning rate.
        """
        u, v, negs = vecs
        u_ix, v_ix, neg_ixs = ixs

        loss.backward()

        self.embeddings[neg_ixs] -= lr * func.scale_grad(negs)
        self.embeddings[u_ix] -= lr * func.scale_grad(u)
        self.embeddings[v_ix] -= lr * func.scale_grad(v)
        self.embeddings = func.project(self.embeddings)
