"""Samplers for data preparation."""
from poincare import preprocess
import numpy as np
import random


class Sampler:
    """For sampling epoch data.

    We will look up the adjacency matrices (see poincare.data) and prepare an
    epoch worth of data according to the "mode" selected:
      up:
        u = child
        v = parent
        N(u) = {all parents that don't appear with u} + {u}
      down:
        u = parent
        v = child
        N(u) = {all children that don't appear with u} + {u}
      both:
        u = children and parents
        v = children and parents
        N(u) = {all nodes not directly connected to u} + {u}

    I believe the purpose of adding u to the N(u) set is to ensure it isn't an
    empty set and therefore the loss function in the paper doesn't encounter a
    divide by zero.

    TODO: Think about memory - we have adj_mat and pos_pairs in memory so far.
    """

    def __init__(self, dataset_name, mode, num_negs):
        """Create a new Sampler.

        Args:
          dataset_name: String.
          mode: String in {}, determining sampling strategy.
          num_negs: Integer, the number of negative examples to sample.
        """
        self.adj_mat = preprocess.get_adjacency(dataset_name, mode)
        self.num_negs = num_negs
        self.n_words = self.adj_mat.shape[0]
        self.pos_pairs = self.get_pos_pairs()

    def epoch_data(self):
        """Get data for a new epoch.

        Returns:
          Generator of Tuples (u, v, [negs]), the contents of which are all
            Integers.
        """
        random.shuffle(self.pos_pairs)
        for u, v in self.pos_pairs:
            yield (u, v, self.neg_samples(u))

    def get_pos_pairs(self):
        """Gets set of all positive pairs from adjacency matrix.

        Returns:
          Set of Tuples (u, v).
        """
        pairs = set([])
        for u in range(self.n_words):
            links = [v for v in np.nonzero(self.adj_mat[u])[0]]
            pairs.update(list(zip([u] * len(links), links)))
        return pairs

    def neg_samples(self, u):
        """Get negative samples for node u.

        Args:
          u: Integer.

        Returns:
          List of integers.
        """
        linked = [vp for vp in np.nonzero(self.adj_mat[u])[0]]
        not_linked = [vp for vp in range(self.n_words) if vp not in linked]
        num_negs = min(len(not_linked), self.num_negs)  # effective negs
        return random.sample(not_linked, num_negs)
