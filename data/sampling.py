"""
For sampling epoch data.

General Notes
--

Shirakawa introduces three sampling strategies: left, right, both. On the
convention that we have two columns, U -> V, where U is the hyponym and V
the hypernym:
  - left sampling considers all hyponyms, excluding those hyponyms of U.
    Overall it therefore considers all nodes except the root of the tree.
  - right sampling considers all hypernyms, excluding hypernyms of U.
    Overall it therefore considers all nodes except the leaves of the tree.
  - both considers all words, except hypo- or hypernyms of U.

Considering the layout of the following diagrams:
https://github.com/qiangsiwei/poincare_embedding
I hypothesize that excluding words from negative sampling reduces the amount
of gradient they receive each epoch, meaning they don't move as far. This
may explain why right sampling appears less spread out than left sampling,
and why both sampling is more spread out than the other two. I intend to
quantify this with an experiment to test it.

Implementation Notes
--

For memory's sake, the data is stored in bins indexed by the first two
letters of the words. In each bin we have a record looking like:
    {'word': {'up': [ixs of word's hypernyms]},
             {'down': [ixs of word's hyponyms]}}

Our algorithm is then as follows. For each U:
  neg_ixs = [u_ix]
  ruled_out = get_ruled_out(u, sampling_strategy)  # looks up bin
  num_negs = min(num_negs, len(vocab) - len(ruled_out))
  while len(neg_ixs) != num_negs:
      neg_ix = random.choice(all_ixs)
      if neg_ix not in ruled_out:
          neg_ixs.append(neg_ix)
  return neg_ixs

I believe the purpose of adding u to the N(u) set is to ensure it isn't an
empty set and therefore the loss function in the paper doesn't encounter a
divide by zero.
"""


from data import preprocess
import random


class Sampler:
    """For getting negative samples."""

    def __init__(self, dataset_name, vocab, mode, num_negs):
        """Create a new Sampler.

        Args:
          dataset_name: String.
          vocab: data.preprocess.Vocab.
          mode: String in {up, down, both}.
          num_negs: Integer, the number of negatives to sample for each word.
        """
        self.dataset_name = dataset_name
        self.vocab = vocab
        self.mode = mode
        self.num_negs = num_negs
        self.all_ixs = list(range(len(vocab)))

    def __call__(self, u):
        return self.sample_neg_ixs(u)

    def get_ruled_out(self, word):
        """Get list of positive ixs ruled out for negative sampling.

        Args:
          word: String.

        Returns:
          List of integers.
        """
        word_bin = preprocess.get_bin(self.dataset_name, word)
        relatives = word_bin[word]
        if self.mode == 'up':
            return relatives['up']
        elif self.mode == 'down':
            return relatives['down']
        elif self.mode == 'both':
            return relatives['up'] + relatives['down']
        else:
            raise ValueError('Unrecognized mode: %r' % self.mode)

    def sample_neg_ixs(self, u):
        """Get negative sample ixs.

        Args:
          u: String, the word in position u.

        Returns:
          List of integer indices representing negative samples.
        """
        neg_ixs = [self.vocab[u]]
        ruled_out = self.get_ruled_out(u)
        num_negs = min(self.num_negs, len(self.vocab) - len(ruled_out))
        while len(neg_ixs) != num_negs:
            neg_ix = random.choice(self.all_ixs)
            if neg_ix not in ruled_out and neg_ix not in neg_ixs:
                neg_ixs.append(neg_ix)
        return neg_ixs
