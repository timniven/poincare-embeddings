"""Dataset and Collator for training."""
from torch.utils.data import dataset, dataloader
from data import preprocess as pp
import random
import numpy as np


class Dataset(dataset.Dataset):
    """Wrapper for a dataset.

    Given a .csv file with U -> V labeled ['child', 'parent'], and a sampling
    strategy, this class wraps access to the dataset, potentially chunked for
    memory efficiency.

    Here we are principally concerned with randomizing the order of the
    potentially chunked data.

    The data is accessed from this class by index, so we need to map this to a
    chunk index then a local index in the chunk.
    """

    def __init__(self, dataset_name):
        """Create a new Dataset.

        Args:
          dataset_name: String.
        """
        self.dataset_name = dataset_name
        self.num_chunks = pp.count_chunks(dataset_name)
        self.data_size = pp.data_size(dataset_name)
        self.chunk_size, self.small_chunk_size = pp.chunk_size(dataset_name)
        self.vocab = pp.get_vocab(dataset_name)
        self.i = 0
        self.chunk_ixs = list(range(self.num_chunks))      # for chunk selection
        self.local_ixs = list(range(self.chunk_size))      # ixs within chunk
        self.small_ixs = list(range(self.small_chunk_size))  # ixs in last chunk
        self.shuffle()

    def __getitem__(self, ix):
        """Get the next item.

        Find the correct randomized bin for the item, loads bin, returns u and
        v, then deletes the bin from memory.

        Args:
          ix: will be an integer index.

        Returns:
          Tuple (Integer u, Integer v), representing one observed relationship
            in the data. This will then be given to the collate function which
            will refer to the Sampler to find the negative samples.
        """
        # Considering the small chunk could be anywhere, map the incoming ix
        # to a chunk, and on that basis select the random chunk index
        chunk_ix, small_status = self.get_chunk_ix(ix)
        rand_chunk_ix = self.chunk_ixs[chunk_ix]

        # Get the index we want inside the chunk
        local_ix = self.get_local_ix(ix, chunk_ix, small_status)
        if small_status == 'in':
            rand_local_ix = self.small_ixs[local_ix]
        else:
            rand_local_ix = self.local_ixs[local_ix]

        # Get the chunk and record, then delete chunk
        chunk = pp.get_chunk(self.dataset_name, rand_chunk_ix)
        u, v = chunk.iloc[rand_local_ix]
        del chunk

        # Iterate the global counter
        self.i += 1

        # Reshuffle and reset when the epoch ends
        if self.i == self.data_size:
            self.i = 0
            self.shuffle()

        return u, v

    def __len__(self):
        return self.data_size

    def get_chunk_ix(self, ix):
        """Get the chunk index.

        A simple view would use:
          int(np.floor(item / self.chunk_size))
        but the problem is we don't know where the shorter, final chunk is in
        the randomized chunk order.

        We therefore require this method to perform that more complicated
        calculation. It is simpler here thanks to small_chunk_bounds being pre-
        calculated with each random shuffle.

        Args:
          ix: Integer, the index passed to __getitem__.

        Returns:
          (Integer ix, String small_status). The small status tells the status
            of the small chunk in relation to this index. Options are 'before',
            'in', or 'after'. This facilitates other computations later.
        """
        if ix < self.small_chunk_start:
            return int(np.floor(ix / self.chunk_size)), 'before'
        elif ix < self.small_chunk_end:
            return self.small_chunk_pos, 'in'
        else:
            prev_count = (self.small_chunk_pos - 1) * self.chunk_size \
                         + self.small_chunk_size
            remaining = ix - prev_count
            return self.small_chunk_pos \
                   + int(np.floor(remaining / self.chunk_size)), \
                   'after'

    def get_local_ix(self, ix, chunk_ix, small_status):
        """Get local index, having already decided which chunk to choose from.

        Args:
          ix: Integer.
          chunk_ix: Integer.
          small_status: String in {before, in, after}.

        Returns:
          Integer.
        """
        if small_status == 'after':
            return (ix - self.small_chunk_size) % self.chunk_size
        else:
            return ix - (self.chunk_size * chunk_ix)

    def shuffle(self):
        """Perform random shuffling."""
        random.shuffle(self.chunk_ixs)
        random.shuffle(self.local_ixs)
        random.shuffle(self.small_ixs)
        # For later calculations: store the ixs of small chunk start and end
        self.small_chunk_pos = np.argmax(self.chunk_ixs)
        self.small_chunk_start, self.small_chunk_end = self.small_chunk_bounds()

    def small_chunk_bounds(self):
        """Get start and end indices of the small chunk in randomized order.

        Returns:
          (Integer start, Integer end).
        """
        start = int((self.small_chunk_pos) * self.chunk_size)
        end = start + self.small_chunk_size
        return start, end


class Collator:
    """Collate function for training."""

    def __init__(self, sampler, vocab):
        """Create a new Collator.

        Args:
          sampler: data.sampling.Sampler.
          vocab: data.preprocess.Vocab.
        """
        self.sampler = sampler
        self.vocab = vocab

    def __call__(self, data):
        """Execute collate function.

        Args:
          data: Tuple of integers (u, v).

        Returns:
          Tuple (u_ix, v_ix, [neg_ixs]).
        """
        u, v = data[0]
        negs = ' '.join([self.vocab[n] for n in self.sampler(u)])
        #print('%s => %s ; %s' % (u, v, negs))
        return self.vocab[u], self.vocab[v], self.sampler(u)


def get_data_loader(data, collator, batch_size=1, shuffle=False):
    """Get a DataLoader.

    Args:
      data: torch.utils.data.dataset.Dataset.
      collator: collate function.
      batch_size: Integer, defaults to 1.
      shuffle: Bool, whether to shuffle. Defaulted to False since I implemented
        this logic in the Dataset class above to account for the chunking and
        memory management.

    Returns:
      torch.utils.data.dataloader.DataLoader
    """
    return dataloader.DataLoader(
        data,
        batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collator)
