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
        self.chunk_size, self.last_chunk_size = pp.chunk_size(dataset_name)
        self.vocab = pp.get_vocab(dataset_name)
        self.i = 0
        self.chunk_ixs = list(range(self.num_chunks))      # for chunk selection
        self.local_ixs = list(range(self.chunk_size))      # ixs within chunk
        self.last_ixs = list(range(self.last_chunk_size))  # ixs in last chunk
        random.shuffle(self.chunk_ixs)
        random.shuffle(self.local_ixs)
        random.shuffle(self.last_ixs)

    def __getitem__(self, item):
        """Get the next item.

        Find the correct randomized bin for the item, loads bin, returns u and
        v, then deletes the bin from memory.

        Args:
          item: will be an integer index.

        Returns:
          Tuple (Integer u, Integer v), representing one observed relationship
            in the data. This will then be given to the collate function which
            will refer to the Sampler to find the negative samples.
        """
        # Map the index to chunk and local ixs without considering randomization
        chunk_ix = int(np.floor(item / self.chunk_size))
        local_ix = item - self.chunk_size * chunk_ix

        # Get the randomized chunk_ix and load the chunk
        rand_chunk_ix = self.chunk_ixs[chunk_ix]
        chunk = pp.get_chunk(self.dataset_name, rand_chunk_ix)

        # Get random local ix and find the local item
        if chunk_ix == self.chunk_size - 1:
            rand_local_ix = self.last_ixs[local_ix]
        else:
            rand_local_ix = self.local_ixs[local_ix]
        u, v = chunk.iloc[local_ix]

        # Delete the chunk
        del chunk

        # Iterate the global counter
        self.i += 1

        # Reshuffle and reset when the epoch ends
        if self.i == self.data_size:
            self.i = 0
            random.shuffle(self.chunk_ixs)
            random.shuffle(self.local_ixs)
            random.shuffle(self.last_ixs)
        return u, v

    def __len__(self):
        return self.data_size


class Collator:
    """Collate function for training."""

    def __init__(self, sampler):
        """Create a new Collator.

        Args:
          sampler: data.sampling.Sampler.
        """
        self.sampler = sampler

    def __call__(self, data):
        """Execute collate function.

        Args:
          data: Tuple of integers (u, v).

        Returns:
          Tuple (u_ix, v_ix, [neg_ixs]).
        """
        u, v = data
        return u, v, self.sampler(u)


def get_data_loader(dataset, collator, batch_size=1, shuffle=False):
    """Get a DataLoader.

    Args:
      dataset: torch.utils.data.dataset.Dataset.
      collator: collate function.
      batch_size: Integer, defaults to 1.
      shuffle: Bool, whether to shuffle. Defaulted to False since I implemented
        this logic in the Dataset class above to account for the chunking and
        memory management.

    Returns:
      torch.utils.data.dataloader.DataLoader
    """
    return dataloader.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collator)
