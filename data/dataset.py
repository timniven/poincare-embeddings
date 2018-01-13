"""Dataset and Collator for training."""
from torch.utils.data import dataset, dataloader
from data import preprocess as pp
import random


class Dataset(dataset.Dataset):
    """Wrapper for a dataset.

    Given a .csv file with U -> V labeled ['child', 'parent'], and a sampling
    strategy, this class wraps access to the dataset, potentially chunked for
    memory efficiency.

    Notes:
    - Randomize order of chunks, then order of rows in the chunks.
    - Designed for use with a PyTorch DataLoader during training.
    - Negative sampling responsibility of Collator in DataLoader, not here.
      Here we are principally concerned with randomizing the order of the
      potentially chunked data.
    """

    def __init__(self, dataset_name, data_size):
        # Master variables
        self.dataset_name = dataset_name
        self.num_chunks = pp.count_chunks(dataset_name)
        self.data_size = data_size

        # Epoch specific variables
        self.chunk_ixs = list(range(self.num_chunks))
        self.current_chunk = list(pp.get_chunk(dataset_name, 0))
        self.local_len = len(self.current_chunk)
        self.local_ixs = list(range(self.local_len))
        self.chunk_i = 0
        self.local_i = 0
        self.global_i = 0

    def __getitem__(self, item):
        # get item
        data = self.next_item()
        self.local_i += 1
        self.global_i += 1
        if self.end_epoch():
            self.new_epoch()
        else:
            if self.end_chunk():
                self.new_chunk()
        return data

    def __len__(self):
        return self.data_size

    def end_chunk(self):
        """Determines if the current chunk is over.

        self.local_i gets iterated after retrieval of an item, so this check
        is performed after that in the __getitem__ method.
        """
        return self.local_i == self.local_len - 1

    def end_epoch(self):
        """Determines if the current epoch is over.

        self.global_i gets iterated after retrieval of an item, so this check
        is performed after that in the __getitem__ method.
        """
        return self.global_i == self.data_size - 1

    def new_chunk(self):
        self.chunk_i += 1
        self.current_chunk = list(pp.get_chunk(self.dataset_name, self.chunk_i))

    def new_epoch(self):
        pass


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


def get_data_loader(dataset, collator):
    return dataloader.DataLoader(
        dataset,
        1,
        shuffle=True,
        num_workers=4,
        collate_fn=collator)
