"""Samplers for data preparation."""
from poincare import data


class Sampler:
    """For sampling epoch data."""

    def __init__(self, dataset_name, mode):
        """Create a new Sampler.

        Args:
          dataset_name: String.
          mode: String in {}, determining sampling strategy.
        """
        self.adj_mat = data.get_adjacency(dataset_name, mode)

    def epoch_data(self):
        pass
