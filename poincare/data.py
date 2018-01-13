"""Data preparation.

Notes:
- Shirakawa's implementation allows for left, right, and both sampling.
- I still don't understand how the undirected graph from the paper should work.
- It's still not very unsupervised, I'd like a totally new way to train from
  pure text.

Decisions:
- Raw data comes in .csv format.
- Must first create a vocab dict from this data.
- Then I use the .csv and the vocab dict to construct an adjacency matrix, given
  conditions (i.e. directed edges or not).
- I use the adjacency matrix to determine negative samples that really don't
  appear in the list.
"""

import pandas as pd
import glovar
import os
from poincare import PKL
import numpy as np


def adjacency_name(dataset_name, mode):
    """Utility function to get adjacency matrix name.

    Args:
      dataset_name: String.
      mode: String in {up, down, both}, determining the nature of the graph.

    Returns:
      String.
    """
    return '%s_adj_%s' % (dataset_name, mode)


def create_adjacency(dataset_name, mode):
    """Create the adjacency matrix.

    Should already have the data and vocab dict in data/ folder for the dataset.

    As for mode, we have:
      up: directed edges from children to parents.
      down: directed edges from parents to children.
      both: undirected edges (i.e. an edge going both ways).

    Args:
      dataset_name: String.
      mode: String in {up, down, both}, determining the nature of the graph.

    Returns:
      numpy.ndarray.

    Raises:
      ValueError if mode not in {up, down, both}.
    """
    print('Creating %s mode adjacency matrix for %s...' % (mode, dataset_name))
    print('Getting data and vocab...')
    data = get_df(dataset_name)
    vocab = get_vocab_dict(dataset_name)
    n_words = len(vocab)
    adj_mat = np.zeros((n_words, n_words), 'int32')
    print('Determining edges...')
    for i, row in data.iterrows():
        child_ix, parent_ix = vocab[row[0]], vocab[row[1]]
        if mode == 'up':
            adj_mat[child_ix, parent_ix] = 1
        elif mode == 'down':
            adj_mat[parent_ix, child_ix] = 1
        elif mode == 'both':
            adj_mat[child_ix, parent_ix] = 1
            adj_mat[parent_ix, child_ix] = 1
        else:
            raise ValueError('Unexpected mode %r' % mode)
    print('Saving...')
    PKL.save(adj_mat, adjacency_name(dataset_name, mode))
    print('Success.')
    return adj_mat


def create_vocab_dict(dataset_name):
    """Create vocab dict for a dataset.

    We assume we already have a dataset_name.csv file in the data/ folder with
    children in the first column and parents in the right.

    Args:
      dataset_name: String.

    Returns:
      Dictionary.
    """
    print('Creating vocab dictionary for %s' % dataset_name)
    df = get_df(dataset_name)
    words = set(df['child'].values + df['parent'].values)
    vocab = dict(zip(words, range(len(words))))
    print('Pickling...')
    PKL.save(vocab, '%s_vocab' % dataset_name)
    print('Success.')
    return vocab


def dataset_file_path(dataset_name):
    """Determine the file path to the .csv for the dataset.

    Args:
      dataset_name: String.

    Returns:
      String.
    """
    return os.path.join(glovar.DATA_DIR, '%s.csv' % dataset_name)


def get_adjacency(dataset_name, mode):
    """Get the adjacency matrix.

    Will create if not found.

    Args:
      dataset_name: String.
      mode: String in {up, down, both}, determining the nature of the graph.

    Returns:
      numpy.ndarray.
    """
    name = adjacency_name(dataset_name, mode)
    if PKL.exists(name):
        return PKL.load(name)
    else:
        return create_adjacency(dataset_name, mode)


def get_df(dataset_name):
    """Get the dataframe for a dataset.

    We assume we already have a dataset_name.csv file in the data/ folder with
    children in the first column and parents in the right.

    Args:
      dataset_name: String.

    Returns:
      pandas.DataFrame.
    """
    file_path = dataset_file_path(dataset_name)
    df = pd.read_csv(file_path, header=None, sep=',')
    df.columns = ['child', 'parent']
    return df


def get_vocab_dict(dataset_name):
    """Get a vocab dict.

    If the vocab dict doesn't exist, will create it.

    Args:
      dataset_name: String.

    Returns:
      Dictionary.
    """
    if PKL.exists('%s_vocab' % dataset_name):
        return PKL.load('%s_vocab' % dataset_name)
    else:
        return create_vocab_dict(dataset_name)
