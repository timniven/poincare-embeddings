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


def create_adjacency(dataset_name, directed=True):
    """Create the adjacency matrix.

    Should already have the data and vocab dict in data/ folder for the dataset.

    We imagine the arrows are going up the tree. When confronted with a child,
    parent pair, we say the edge goes from child to parent, an "is-a" relation.
    For

    Args:
      dataset_name: String.
      directed: Bool. If true we only include directed edges. If False we will
        add edges between both nodes for each relationship.

    Returns:
      numpy.ndarray.
    """
    data = get_df(dataset_name)
    vocab = get_vocab_dict(dataset_name)
    n_words = len(vocab)
    adj_mat = np.zeros((n_words, n_words), 'int32')
    for i, row in data.iterrows():
        child_ix, parent_ix = vocab[row[0]], vocab[row[1]]
        adj_mat[child_ix, parent_ix] = 1
        if not directed:
            adj_mat[parent_ix, child_ix] = 1
    PKL.save(adj_mat, '%s_adj' % dataset_name)
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
    words = set(df['child'].values + df['parent'].values())
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


def get_adjacency(dataset_name):
    """Get the adjacency matrix.

    This one won't create if not found (since args required: directed) and will
    therefore raise an exception if not found.

    Args:
      dataset_name: String.

    Returns:
      numpy.ndarray.
    """
    return PKL.load(dataset_name + '_adj')


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
