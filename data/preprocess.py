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


def create_vocab(dataset_name):
    """Create vocab dict for a dataset.

    We assume we already have a dataset_name.csv file in the data/ folder with
    children in the first column and parents in the right.

    Args:
      dataset_name: String.

    Returns:
      Dictionary.
    """
    print('Creating vocab dictionary for %s' % dataset_name)
    data = get_df(dataset_name)
    words = set([])
    for chunk in data:
        words.update(list(chunk['child']))
        words.update(list(chunk['parent']))
    n_words = len(words)
    print('Found %s words.' % n_words)
    vocab = dict(zip(words, range(n_words)))
    print('Pickling...')
    PKL.save(vocab, dataset_name + '_vocab', [dataset_name])
    print('Success.')
    return vocab


def dataset_file_path(dataset_name):
    """Determine the file path to the .csv for the dataset.

    Args:
      dataset_name: String.

    Returns:
      String.
    """
    return os.path.join(glovar.DATA_DIR, dataset_name, dataset_name + '.csv')


def get_df(dataset_name, chunk_size=10000):
    """Get the dataframe for a dataset.

    We assume we already have a dataset_name.csv file in the data/ folder with
    children in the first column and parents in the right.

    Args:
      dataset_name: String.
      chunk_size: Int, for loading the DataFrame in chunks for large files.
        Default is 10,000.

    Returns:
      pandas.DataFrame.
    """
    file_path = dataset_file_path(dataset_name)
    df = pd.read_csv(file_path, sep=',', chunksize=chunk_size)
    df.columns = ['child', 'parent']
    return df


def get_vocab(dataset_name):
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
        return create_vocab(dataset_name)


def preprocess(dataset_name):
    """Due to memory issues, try this instead...

    Iterate over each line of the dataset
    Keep a list of linked nodes for each node

    """
    data = get_df(dataset_name)  # chunked by 10,000s
    vocab = get_vocab(dataset_name)
    for chunk in data:
        for row in chunk.iterrows():
            child, parent = row[1]
            child_bin = get_bin(dataset_name, child)
            parent_bin = get_bin(dataset_name, parent)
            child_bin.update(parent)
            parent_bin.update(child)


            a = {
                'word': {
                    'up': [1, 3, 4],
                    'down': [7, 10]
                }
            }


def get_bin(dataset_name, word):
    bin_name = dataset_name + '_' + word[0] + word[1]
    if PKL.exists(bin_name):
        return PKL.load(bin_name)


