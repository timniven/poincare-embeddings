"""Data preparation.

The big issue is memory efficiency, especially with large datasets.

Pre-process process:
- .csv in
  -> chunked if necessary
  -> vocab created
  -> bins created for nodes

Folder and file structure as follows:
poincare-embeddings/
  data/
    dataset_name/
      bins/
        bin_aa.pkl
        bin_ab.pkl
        ...
      chunks/
        chunk0.pkl
        chunk1.pkl
        ...
      dataset_name.csv
      dataset_vocab.pkl
"""

import pandas as pd
import glovar
import os
from poincare import PKL
import shutil


#
# Master pre-process function


def preprocess(dataset_name, chunk_size=10000):
    """Perform all pre-processing for the dataset.

    Pre-process process:
      *  .csv already saved
      1) chunked if necessary
      2) vocab created
      3) bins created for nodes

    Args:
      dataset_name: String.
      chunk_size: Integer. Default is 10,000.
    """
    print('Performing all pre-processing for %s...' % dataset_name)
    print('Chunk size will be %s.' % chunk_size)

    # 1. Chunking
    num_chunks = create_chunks(dataset_name, chunk_size)

    # 2. Vocab
    vocab = get_vocab(dataset_name)

    # 3. Bins
    create_bins(dataset_name, num_chunks, vocab)

    print('All preprocessing completed successfully.')


#
# Loading data


def data_file_path(dataset_name):
    """Determine the file path to the .csv for the dataset.

    Args:
      dataset_name: String.

    Returns:
      String.
    """
    return os.path.join(glovar.DATA_DIR, dataset_name, dataset_name + '.csv')


def data_size(dataset_name):
    """Get the size of a dataset.

    Will load the chunks one by one and increment the count. Will also delete
    the chunks, freeing memory upon completion.

    Args:
      dataset_name: String.

    Returns:
      Integer.
    """
    size = 0
    for j in range(count_chunks(dataset_name)):
        chunk = get_chunk(dataset_name, j)
        size += len(chunk)
        del chunk
    return size


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
    file_path = data_file_path(dataset_name)
    df = pd.read_csv(file_path, sep=',', chunksize=chunk_size)
    df.columns = ['child', 'parent']
    return df


#
# Chunking


def chunk_path(dataset_name, j):
    """Get the file path to a chunk.

    Args:
      dataset_name: String.
      j: Integer, the chunk index.

    Returns:
      String.
    """
    return os.path.join(
        glovar.DATA_DIR, dataset_name, 'chunks', '%s%s.pkl' % (dataset_name, j))


def chunk_size(dataset_name):
    """Get the size of the chunks.

    Args:
      dataset_name: String.

    Returns:
      Integer, Integer. The first is the defined chunk size, the second is the
        size of the last chunk (that will be smaller than the defined size).
    """
    num_chunks = count_chunks(dataset_name)
    dataset_size = data_size(dataset_name)
    last_chunk = get_chunk(dataset_name, num_chunks - 1)
    return int((dataset_size - len(last_chunk)) / (num_chunks - 1))


def count_chunks(dataset_name):
    """Count how many chunks there are for a dataset.

    Given the chunks start from 0 index, we can use a range(count_chunks(name))
    to index them.

    Args:
      dataset_name: String.

    Returns:
      Integer.
    """
    folder = os.path.join(glovar.DATA_DIR, dataset_name, 'chunks')
    return len([file_name for file_name in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, file_name))])


def create_chunks(dataset_name, chunk_size=10000):
    """Create chunks from the data.

    Args:
      dataset_name: String.
      chunk_size: Integer, the max number of records in a chunk. Default 10,000.

    Returns:
      Integer: the number of chunks created.
    """
    print('Creating chunks for %s...' % dataset_name)
    data = get_df(dataset_name, chunk_size=chunk_size)
    num_chunks = -1
    for chunk in data:
        num_chunks += 1
        PKL.save(chunk, 'chunk%s' % num_chunks, [dataset_name, 'chunks'])
    assert count_chunks(dataset_name) == num_chunks + 1  # checking it worked
    print('Successfully created %s chunks.' % num_chunks)
    return num_chunks + 1


def get_chunk(dataset_name, j):
    """Get a chunk.

    Args:
      dataset_name: String.
      j: Integer, the chunk index.

    Returns:
      pandas.DataFrame.
    """
    return PKL.load('chunk%s' % j, [dataset_name, 'chunks'])


#
# Vocab dictionaries


class Vocab:
    """Wrapper for a vocab dict.

    Exposes indexing for both strings and integers for forward and reverse
    lookup.

    Attributes:
      dataset_name: String.
      ix: Dictionary, mapping word keys to index values.
      word: Dictionary, mapping index keys to word values.
      n: Integer, the number of words in the vocabulary.
    """

    def __init__(self, dataset_name, words):
        """Create a new Vocab.

        Args:
          words: Set of strings.
        """
        self.dataset_name = dataset_name
        self.ix = dict(zip(words, range(len(words))))
        self.word = {v: k for k, v in self.ix.items()}
        self.n = len(self.ix)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.word[item]
        elif isinstance(item, str):
            return self.ix[item]
        else:
            raise ValueError('Unexpected type: %s.' % type(item))

    def __len__(self):
        return self.n

    def __repr__(self):
        return 'Vocab for %s with %s words.' % (self.dataset_name, self.n)


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
    data = get_df(dataset_name)  # default chunk_size OK
    words = set([])
    for chunk in data:
        words.update(list(chunk['child']))
        words.update(list(chunk['parent']))
    print('Found %s words.' % len(words))
    vocab = Vocab(dataset_name, words)
    print('Pickling...')
    PKL.save(vocab, dataset_name + '_vocab', [dataset_name])
    print('Vocab creation completed successfully.')
    return vocab


def get_vocab(dataset_name):
    """Get a vocab dict.

    If the vocab dict doesn't exist, will create it.

    Args:
      dataset_name: String.

    Returns:
      Dictionary.
    """
    pkl_name = dataset_name + '_vocab'
    sub_folders = [dataset_name]
    if PKL.exists(pkl_name, sub_folders):
        return PKL.load(pkl_name, sub_folders)
    else:
        return create_vocab(dataset_name)


#
# Bins for sampling


class Bin:
    """Bin for u -> pos_ixs.

    Convention is to organize them by first two letters in words. So hash is
    just that.
    """

    def __init__(self, dataset_name, hash):
        """Create a new Bin.

        Args:
          dataset_name: String.
          hash: String, first two letters of the words in this bin.
        """
        self.dataset_name = dataset_name
        self.hash = hash
        self.words = {}
        self.bin_name = dataset_name + '_' + hash

    def __getitem__(self, item):
        return self.words[item]

    def update(self, word1, word2_ix, direction):
        """Update a relation.

        Args:
          word1: String.
          word2_ix: Integer.
          direction: String in {up, down}, defines the direction of the
            relationship.
        """
        if word1 not in self.words.keys():
            self.words[word1] = {}
            self.words[word1]['up'] = set([])
            self.words[word1]['down'] = set([])
        self.words[word1][direction].update([word2_ix])

    def save(self):
        """Save this bin (re-pickle it)."""
        PKL.save(self, self.bin_name, [self.dataset_name, 'bins'])


def create_bins(dataset_name, num_chunks, vocab):
    """Create bins for this dataset.

    Args:
      dataset_name: String.
      num_chunks: Integer.
      vocab: data.preprocess.Vocab.
    """
    print('Creating word bins for %s...' % dataset_name)
    print('Deleting old files...')
    shutil.rmtree(os.path.join(glovar.DATA_DIR, dataset_name, 'bins'))
    print('Creating...')
    for j in range(num_chunks):
        print('Chunk %s...' % j)
        chunk = get_chunk(dataset_name, j)
        for row in chunk.iterrows():
            child, parent = row[1]
            child_bin = get_bin(dataset_name, child)
            child_bin.update(child, vocab[parent], 'up')
            child_bin.save()
            del child_bin
            parent_bin = get_bin(dataset_name, parent)
            parent_bin.update(parent, vocab[child], 'down')
            parent_bin.save()
            del parent_bin
    print('Binning completed successfully.')


def get_bin(dataset_name, word):
    """Get or create a bin.

    Args:
      dataset_name: String.
      word: String.

    Returns:
      data.preprocess.Bin.
    """
    bin_hash = get_hash(word)
    bin_name = dataset_name + '_' + bin_hash
    sub_dirs = [dataset_name, 'bins']
    if PKL.exists(bin_name, sub_dirs):
        return PKL.load(bin_name, sub_dirs)
    else:
        word_bin = Bin(dataset_name, bin_hash)
        PKL.save(word_bin, bin_name, sub_dirs)
        return word_bin


def get_hash(word):
    """Get hash for binning's sake.

    Args:
      word: String.

    Returns:
      String.
    """
    # This hack for the testdata
    if len(word) == 1:
        return word
    word_hash = word[0] + word[1]
    word_hash = word_hash.replace('/', '-fs-')
    word_hash = word_hash.replace('\\', '-bs-')
    return word_hash
