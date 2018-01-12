"""WordNet data.

Credit:
https://github.com/TatsuyaShirakawa/poincare-embedding/blob/master/scripts/create_wordnet_noun_hierarchy.py

Notes:
- My convention is to store the pairwise relation data in .csv files. From there
  poincare.data is used to transform, sample, etc. The file should have the
  form dataset_name.csv. So here, wordnet.csv is the output. And this file lives
  in glovar.DATA_DIR.
"""

from nltk.corpus import wordnet as wn
import glovar
import os
import pandas as pd


DEFAULT_FILE_PATH = os.path.join(glovar.DATA_DIR, 'wordnet.csv')


def transitive_closure(synsets):
    hypernyms = set([])
    for s in synsets:
        paths = s.hypernym_paths()
        for path in paths:
            hypernyms.update((s, h) for h in path[1:] if h.pos() == 'n')
    return hypernyms


def generate_wordnet(target=DEFAULT_FILE_PATH):
    """Generates wordnet data.

    Args:
      target: String. Defaults to glovar.DATA_DIR + 'wordnet.csv'.
    """
    print('Generating WordNet synsets and writing to %r' % target)
    words = wn.words()
    nouns = set([])
    print('Updating noun set...')
    for word in words:
        nouns.update(wn.synsets(word, pos='n'))
    print('Getting transitive closure...')
    hypernyms = list(transitive_closure(nouns))
    print('Writing file...')
    with open(target, 'w') as file:
        for n1, n2 in hypernyms:
            file.write('%s,%s\n' % (n1.name(), n2.name()))
    print('Success.')


def get_wordnet(target=DEFAULT_FILE_PATH):
    df = pd.read_csv(target, header=None, sep=',')
    df.columns = ['child', 'parent']
    return df
