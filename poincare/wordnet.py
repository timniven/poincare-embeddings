"""WordNet data.

Credit:
https://github.com/TatsuyaShirakawa/poincare-embedding/blob/master/scripts/create_wordnet_noun_hierarchy.py
"""
from nltk.corpus import wordnet as wn
import glovar


def transitive_closure(synsets):
    hypernyms = set([])
    for s in synsets:
        paths = s.hypernym_paths()
        for path in paths:
            hypernyms.update((s, h) for h in path[1:] if h.pos() == 'n')
    return hypernyms


def generate_wordnet_synsets(target_file_path=glovar.DATA_DIR):
    """Generates wordnet data.

    Args:
      target_file_path: String. Defaults to glovar.DATA_DIR.
    """
    print('Generating WordNet synsets and writing to %r' % target_file_path)
    words = wn.words()
    nouns = set([])
    print('Updating noun set...')
    for word in words:
        nouns.update(wn.synsets(word, pos='n'))
    print('Getting transitive closure...')
    hypernyms = list(transitive_closure(nouns))
    print('Writing file...')
    with open(target_file_path, 'w') as file:
        for n1, n2 in hypernyms:
            file.write('%s\t%s\n' % (n1.name(), n2.name()))
    print('Success.')
