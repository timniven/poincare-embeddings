"""Plotting embeddings.

Credit to:
https://github.com/TatsuyaShirakawa/poincare-embedding/blob/master/scripts/plot_mammal_subtree.py
"""
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_embs(embs, vocab):
    """Creates a plot of the embeddings."""
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax.add_artist(circle)

    for word, ix in vocab.items():
        x, y = embs[ix]
        ax.plot(x, y, 'o', color='y')
        ax.text(x+0.001, y+0.001, word, color='b', alpha=0.6)
    plt.show()


def plot_change(embs1, embs2, vocab):
    pass
