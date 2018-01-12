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


def plot_change(embs1, embs2, changed_ixs, vocab):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()

    max_norm = max(embs1.norm(p=2, dim=1).max(),
                   embs2.norm(p=2, dim=1).max())
    lim = max_norm * 1.1
    text_offset = max_norm / 100.

    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))

    circle = plt.Circle((0, 0), lim, color='black', fill=False)
    ax.add_artist(circle)

    for word, ix in vocab.items():
        if ix in changed_ixs:
            x1, y1 = embs1[ix]
            x2, y2 = embs2[ix]
            ax.plot(x1, y1, 'o', color='y')
            ax.plot(x2, y2, 'x', color='r')
            ax.text(x1 + text_offset, y1 + text_offset, word, color='b',
                    alpha=0.6)
            ax.text(x2 + text_offset, y2 + text_offset, word, color='b',
                    alpha=0.6)

    plt.show()
