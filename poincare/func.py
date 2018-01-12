"""Mathematical functions.

Credit due here:
https://github.com/TatsuyaShirakawa
https://github.com/qiangsiwei/poincare_embedding (also based on the first)
"""
import torch


EPS = 1e-6


def arcosh(x):
    """Area cosinus hyperbolicus (thanks for the Latin, Wikipedia).

    Beware: the domain of arcosh is [1, +inf]. I haven't implemented a check
    here but bad values will cause trouble.

    Works with vectors.
    """
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def poincare_distance(u, v):
    """Poincaré distance between two vectors.

    Expecting vectors that are two-dimensional. This is to allow for the case
    of comparing a set of vectors in a matrix, vectorizing the implementation of
    negative samples.

    .. math::
        d(u, v) = \arcosh
        \left(
            1 + 2 \frac{\|u - v\|^2}
                       {(1 - \|u\|^2)(1 - \|v\|^2)}
        \right)

    THOUGHTS:
    - Seems hacky to have to clamp values; makes me wonder if there is a better
      way to calculate all this.

    Args:
      u: torch.FloatTensor, always a column vector (2 dimensions).
      v: torch.FloatTensor, sometimes a column vector sometimes a matrix of
        row vectors.

    Returns:
      torch.FloatTensor with scalar value.
    """
    uu = u.t().dot(u)
    # v is sometimes a matrix of negs
    vv = v.norm(dim=1) ** 2  # so norm then square instead of dot product
    uv = u.mm(v.t())  # this works for both vector and matrix case
    numerator = uu - 2 * uv + vv
    alpha = (1 - uu).clamp(min=EPS, max=1-EPS)
    beta = (1 - vv).clamp(min=EPS, max=1-EPS)
    # remembering that the domain of arcosh is [1, +inf]
    gamma = (1 + 2 * numerator / alpha / beta).clamp(min=1.)
    distance = arcosh(gamma)
    return distance
