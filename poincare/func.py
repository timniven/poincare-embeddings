"""Mathematical functions.

Credit due here:
https://github.com/TatsuyaShirakawa
https://github.com/qiangsiwei/poincare_embedding (also based on the first)
"""
import torch
import glovar


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
      u: torch.FloatTensor, always a row vector (2 dimensions).
      v: torch.FloatTensor, sometimes a row vector sometimes a matrix of
        row vectors.

    Returns:
      torch.FloatTensor with scalar value.
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    uu = u.t().dot(u)
    # v is sometimes a matrix of negs
    vv = v.norm(dim=1) ** 2  # so norm then square instead of dot product
    uv = u.mm(v.t())  # this works for both vector and matrix case
    numerator = uu - 2 * uv + vv
    alpha = (1 - uu).clamp(min=glovar.EPS, max=1 - glovar.EPS)
    beta = (1 - vv).clamp(min=glovar.EPS, max=1 - glovar.EPS)
    # remembering that the domain of arcosh is [1, +inf]
    gamma = (1 + 2 * numerator / alpha / beta).clamp(min=1.)
    distance = arcosh(gamma)
    return distance


def project(theta):
    """Project embeddings to remain within the Poincare ball.

    Args:
      theta: Tensor, the embedding matrix.

    Returns:
        Tensor.
    """
    norm = theta.norm(p=2, dim=1).unsqueeze(1)
    norm[norm < 1] = 1
    norm[norm >= 1] += glovar.EPS
    return theta.div(norm)


def scale_grad(theta):
    """Scale Euclidean gradient to hyperbolic gradient.

    Args:
      theta: Tensor, the vector or matrix in the calculation. Is not the grad,
        is the Variable containing the value and the grad.

    Returns:
      Tensor:
        1D if originally a vector; 2D if a matrix.
    """
    if theta.dim() == 2:
        return (((1 - theta.norm(dim=1)**2)**2) / 4.).data.unsqueeze(1) \
               * theta.grad.data
    return (((1 - theta.norm()**2)**2) / 4.).data * theta.grad.data
