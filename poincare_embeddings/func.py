"""Mathematical functions."""
import numpy as np
import torch


STABILITY = 1e-6


def arcosh(x):
    """Area cosinus hyperbolicus (thanks for the Latin, Wikipedia).

    NOTE: not implemented for vectors. Only use seems to be in Poincare
    distance function. Leaving as is.

    Bewar: the domain of arcosh is [1, +inf]. I haven't implemented a check
    here but bad values will cause trouble.

    Args:
      x: Float.

    Returns:
      Float.
    """
    return torch.log(x + torch.sqrt(x**2 - 1))


def euclidean_distance(u, v):
    """Calculate the Euclidean distance.

    Args:
      u: FloatTensor.
      v: FloatTensor.

    Returns:
      Float.
    """
    return torch.dist(u, v)


def l2_squared(x):
    """Get the L2-norm for a vector.

    Args:
      x: FloatTensor. Must have dimension 1.

    Returns:
      Float.
    """
    return x.dot(x)


def poincare_distance(u, v):
    """Calculate the Poincaré distance.

    Make sure the norms of u and v are less than 1 or there will be trouble.
    This function therefore utilizes clipping to prevent overflow.

    .. math::
        d(u, v) = \arcosh
        \left(
            1 + 2 \frac{\|u - v\|^2}
                       {(1 - \|u\|^2)(1 - \|v\|^2)}
        \right)

    Args:
      u: FloatTensor with L2 norm less than 1.
      v: FloatTensor with L2 norm less than 1.

    Returns:
      Float.
    """
    uu = u.dot(u)
    vv = v.dot(v)
    u_min_v = u - v
    uv = u_min_v.dot(u_min_v)
    alpha = (1 - uu).clamp(min=STABILITY)
    beta = (1 - vv).clamp(min=STABILITY)
    gamma = (1 + 2 * (uv / (alpha * beta))).clamp(min=1 + STABILITY)
    distance = arcosh(gamma)
    return distance
