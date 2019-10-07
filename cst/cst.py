import numpy as np
import scipy.optimize as opt

from scipy.special import binom
from typing import Union, List, Tuple, Optional


def cls(
    psi: Union[float, List[float], np.ndarray], n1: float, n2: float, norm: bool = True
) -> np.ndarray:
    """Compute class function.

    Parameters
    ----------
    psi : array_like
        Points to evaluate class function for
    n1, n2 : int
        Class function parameters
    norm : bool, optional
        True (default) if the class function should be normalized

    Returns
    -------
    np.array
        Class function value for each given point

    Notes
    -----
    It is assumed all points in psi are between 0 and 1.
    """
    c = (psi ** n1) * ((1.0 - psi) ** n2)
    c /= 1. if not norm or n1 == n2 == 0 else (((n1 / (n1 + n2)) ** n1) * ((n2 / (n1 + n2)) ** n2))
    return c


def bernstein(psi: Union[float, List[float], np.ndarray], r: int, n: int) -> np.array:
    """Compute Bernstein basis polynomial.

    Parameters
    ----------
    psi : array_like
        Points to evaluate the Bernstein polynomial at
    r, n : int
        Bernstein polynomial index and degree

    Returns
    -------
    np.array
        Values of the Bernstein polynomial at the given points

    Notes
    -----
    It is assumed r <= n.
    """
    return binom(n, r) * (psi ** r) * (1.0 - psi) ** (n - r)


def cst(
    x: Union[float, List[float], np.ndarray],
    a: Union[List[float], np.ndarray],
    c: float = 1.0,
    delta: Tuple[float, float] = (0.0, 0.0),
    n1: float = 0.5,
    n2: float = 1.0,
) -> np.ndarray:
    """Compute coordinates of a CST-decomposed curve.

    This function uses the Class-Shape Transformation (CST) method to compute the y-coordinates as a function of a given
    set of x-coordinates, `x`, and a set of coefficients, `a`. The x-coordinates can be scaled by providing a length
    scale, `c`. The starting and ending points of the curve can be displaced by providing non-zero values for `delta`.
    Finally, the class of shapes generated can be adjusted with the `n1` and `n2` parameters. By default, these are 0.5
    and 1.0 respectively, which are good values for generating airfoil shapes.

    Parameters
    ----------
    x : float or array_like
        X-coordinates.
    a : array_like
        Bernstein coefficients.
    c : float
        Scaling length. Default is 1.
    delta : tuple of two floats
        Vertical displacements of the start- and endpoints of the curve. Default is (0., 0.).
    n1, n2 : float
        Class parameters. These determine the general "class" of the shape. They default to n1=0.5 and n2=1.0 for
        airfoil-like shapes.

    Returns
    -------
    y : np.ndarray
        Y-coordinates.

    References
    ----------
    [1] Brenda M. Kulfan, '"CST" Universal Parametric Geometry Representation Method With Applications to Supersonic
     Aircraft,' Fourth International Conference on Flow Dynamics, Sendai, Japan, September 2007.
    """
    # Ensure x is a numpy array
    x = np.atleast_1d(x)

    # Non-dimensional x-coordinates and Bernstein polynomial degree
    psi = x / c
    n = len(a) - 1

    # Compute Class and Shape functions
    _class = cls(psi, n1, n2)
    _shape = sum(a[r] * bernstein(psi, r, n) for r in range(len(a)))

    # Compute normalized coordinates coordinates and return de-normalized coordinates
    eta = _class * _shape + ((1.0 - psi) * delta[0] + psi * delta[1]) / c
    return c * eta


def fit(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    n: int,
    delta: Optional[Tuple[float, float]] = None,
    n1: float = 0.5,
    n2: float = 1.0,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Fit a set of coordinates to a CST representation.

    Parameters
    ----------
    x, y : array_like
        X- and y-coordinates of a curve.
    n : int
        Number of Bernstein coefficients.
    delta : tuple of two floats, optional
        Manually set the start- and endpoint displacements.
    n1, n2 : float
        Class parameters. Default values are 0.5 and 1.0 respectively.

    Returns
    -------
    A : np.ndarray
        Bernstein coefficients describing the curve.
    delta : tuple of floats
        Displacements of the start- and endpoints of the curve.
    """
    # Ensure x and y are np.ndarrays
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(y) != np.ndarray:
        y = np.ndarray(y)

    # Make sure the coordinates are sorted by x-coordinate
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    # Non-dimensionalize coordinates
    c = x[-1] - x[0]
    x = x / c
    y = y / c

    if delta is None:
        delta = (y[0], y[-1])

    def f(_x):
        return np.sqrt(np.mean((y - cst(x, _x, delta=delta, n1=n1, n2=n2)) ** 2))

    # Fit the curve
    res = opt.minimize(f, np.zeros(n))

    return res.x, (y[0] * c, y[-1] * c)
