import numpy as np
import unittest
import itertools

from parameterized import parameterized

from cst import cst, fit


def rms(e):
    """
    Compute the root-mean-squared of an error.

    Parameters
    ----------
    e : array_like
        Error

    Returns
    -------
    array_like
        Root-mean-squared of the error
    """
    return np.sqrt(np.mean(e * e))


class TestCSTParabola(unittest.TestCase):
    """
    Set of tests based on fitting a parabola with the CST.
    """

    def setUp(self) -> None:
        self.x = np.linspace(0, 1)
        self.y = self.x * self.x

    def rms(self, y_fit):
        e = y_fit - self.y
        return rms(e)

    @parameterized.expand([
        (f"n={param[0]},delta {'computed' if param[0] is None else 'fixed'}",) + param
        for param in itertools.product((3, 6, 12), ((0, 0), None), (6,))
    ])
    def test(self, _, n, delta, precision):
        a, delta = fit(self.x, self.y, n, delta=delta, n1=0, n2=0)
        y_fit = cst(self.x, a, delta=delta, n1=0, n2=0)
        self.assertAlmostEqual(self.rms(y_fit), 0.0, precision)


class TestCSTPropeller(unittest.TestCase):
    """
    Test case based on fitting the distribution curves of a references propeller blade.
    """

    def setUp(self) -> None:
        self.geom = np.loadtxt("nr640_blade.dat")

    @parameterized.expand([("n=3", 3, 0.25), ("n=6", 6, 0.02), ("n=12", 12, 0.002)])
    def test(self, _, n, precision):
        for i in range(1, 4):
            a, _ = fit(self.geom[:, 0], self.geom[:, i], n, n1=0, n2=0, delta=(0, 0))
            y_fit = cst(self.geom[:, 0], a, n1=0, n2=0)
            e = (y_fit - self.geom[:, i]) / np.mean(self.geom[:, i])

            self.assertLess(rms(e), precision)


class TestCSTAirfoil(unittest.TestCase):
    """
    Test case based on fitting a Clark-Y airfoil.
    """

    def setUp(self) -> None:
        coords = np.loadtxt("clark-y.dat")

        i = np.argmin(coords[:, 0])
        coords_upper = np.flipud(coords[: i + 1])
        coords_lower = coords[i:]
        self.x = (1 - np.cos(np.linspace(0, 1) * np.pi)) / 2
        self.y_u = np.interp(self.x, coords_upper[:, 0], coords_upper[:, 1])
        self.y_l = np.interp(self.x, coords_lower[:, 0], coords_lower[:, 1])

    @parameterized.expand([("n=3", 3, 0.002), ("n=6", 6, 0.002), ("n=12", 12, 0.002)])
    def test(self, _, n, precision):
        for y in [self.y_u, self.y_l]:
            a, (_, d_te) = fit(self.x, y, n)
            y_fit = cst(self.x, a, delta=(0, d_te))
            e = y_fit - y

            self.assertLess(rms(e), precision)


if __name__ == "__main__":
    unittest.main()
