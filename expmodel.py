import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


class Logarithmic1D(Fittable1DModel):
    """
    One dimensional logarithmic model.
    Parameters
    ----------
    amplitude : float, optional
    x_0 : float, optional
    See Also
    --------
    Exponential1D, Gaussian1D
    """

    amplitude = Parameter(default=1)
    x_0 = Parameter(default=1)

    @staticmethod
    def evaluate(x, amplitude, x_0):
        return amplitude * np.log(x / x_0)

    @staticmethod
    def fit_deriv(x, amplitude, x_0):
        d_amplitude = np.log(x / x_0)
        d_x_0 = np.zeros(x.shape) - (amplitude / x_0)
        return [d_amplitude, d_x_0]

    @property
    def inverse(self):
        new_amplitude = self.x_0
        new_x_0 = self.amplitude
        return Exponential1D(amplitude=new_amplitude, x_0=new_x_0)

    @x_0.validator
    def x_0(self, val):
        if val == 0:
            raise ValueError("0 is not an allowed value for x_0")


class Exponential1D(Fittable1DModel):
    """
    One dimensional exponential model.
    Parameters
    ----------
    amplitude : float, optional
    x_0 : float, optional
    See Also
    --------
    Exponential1D, Gaussian1D
    """
    amplitude = Parameter(default=1)
    x_0 = Parameter(default=1)

    @staticmethod
    def evaluate(x, amplitude, x_0):
        return amplitude * np.exp(x / x_0)

    @staticmethod
    def fit_deriv(x, amplitude, x_0):
        d_amplitude = np.exp(x / x_0)
        d_x_0 = -amplitude * np.exp(x / (x_0 ** 2))
        return [d_amplitude, d_x_0]

    @property
    def inverse(self):
        new_amplitude = self.x_0
        new_x_0 = self.amplitude
        return Logarithmic1D(amplitude=new_amplitude, x_0=new_x_0)

    @x_0.validator
    def x_0(self, val):
        if val == 0:
            raise ValueError("0 is not an allowed value for x_0")
