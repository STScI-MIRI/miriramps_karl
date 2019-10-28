import numpy as np

from expmodel import Exponential1D
from astropy.modeling.models import Polynomial1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter


__all__ = ["get_ramp", "get_good_ramp", "fit_diffs", "calc_lincor"]


def get_ramp(hdu, pix_x, pix_y, int_num, min_dn=0.0, max_dn=55000.0):
    """
    Get the ramp data for a single pixel for a single integration in an exposure

    Parameters
    ----------
    hdu : astropy.fits.hdu
        header data unit

    pix_x, pix_y : int
        x, y pixel values (starting at 0, 0)

    int_num : int
        integration number (starts at 0)

    Returns
    -------
    gnum, ydata : tuple of ndarrays
        gnum is the group numbers (starting at 0)
        ydata is the DN values at each group
    """
    ngrps = hdu.header["NGROUPS"]
    k1 = int_num * ngrps
    k2 = k1 + ngrps
    gnum = np.array(range(ngrps))
    ydata = (hdu.data[k1:k2, pix_x, pix_y]).astype(float)

    return (gnum, ydata)


def get_good_ramp(gnum, ydata, min_dn=0.0, max_dn=55000.0):
    """
    Get the data on the good portions of a ramp

    Parameters
    ----------
    gnum : ndarray
        gnum is the group numbers (starting at 0)

    ydata : ndarray
        ydata is the DN values at each group

    min_dn : float
        minimum DN value to use [default=0]

    max_dn : float
        maximum DN value to use [default=55000]

    Returns
    -------
    ggnum, gdata, aveDN, diffDN : tuple of ndarrays
        ggnum is group numbers of the good data
        gdata is the DN values at each good group
        aveDN is the average values between each two groups
        diffDN is the difference between each two groups (2pt diff or CDS)
    """
    # create clean versions of the data
    gindxs, = np.where((ydata > min_dn) & (ydata < max_dn))
    gdata = ydata[gindxs]
    ggnum = gnum[gindxs]

    diffDN = np.diff(gdata)
    aveDN = 0.5 * (gdata[:-1] + gdata[1:])

    return (ggnum, gdata, aveDN, diffDN)


def fit_diffs(x, y):
    """
    Fit the average DN versus 2pt differences with a model that accounts
    for the non-linearities and RSCD exponential decay at beginning of ramp

    Parameters
    ----------
    x : ndarray
        average DN values for each 2pt difference

    y : ndarray
        2pt differences

    Returns
    -------
    mod : astropy model
        fitted model
    """
    mod_init = Exponential1D(
        x_0=-2000.0,
        amplitude=2500.0,
        bounds={"amplitude": [100.0, 5000.0], "x_0": [-10000.0, 0.0]},
    ) + Polynomial1D(degree=2)

    fit = LevMarLSQFitter()

    mod = fit(mod_init, x, y, maxiter=10000)

    return mod


def calc_lincor(mod2pt, DNvals, startDN):
    """
    Calculate the linearity correction based on the fit to the 2pt differences
    and a single ramp

    Parameters
    ----------
    mod2pt : astropy model
        model giving the 2pt difference versus DN

    DNvals : ndarray
        measured DN values for a single ramp

    Returns
    -------

    """
    # difference in the actual and ideal cases
    diff_exp = mod2pt(DNvals)
    diff_ideal = np.full((len(DNvals)), mod2pt.c0)
    # now create ramps from both allowing for non-zero ramp starting points
    DN_exp = np.cumsum(diff_exp) + startDN
    DN_ideal = np.cumsum(diff_ideal) + startDN
    cor = DN_ideal / DN_exp

    # fit the corretion to a polynomical
    cor_mod_init = Polynomial1D(degree=3)
    fit = LevMarLSQFitter()
    cor_mod = fit(cor_mod_init, DN_exp, cor, maxiter=10000)
    # print(cor_mod)

    return (DN_exp, cor, cor_mod)
