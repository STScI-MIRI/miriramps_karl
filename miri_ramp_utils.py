import numpy as np

from expmodel import Exponential1D
from astropy.modeling.models import Polynomial1D, Shift
from astropy.modeling.fitting import LevMarLSQFitter


__all__ = ["get_ramp", "get_good_ramp", "fit_diffs", "calc_lincor"]


def get_ramp(hdu, pix_x, pix_y, int_num, min_dn=0.0, rampoffval=0.0):
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

    rampoffval : float
        value to offset ramp due to changing video offset

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
    ydata = (hdu.data[k1:k2, pix_y, pix_x]).astype(float)
    ydata += rampoffval

    return (gnum, ydata)


def get_good_ramp(gnum, ydata, min_dn=0.0, max_dn=62500.0, keepfirst=False, nmax=None):
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

    keepfirst : boolean
        include the first frame, by default it is removed

    nmax : int
        maximum number of groups to use, by default all are used

    Returns
    -------
    ggnum, gdata, aveDN, diffDN : tuple of ndarrays
        ggnum is group numbers of the good data
        gdata is the DN values at each good group
        aveDN is the average values between each two groups
        diffDN is the difference between each two groups (2pt diff or CDS)
    """
    # create clean versions of the data
    if keepfirst:
        startk = 0
    else:
        startk = 1
    if nmax is None:
        nmax = -1

    ydatat = ydata[startk:nmax]
    gnumt = gnum[startk:nmax]
    (gindxs,) = np.where((ydatat > min_dn) & (ydatat < max_dn))
    gdata = ydatat[gindxs]
    ggnum = gnumt[gindxs]

    diffDN = np.diff(gdata)
    aveDN = 0.5 * (gdata[:-1] + gdata[1:])

    return (ggnum, gdata, aveDN, diffDN)


def fit_diffs(x, y, ndegree=2, noexp=False):
    """
    Fit the average DN versus 2pt differences with a model that accounts
    for the non-linearities and RSCD exponential decay at beginning of ramp

    Parameters
    ----------
    x : ndarray
        average DN values for each 2pt difference

    y : ndarray
        2pt differences

    ndegree : int
        degree of polynomial for fit

    noexp : boolean
        set to true to only do a polynomial (no exponential)

    Returns
    -------
    mod : astropy model
        fitted model
    """
    # print(min(x))

    if noexp:
        mod_init = Polynomial1D(degree=ndegree)
    else:
        mod_init = (
            Shift(offset=5000.0, bounds={"offset": [-100000.0, 20000.0]})
            | Exponential1D(
                x_0=-1000.0,
                amplitude=2500.0,
                bounds={"amplitude": [100.0, 100000.0], "x_0": [-10000.0, -0.1]},
            )
        ) + Polynomial1D(degree=ndegree)

    fit = LevMarLSQFitter()

    mod = fit(mod_init, x, y, maxiter=10000)

    # print(mod)

    # exit()

    return mod


def calc_lincor(mod2pt, DNvals, startDN, ndegree=3):
    """
    Calculate the linearity correction based on the fit to the 2pt differences
    and a single ramp

    Parameters
    ----------
    mod2pt : astropy model
        model giving the 2pt difference versus DN

    DNvals : ndarray
        measured DN values for a single ramp

    ndegree : int
        degree of polynomial for fit

    Returns
    -------
    DN_exp, cor, cor_mod : tuple
        DN_exp is the expected DN values given the 2pt diff mdoel and startDN value
        cor is the correction for the DN_exp values
        cor_mod is an astropy.modeling.model giving the polynomial fit to DN_exp and cor
    """
    # difference in the actual and ideal cases
    # DNvalst = DNvals[5:]
    DNvalst = DNvals
    diff_exp = mod2pt(DNvalst)
    diff_ideal = np.full((len(DNvalst)), mod2pt.c0)
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


def lincor_data(lincormod, gdata, aveDN, diffDN):
    """
    Applies the ramp based linearity correction to the data

    Parameters
    ----------
    lincormod : astropy model
        model given the behavior of the 2pt diffs versus average DN

    gdata : array
        the DN values at each good group

    aveDN : array
        the average values between each two groups

    diffDN :  array
        the difference between each two groups (2pt diff or CDS)

    Returns
    -------
    gdata_cor : array
        linearity correct ramp
    """

    ycor = lincormod(aveDN) / lincormod.c0.value
    diffDN_cor = diffDN / ycor
    gdata_cor = np.concatenate(([gdata[0]], gdata[0] + np.cumsum(diffDN_cor)))

    return gdata_cor
