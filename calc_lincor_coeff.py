import argparse

from tqdm import tqdm
import numpy as np
from astropy.io import fits

from miri_ramp_utils import get_ramp, get_good_ramp, fit_diffs, calc_lincor


def compute_lincor_model(hdu, pix_x, pix_y, ndegree=3):
    """
    Compute the lineary correction model

    Parameters
    ----------
    hdu : astropy.fits.hdu
        header data unit

    pix_x, pix_y: int
        x, y pixel coordiates

    Returns
    -------
    cormod : astropy.modeling.model
        model the linearity correction with x = measured DN
    """
    # for fitting
    x = []
    y = []

    # get all the diffs
    for k in range(nints):
        gnum, ydata = get_ramp(hdu, pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        # accumulate data
        x.append(aveDN)
        y.append(diffDN)

        # find the ramp that spans the largest range of DN
        # and save some info -> needed for creating the correction
        # if (gdata.max() - gdata.min()) > mm_delta:
        #    mm_delta = gdata.max() - gdata.min()
        if k == 3:
            max_ramp_k = k
            # max_ramp_gdata = gdata
            max_ramp_aveDN = aveDN

    # fit the aveDN versus diffDN combined data from all integrations
    x = np.concatenate(x)
    y = np.concatenate(y)
    mod = fit_diffs(x, y, ndegree=ndegree - 1)

    # determine the startDN for all ramps (all not actually needed)
    startDNvals = np.arange(0.0, 20000.0, 200.0)
    chival = np.zeros((nints, len(startDNvals)))
    ints = range(nints)
    for i, startDN in enumerate(startDNvals):
        (DN_exp, cor, cor_mod) = calc_lincor(
            mod[1], max_ramp_aveDN, startDN, ndegree=ndegree
        )
        for k in ints:
            gnum, ydata = get_ramp(hdu, pix_x, pix_y, k)
            ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)
            # correct the ramps
            ycor = cor_mod(gdata)
            gdata_cor = gdata * ycor
            # calculate the chisqr for each integration set of differences
            # from the expected flat line
            diffDN = np.diff(gdata_cor)
            aveDN = 0.5 * (gdata[:-1] + gdata[1:])
            cindxs, = np.where(aveDN > 10000.0)
            chival[k, i] = np.sum((diffDN[aveDN > 15000.0] - mod[1].c0) ** 2)

    minindx = np.zeros((nints), dtype=int)
    for k in ints[1:]:
        minindx[k] = np.argmin(chival[k, :])

    # only use the startDN from one ramp
    startDN = startDNvals[minindx[max_ramp_k]]

    # get the correction
    (DN_exp, cor, cor_mod) = calc_lincor(mod[1], max_ramp_aveDN, startDN)

    return cor_mod


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    hdu = fits.open(args.filename)

    ngrps = hdu[0].header["NGROUPS"]
    nints = hdu[0].header["NINT"]

    # coeff images for output
    n_x = int(hdu[0].header["NAXIS1"])
    n_y = int(hdu[0].header["NAXIS2"])
    ndegree = 3
    coeffs = np.zeros((ndegree + 1, n_y, n_x))

    # loop over all good pixels
    pix_x = [450, 550]
    pix_y = [450, 550]
    for i in tqdm(range(pix_x[0], pix_x[1], 1)):
        for j in tqdm(range(pix_y[0], pix_y[1], 1), leave=False):
            cormod = compute_lincor_model(hdu[0], i, j, ndegree=ndegree)
            coeffs[:, j, i] = cormod.parameters

    fits.writeto(args.filename.replace(".fits", "_lincor.fits"), coeffs, overwrite=True)
