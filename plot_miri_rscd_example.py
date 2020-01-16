import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter

from expmodel import Exponential1D

from miri_ramp_utils import get_ramp, get_good_ramp, fit_diffs, lincor_data


if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixel",
        help="x y pixel values",
        metavar=("x", "y"),
        type=int,
        nargs=2,
        default=[512, 512],
    )
    parser.add_argument(
        "--nrej", help="number of groups to ignore in linear fit", type=int, default=0
    )
    parser.add_argument("--png", help="save figure as an png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    noexp = False
    # all_filenames = ["Data/MIRI_5708_137_S_20191027-201705_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5708_154_S_20191027-225136_SCE1.fits"]
    # noexp = True
    # all_filenames = ["Data/MIRI_5709_290_S_20191030-055245_SCE1.fits"]

    # all_filenames = ["Data/MIRI_5709_292_S_20191030-060729_SCE1.fits"]
    all_filenames = ["Data/MIRI_5709_294_S_20191030-062243_SCE1.fits"]
    noexp = True

    # open the fits file
    hdu = fits.open(all_filenames[0], memmap=False)

    fig, sax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    ax = [sax[0, 0], sax[1, 0], sax[0, 1], sax[1, 1]]

    # plotting setup for easier to read plots
    fontsize = 12
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=2)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    pix_x, pix_y = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = min([10, hdu[0].header["NINT"]])
    nrej = args.nrej

    # for fitting
    x = []
    y = []

    # for plotting
    pcol = plt.cm.jet(np.linspace(0, 1, nints))

    # plot all integrations folded
    for k in range(nints):
        gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        ax[0].plot(gnum, ydata, label=f"Int #{k+1}", color=pcol[k])

        # plot the 2pt diffs versus average DN
        ax[1].plot(aveDN, diffDN, label=f"Int #{k+1}", color=pcol[k])

        if k == 0:
            ax[1].set_ylim(0.9 * min(diffDN), 1.4 * max(diffDN))

        # accumulate data for later plotting
        x.append(aveDN)
        y.append(diffDN)

    # fit the aveDN versus diffDN combined data from all integrations
    # e.g., derive the non-linearity correction
    x = np.concatenate(x)
    y = np.concatenate(y)
    mod = fit_diffs(x, y, noexp=noexp)
    if noexp:
        polymod = mod
    else:
        polymod = mod[2]
    lincormod = polymod

    ints = range(nints)

    # setup ramp fit
    line_init = Linear1D()
    fit_line = LinearLSQFitter()
    mult_comp = False

    intslopes = np.zeros((nints))
    linfit_metric = np.zeros((nints))
    intexpamp = np.zeros((nints))
    for k in range(nints):
        gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        # correct the ramps and plot
        gdata_cor = lincor_data(lincormod, gdata, aveDN, diffDN)

        # plot the linearied 2pt diffs versus average DN
        diffDN = np.diff(gdata_cor)
        aveDN = 0.5 * (gdata[:-1] + gdata[1:])
        ax[3].plot(aveDN, diffDN, color=pcol[k])

        # compute the corrected ramp divided by a linear fit
        line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
        if mult_comp:
            intslopes[k] = line_mod[1].slope.value
            intexpamp[k] = line_mod[0].amplitude.value
        else:
            intslopes[k] = line_mod.slope.value
        linfit_ratio = gdata_cor / line_mod(ggnum)

        ax[3].plot([0., 65000.], [intslopes[k], intslopes[k]], "k--",
                   color=pcol[k], alpha=0.5)

        # compute metric on deviations from the linear fit
        linfit_metric[k] = np.sum(np.power(linfit_ratio[nrej:] - 1.0, 2.0)) / len(
            linfit_ratio[nrej:]
        )

    # plot the measured slopes
    aveslope = np.average(intslopes)
    for k in range(nints):
        ax[2].plot(
            [ints[k] + 1],
            [intslopes[k] / aveslope],
            "ko",
            label=f"Int #{ints[k]+1} slope = {intslopes[k]:.2f} DN/group",
            color=pcol[k],
        )
    ax[2].set_ylim(0.995, 1.02)

    # plot the model
    mod_x = np.linspace(0.0, 65000.0, num=100)
    ax[1].plot(mod_x, mod(mod_x), "k--", label="Exp1D+Poly1D")
    ax[1].plot(mod_x, polymod(mod_x), "k-.", label="Poly1D only")

    # finish the plots
    ax[0].set_xlabel("group #", fontdict=font)
    ax[0].set_ylabel("DN", fontdict=font)

    ax[1].set_xlabel("DN", fontdict=font)
    ax[1].set_ylabel("DN/group", fontdict=font)

    ax[2].set_xlabel("integration #")
    ax[2].set_ylabel("slope / ave")

    ax[3].set_xlabel("DN")
    ax[3].set_ylabel("DN(linearized)/group")

    ax[0].legend(fontsize=10)
    ax[1].legend(fontsize=9)
    ax[2].legend(fontsize=9)

    fig.suptitle(f"{all_filenames[0]}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_rscd_example_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename + '.png')
    elif args.pdf:
        fig.savefig(out_basename + '.pdf')
    else:
        plt.show()
