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
    all_filenames = ["Data/MIRI_5708_137_S_20191027-201705_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5708_154_S_20191027-225136_SCE1.fits"]
    # noexp = True
    # all_filenames = ["Data/MIRI_5709_290_S_20191030-055245_SCE1.fits"]

    # all_filenames = ["Data/MIRI_5709_292_S_20191030-060729_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5709_294_S_20191030-062243_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5709_76_S_20191029-043558_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5709_92_S_20191029-063439_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5709_78_S_20191029-045032_SCE1.fits"]
    # all_filenames = ["Data/MIRI_5709_82_S_20191029-052000_SCE1.fits"]
    # noexp = True

    # open the fits file
    hdu = fits.open(all_filenames[0], memmap=False)

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    ax = [fig.add_subplot(gs[0, :]),
          fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1])]

    # plotting setup for easier to read plots
    fontsize = 16
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=2)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    for k in range(3):
        ax[k].tick_params(axis='both', which='major', labelsize=fontsize)

    pix_x, pix_y = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = min([5, hdu[0].header["NINT"]])
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

        ax[0].plot(gnum + k * ngrps, ydata, label=f"Int #{k+1}", color=pcol[k])

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
        ax[1].plot(ggnum[:-1] + 0.5, diffDN, color=pcol[k])
        # ax[1].plot(aveDN, diffDN, color=pcol[k])

        # compute the corrected ramp divided by a linear fit
        line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
        if mult_comp:
            intslopes[k] = line_mod[1].slope.value
            intexpamp[k] = line_mod[0].amplitude.value
        else:
            intslopes[k] = line_mod.slope.value
        linfit_ratio = gdata_cor / line_mod(ggnum)

        # compute metric on deviations from the linear fit
        linfit_metric[k] = np.sum(np.power(linfit_ratio[nrej:] - 1.0, 2.0)) / len(
            linfit_ratio[nrej:]
        )

    # plot the measured slopes
    aveslope = np.average(intslopes)
    for k in range(nints):
        ax[2].plot(
            [ints[k] + 1],
            [intslopes[k] / intslopes[0]],
            "ko",
            label=f"Int #{ints[k]+1} slope = {intslopes[k]:.2f} DN/group",
            color=pcol[k],
            markersize=10,
        )
    # ax[2].set_ylim(0.995, 1.02)

    # finish the plots
    ax[0].set_xlabel("group #", fontdict=font)
    ax[0].set_ylabel("DN", fontdict=font)

    ax[2].set_xlabel("integration #", fontdict=font)
    ax[2].set_ylabel("slope / slope(int1)", fontdict=font)

    ax[1].set_xlabel("group #", fontdict=font)
    ax[1].set_ylabel("DN/group", fontdict=font)

    ax[0].errorbar([20.], [11000.], yerr=[6000.], ecolor='k', capthick=2, capsize=10)
    ax[0].annotate('RSCD \nramp \noffset', xy=(25, 10000.), xytext=(40., 20000.),
                   arrowprops=dict(facecolor='black', shrink=0.025), fontsize=20)
    # ax[1].text(20000., 1450., 'RSCD fast decay', fontsize=20)
    # ax[1].errorbar([10000.], [1500.], xerr=[2000.], ecolor='k', capthick=2, capsize=10)
    ax[1].annotate('RSCD fast decay', xy=(3., 1300.), xytext=(10., 1350),
                   arrowprops=dict(facecolor='black', shrink=0.025), fontsize=20)
    ax[2].errorbar([1.5], [0.9946], yerr=[0.0052], ecolor='k', capthick=2, capsize=10)
    ax[2].annotate('RSCD slope difference', xy=(1.6, 0.9946), xytext=(2.5, 0.999),
                   arrowprops=dict(facecolor='black', shrink=0.025), fontsize=20)

    # ax[0].legend(fontsize=10)
    # ax[1].legend(fontsize=9)
    # ax[2].legend(fontsize=9)

    fig.suptitle(f"{all_filenames[0]}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_rscd_example_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename + '.png')
    elif args.pdf:
        fig.savefig(out_basename + '.pdf')
    else:
        plt.show()
