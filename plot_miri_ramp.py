import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter

from miri_ramp_utils import get_ramp, get_good_ramp, fit_diffs, calc_lincor


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
    # parser.add_argument(
    #     "--startDN", help="DN for start of ramp in correction", type=float, default=0.0
    # )
    parser.add_argument("--png", help="save figure as an png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # filename = "Data/JPL8/M2_RSCD_Supplement/MIRI_5604_16_S_20180323-170413_SCE1.fits"
    # filename = "Data/MIRI_5604_111_S_20180323-235653_SCE1.fits"
    filename = "Data/MIRI_5692_18_S_20191017-193412_SCE1.fits"
    hdu = fits.open(filename)

    fig, sax = plt.subplots(ncols=4, nrows=2, figsize=(18, 9))
    ax = [
        sax[0, 0],
        sax[1, 0],
        sax[1, 1],
        sax[0, 1],
        sax[0, 2],
        sax[1, 2],
        sax[1, 3],
        sax[0, 3],
    ]

    pix_y, pix_x = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = hdu[0].header["NINT"]

    # for fitting
    x = []
    y = []

    # plot all integrations folded
    mm_delta = 0.0
    max_ramp_k = -1
    for k in range(nints):
        gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        ax[0].plot(gnum, ydata, label=f"Int #{k+1}")

        # plot the 2pt diffs
        ax[1].plot(ggnum[:-1], diffDN, label=f"Int #{k+1}")

        # plot the 2pt diffs versus average DN
        ax[2].plot(aveDN, diffDN, label=f"Int #{k+1}")

        if k == 0:
            ax[1].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))
            ax[2].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))

        # accumulate data for later plotting
        x.append(aveDN)
        y.append(diffDN)

        # find the ramp that spans the largest range of DN
        # and save some info -> needed for creating the correction
        # if (gdata.max() - gdata.min()) > mm_delta:
        #    mm_delta = gdata.max() - gdata.min()
        if k == 3:
            max_ramp_k = k
            max_ramp_gdata = gdata
            max_ramp_aveDN = aveDN

    # fit the aveDN versus diffDN combined data from all integrations
    x = np.concatenate(x)
    y = np.concatenate(y)
    mod = fit_diffs(x, y)

    # plot the model
    mod_x = np.linspace(0.0, 65000.0, num=100)
    ax[2].plot(mod_x, mod(mod_x), "k--", label="Exp1D+Poly1D")
    ax[2].plot(mod_x, mod[1](mod_x), "k-.", label="Poly1D only")

    # for k in range(nints):
    #    gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
    #    ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

    #    (DN_exp, cor, cor_mod) = calc_lincor(mod[1], gdata, args.startDN)
    #    ax[3].plot(DN_exp, cor, "--", label=f"Int #{k+1}")

    startDNvals = np.arange(0.0, 20000.0, 200.0)
    chival = np.zeros((nints, len(startDNvals)))
    ints = range(nints)
    for i, startDN in enumerate(startDNvals):
        (DN_exp, cor, cor_mod) = calc_lincor(mod[1], max_ramp_aveDN, startDN)
        for k in ints:
            gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
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
        ax[6].plot(startDNvals, chival[k, :], label=f"Int #{k+1}")
        minindx[k] = np.argmin(chival[k, :])
    startDN = startDNvals[minindx[max_ramp_k]]

    # get the correction
    (DN_exp, cor, cor_mod) = calc_lincor(mod[1], max_ramp_aveDN, startDN)

    ax[3].plot(DN_exp, cor, "ko", label=f"Int #{max_ramp_k+1} StartDN={startDN:.1f}")
    ax[3].plot(mod_x, cor_mod(mod_x), "k--", label="Cor Poly1D")

    # apply the correction
    line_init = Linear1D()
    fit_line = LinearLSQFitter()
    intslopes = np.zeros((nints))
    for k in range(nints):
        gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        # correct the ramps and plot
        ycor = cor_mod(gdata)
        gdata_cor = gdata * ycor
        ax[0].plot(ggnum, gdata_cor, "--", label=f"Cor Int #{k+1}")

        # plot the corrected ramp divided by a linear fit
        nrej = 5
        line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
        intslopes[k] = line_mod.slope.value
        ax[4].plot(ggnum, gdata_cor / line_mod(ggnum), "--", label=f"Int #{k+1}")

        # plot the 2pt diffs versus average DN
        diffDN = np.diff(gdata_cor)
        aveDN = 0.5 * (gdata[:-1] + gdata[1:])
        ax[5].plot(aveDN, diffDN, label=f"Int #{k+1}")

        # diffDN_orig = np.diff(gdata)
        # ax[2].plot(aveDN, diffDN_orig - mod[0](aveDN), '--')

        if k == 0:
            ax[5].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))

    ax[5].plot(ax[5].get_xlim(), [mod[1].c0, mod[1].c0], "k--", label="c_0")

    aveslope = np.average(intslopes)
    ax[7].plot(np.array(ints) + 1, intslopes / aveslope, 'ko', label=f"Ave = {aveslope:.2f}")
    ax[7].set_xlabel("integration #")
    ax[7].set_ylabel("slope / ave")

    # finish the plots
    ax[0].set_xlabel("group #")
    ax[0].set_ylabel("DN")

    ax[4].set_xlabel("group #")
    ax[4].set_ylabel("DN_cor/line_fit")
    ax[4].set_ylim(0.99, 1.01)

    ax[1].set_xlabel("group #")
    ax[1].set_ylabel("DN/group")

    ax[5].set_xlabel("DN")
    ax[5].set_ylabel("DN_cor/group")

    ax[2].set_xlabel("DN")
    ax[2].set_ylabel("DN/group")

    ax[3].set_xlabel("DN")
    ax[3].set_ylabel("Mult Correction")

    ax[6].set_xlabel("startDN")
    ax[6].set_ylabel("chisqr")

    for k in range(len(ax)):
        ax[k].legend()

    fig.suptitle(f"{filename}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_ramp_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename)
    elif args.pdf:
        fig.savefig(out_basename)
    else:
        plt.show()
