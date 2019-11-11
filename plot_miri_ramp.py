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
    parser.add_argument(
        "--primeonly", help="plot the primary exposure", action="store_true"
    )
    parser.add_argument("--png", help="save figure as an png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # filename = "Data/JPL8/M2_RSCD_Supplement/MIRI_5604_16_S_20180323-170413_SCE1.fits"
    # filename = "Data/MIRI_5604_111_S_20180323-235653_SCE1.fits"
    filename = "Data/MIRI_5692_17_S_20191017-184107_SCE1.fits"
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

    pix_x, pix_y = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = hdu[0].header["NINT"]

    # for fitting
    x = []
    y = []

    # for plotting
    pcol = ["b", "g", "r", "c"]

    # plot all integrations folded
    mm_delta = 0.0
    max_ramp_k = -1
    for k in range(nints):
        gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
        ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

        ax[0].plot(gnum, ydata, label=f"Int #{k+1}", color=pcol[k])

        # plot the 2pt diffs
        ax[1].plot(ggnum[:-1], diffDN, label=f"Int #{k+1}", color=pcol[k])

        # plot the 2pt diffs versus average DN
        ax[2].plot(aveDN, diffDN, label=f"Int #{k+1}", color=pcol[k])

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
    polymod = mod[2]

    # plot the model
    mod_x = np.linspace(0.0, 65000.0, num=100)

    # for k in range(nints):
    #    gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
    #    ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

    #    (DN_exp, cor, cor_mod) = calc_lincor(mod[1], gdata, args.startDN)
    #    ax[3].plot(DN_exp, cor, "--", label=f"Int #{k+1}")

    startDNvals = np.arange(0.0, 20000.0, 200.0)
    chival = np.zeros((nints, len(startDNvals)))
    ints = range(nints)
    for i, startDN in enumerate(startDNvals):
        (DN_exp, cor, cor_mod) = calc_lincor(polymod, max_ramp_aveDN, startDN)
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
            chival[k, i] = np.sum((diffDN[aveDN > 15000.0] - polymod.c0) ** 2)

    minindx = np.zeros((nints), dtype=int)
    for k in ints[1:]:
        ax[6].plot(startDNvals, chival[k, :], label=f"Int #{k+1}", color=pcol[k])
        minindx[k] = np.argmin(chival[k, :])
    startDN = startDNvals[minindx[max_ramp_k]]

    # get the correction
    (DN_exp, cor, cor_mod) = calc_lincor(polymod, max_ramp_aveDN, startDN)

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
        ax[0].plot(ggnum, gdata_cor, "--", label=f"Cor Int #{k+1}", color=pcol[k])

        # plot the corrected ramp divided by a linear fit
        nrej = 5
        line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
        intslopes[k] = line_mod.slope.value
        ax[4].plot(
            ggnum, gdata_cor / line_mod(ggnum), "--", label=f"Int #{k+1}", color=pcol[k]
        )

        # plot the 2pt diffs versus average DN
        diffDN = np.diff(gdata_cor)
        aveDN = 0.5 * (gdata[:-1] + gdata[1:])
        ax[5].plot(aveDN, diffDN, label=f"Int #{k+1}", color=pcol[k])

        # diffDN_orig = np.diff(gdata)
        # ax[2].plot(aveDN, diffDN_orig - mod[0](aveDN), '--')

        if k == 0:
            ax[5].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))

    aveslope = np.average(intslopes)
    ax[7].plot(
        np.array(ints) + 1,
        intslopes / aveslope,
        "ko",
        label=f"Exp 1: Ave = {aveslope:.2f}",
    )

    # ***

    filenames = [
        "Data/MIRI_5692_18_S_20191017-193412_SCE1.fits",
        "Data/MIRI_5692_19_S_20191017-202738_SCE1.fits",
        "Data/MIRI_5692_20_S_20191017-212044_SCE1.fits",
        "Data/MIRI_5692_21_S_20191017-221350_SCE1.fits",
        "Data/MIRI_5694_21_S_20191018-170349_SCE1.fits",
        "Data/MIRI_5694_22_S_20191018-172524_SCE1.fits",
        "Data/MIRI_5694_23_S_20191018-174658_SCE1.fits",
        "Data/MIRI_5694_24_S_20191018-180833_SCE1.fits",
        "Data/MIRI_5694_25_S_20191018-183008_SCE1.fits",
    ]
    if args.primeonly:
        filenames = []

    lin_off_val = 0.01
    for z, cfile in enumerate(filenames):
        hdu = fits.open(cfile)
        off_int = (z + 1) * nints

        # plot all integrations folded
        for k in range(nints):
            gnum, ydata = get_ramp(hdu[0], pix_x, pix_y, k)
            ggnum, gdata, aveDN, diffDN = get_good_ramp(gnum, ydata)

            ax[0].plot(gnum, ydata, color=pcol[k])

            # plot the 2pt diffs
            ax[1].plot(ggnum[:-1], diffDN, color=pcol[k])

            # plot the 2pt diffs versus average DN
            ax[2].plot(aveDN, diffDN, color=pcol[k])

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
            ax[0].plot(
                ggnum, gdata_cor, "--", color=pcol[k]
            )  # , label=f"Cor Int #{k+1+off_int}")

            # plot the corrected ramp divided by a linear fit
            nrej = 10
            line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
            intslopes[k] = line_mod.slope.value
            ax[4].plot(
                ggnum,
                gdata_cor / line_mod(ggnum) + z * lin_off_val,
                "--",
                color=pcol[k],
            )

            # plot the 2pt diffs versus average DN
            diffDN = np.diff(gdata_cor)
            aveDN = 0.5 * (gdata[:-1] + gdata[1:])
            ax[5].plot(aveDN, diffDN)

        aveslope = np.average(intslopes)
        ax[7].plot(
            np.array(ints) + 1 + off_int,
            intslopes / aveslope,
            "o",
            label=f"Exp {z+2}: Ave = {aveslope:.2f}",
        )

    # ***

    ax[2].plot(mod_x, mod(mod_x), "k--", label="Exp1D+Poly1D")
    ax[2].plot(mod_x, polymod(mod_x), "k-.", label="Poly1D only")

    ax[5].plot(ax[5].get_xlim(), [polymod.c0, polymod.c0], "k--", label="c_0")

    ax[7].set_xlabel("integration #")
    ax[7].set_ylabel("slope / ave")
    ax[7].set_ylim(0.99, 1.05)

    # finish the plots
    ax[0].set_xlabel("group #")
    ax[0].set_ylabel("DN")

    ax[4].set_xlabel("group #")
    ax[4].set_ylabel("DN_cor/line_fit")
    ax[4].set_ylim(0.99, 1.01 + lin_off_val * len(filenames))

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

    ax[k].legend(ncol=2)

    fig.suptitle(f"{filename}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_ramp_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename)
    elif args.pdf:
        fig.savefig(out_basename)
    else:
        plt.show()
