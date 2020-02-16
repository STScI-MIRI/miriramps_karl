import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LinearLSQFitter

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
        "--nrej", help="number of groups to ignore in linear fit", type=int, default=4
    )
    parser.add_argument(
        "--nmax", help="max number groups in linear fit", type=int, default=None
    )
    parser.add_argument(
        "--keepfirst", help="Include the first frame", action="store_true"
    )
    parser.add_argument(
        "--subarray",
        choices=["FULL", "4QPM", "SUB64", "SLOW"],
        default="FULL",
        help="subarray indicates data used",
    )
    parser.add_argument("--png", help="save figure as an png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    noexp = False
    plot_skip = False
    max_fr_group = 20
    if args.subarray == "SUB64":
        all_filenames = [
            "Data/MIRI_5709_112_S_20191029-090719_SCE1.fits",
            "Data/MIRI_5709_114_S_20191029-092123_SCE1.fits",
            "Data/MIRI_5709_120_S_20191029-100335_SCE1.fits",
            "Data/MIRI_5709_122_S_20191029-101739_SCE1.fits",
            #  "Data/MIRI_5709_128_S_20191029-110020_SCE1.fits",
        ]
        frameresets = [0, 1, 4, 8]  # , 30]
        # ptitle = "SCA106 5709(112, 114, 120, 122, 128) SUB64 subarray"
        ptitle = "SCA106 5709(112, 114, 120, 122) SUB64 subarray"
    elif args.subarray == "4QPM":
        all_filenames = [
            "Data/MIRI_5709_76_S_20191029-043558_SCE1.fits",
            "Data/MIRI_5709_78_S_20191029-045032_SCE1.fits",
            "Data/MIRI_5709_84_S_20191029-053444_SCE1.fits",
            "Data/MIRI_5709_86_S_20191029-054928_SCE1.fits",
            # "Data/MIRI_5709_92_S_20191029-063439_SCE1.fits",
        ]
        frameresets = [0, 1, 4, 8]  # , 30]
        # ptitle = "SCA106 5709(76, 78, 84, 86, 92) 4QPM subarray"
        ptitle = "SCA106 5709(76, 78, 84, 86) 4QPM subarray"
    elif args.subarray == "SLOW":
        all_filenames = [
            "Data/MIRI_5708_42_S_20191026-173002_SCE1.fits",
            "Data/MIRI_5708_41_S_20191026-170526_SCE1.fits",
            "Data/MIRI_5708_40_S_20191026-165240_SCE1.fits",
        ]
        frameresets = [0, 0, 0]
        ptitle = "SCA106 5708(42) FULL SLOW"
        plot_skip = False
        max_fr_group = 10
    else:
        all_filenames = [
            "Data/MIRI_5709_34_S_20191029-011921_SCE1.fits",
            "Data/MIRI_5709_32_S_20191029-010437_SCE1.fits",
            "Data/MIRI_5709_30_S_20191029-004923_SCE1.fits",
            "Data/MIRI_5709_28_S_20191029-003349_SCE1.fits",
            "Data/MIRI_5709_18_S_20191028-231009_SCE1.fits",
        ]
        frameresets = [0, 1, 2, 3, 10]
        ptitle = "SCA106 5709(34, 32, 30 28, 18) FULL"

    # noexp = True
    n_files = len(all_filenames)

    psize = 3
    fig, sax = plt.subplots(
        ncols=n_files, nrows=3, figsize=(psize * n_files, 3 * psize), sharey="row",
    )

    # plotting setup for easier to read plots
    fontsize = 14
    font = {"size": fontsize}
    mpl.rc("font", **font)
    mpl.rc("lines", linewidth=2)
    mpl.rc("axes", linewidth=2)
    mpl.rc("xtick.major", width=2)
    mpl.rc("xtick.minor", width=2)
    mpl.rc("ytick.major", width=2)
    mpl.rc("ytick.minor", width=2)

    for j in range(n_files):
        for k in range(3):
            sax[k, j].tick_params(axis='both', which='major', labelsize=fontsize)

    aveslope = 0.0

    fr0_ggnum = []
    fr0_gdata_cor = []

    for z, cfile in enumerate(all_filenames):

        # open the fits file
        hdu = fits.open(cfile, memmap=False)

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

            sax[0, z].plot(gnum, ydata, label=f"Int #{k+1}", color=pcol[k])

            # plot the 2pt diffs versus average DN
            # ax[1].plot(aveDN, diffDN, label=f"Int #{k+1}", color=pcol[k])

            # if k == 0:
            #    ax[1].set_ylim(0.9 * min(diffDN), 1.4 * max(diffDN))

            # accumulate data for fitting
            x.append(aveDN)
            y.append(diffDN)

        if z == 0:
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

        # more plots
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
            ggnum, gdata, aveDN, diffDN = get_good_ramp(
                gnum, ydata, keepfirst=args.keepfirst, nmax=args.nmax
            )

            # correct the ramps and plot
            gdata_cor = lincor_data(lincormod, gdata, aveDN, diffDN)

            # plot the linearied 2pt diffs versus average DN
            diffDN = np.diff(gdata_cor)
            aveDN = 0.5 * (gdata[:-1] + gdata[1:])
            avegnum = 0.5 * (ggnum[:-1] + ggnum[1:])
            sax[1, z].plot(avegnum, diffDN, color=pcol[k])

            # compute the corrected ramp divided by a linear fit
            line_mod = fit_line(line_init, ggnum[nrej:], gdata_cor[nrej:])
            intslopes[k] = line_mod.slope.value
            linfit_ratio = gdata_cor / line_mod(ggnum)

            # compute metric on deviations from the linear fit
            linfit_metric[k] = np.sum(np.power(linfit_ratio[nrej:] - 1.0, 2.0)) / len(
                linfit_ratio[nrej:]
            )

            # plot the slope of the framereset=0 data with the same number of frames
            # rejected as the current data framereset value
            if z == 0:
                fr0_ggnum.append(ggnum)
                fr0_gdata_cor.append(gdata_cor)
            else:
                fr0_nrej = nrej + frameresets[z]
                if fr0_nrej < len(fr0_ggnum[k]) - 2:
                    line_mod = fit_line(
                        line_init, fr0_ggnum[k][fr0_nrej:], fr0_gdata_cor[k][fr0_nrej:]
                    )
                    if plot_skip:
                        if k == 0:
                            intslope_nrej_fr = line_mod.slope.value
                        sax[2, z].plot(
                            [ints[k] + 1],
                            [line_mod.slope.value] / intslope_nrej_fr,
                            "ko",
                            fillstyle="none",
                            markersize=10,
                        )

        for k in range(nints):
            sax[2, z].plot(
                [ints[k] + 1],
                [intslopes[k] / intslopes[0]],
                "ko",
                label=f"Int #{ints[k]+1} slope = {intslopes[k]:.2f} DN/group",
                color="k",  # pcol[k],
            )

        # plot the model
        mod_x = np.linspace(0.0, 65000.0, num=100)
        # sax[z, 1].plot(mod_x, mod(mod_x), "k--", label="Exp1D+Poly1D")
        # ax[1].plot(mod_x, polymod(mod_x), "k-.", label="Poly1D only")

        # finish the plots
        sax[0, z].set_title(f"FRAMERESETS={frameresets[z]}", fontdict=font)

        sax[0, z].set_xlabel("group #", fontdict=font)
        sax[1, z].set_xlabel("group #", fontdict=font)
        sax[2, z].set_xlabel("integration #", fontdict=font)

        sax[1, z].set_xlim(0, max_fr_group)

        # sax[2, z].set_ylim(0.98, 1.02)

        if z == 0:
            sax[0, z].set_ylabel("DN", fontdict=font)
            sax[1, z].set_ylabel("DN/group", fontdict=font)
            sax[2, z].set_ylabel("slope / slope(int=1)", fontdict=font)
            sax[0, z].legend(fontsize=8)

        # ax[3].set_xlabel("DN")
        # ax[3].set_ylabel("DN(linearized)/group")

        # ax[1].legend(fontsize=9)
        # ax[2].legend(fontsize=9)

    # custom legend for the bottom row of plots
    if plot_skip:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="FR=N data",
                fillstyle="none",
                markerfacecolor="k",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="FR=0 skip=N",
                fillstyle="none",
                markeredgecolor="k",
                markersize=10,
            ),
        ]
        sax[2, 0].legend(handles=legend_elements)

    fig.suptitle(f"{ptitle}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_rscd_frameresest_{args.subarray}_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename + ".png")
    elif args.pdf:
        fig.savefig(out_basename + ".pdf")
    else:
        plt.show()
