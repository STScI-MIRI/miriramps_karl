import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.models import Polynomial1D
from expmodel import Exponential1D
from astropy.modeling.fitting import LevMarLSQFitter  # , LinearLSQFitter

if __name__ == "__main__":

    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pixel",
        help="x y pixel values",
        metavar=("x", "y"),
        type=int,
        nargs=2,
        default=[150, 150],
    )
    parser.add_argument("--png", help="save figure as an png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # filename = "Data/JPL8/M2_RSCD_Supplement/MIRI_5604_16_S_20180323-170413_SCE1.fits"
    filename = "Data/MIRI_5604_111_S_20180323-235653_SCE1.fits"
    hdu = fits.open(filename)

    fig, sax = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
    ax = [sax[0, 0], sax[1, 0], sax[1, 1], sax[0, 1], sax[0, 2], sax[1, 2]]

    pix_y, pix_x = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = hdu[0].header["NINT"]

    # for fitting
    x = []
    y = []

    # for the correction
    rmin = np.zeros((ngrps))
    rmax = np.zeros((ngrps))

    # plot all integrations folded
    gnum = np.array(range(ngrps))
    for k in range(nints):
        k1 = k * ngrps
        k2 = k1 + ngrps
        ydata = (hdu[0].data[k1:k2, pix_x, pix_y]).astype(float)
        ax[0].plot(gnum, ydata, label=f"Int #{k+1}")

        # create clean versions of the data
        gindxs, = np.where((ydata > 0.0) & (ydata < 55000.))
        gdata = ydata[gindxs]
        ggnum = gnum[gindxs]

        # determine the DN min/max of ramp
        rmin[k] = gdata.min()
        rmax[k] = gdata.max()

        # plot the 2pt diffs
        diffDN = np.diff(gdata)
        ax[1].plot(ggnum[:-1], diffDN, label=f"Int #{k+1}")

        # plot the 2pt diffs versus average DN
        aveDN = 0.5 * (gdata[:-1] + gdata[1:])
        ax[2].plot(aveDN, diffDN, label=f"Int #{k+1}")

        if k == 0:
            ax[1].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))
            ax[2].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))

        # accumulate data for later plotting
        x.append(aveDN)
        y.append(diffDN)

    # fit the aveDN versus diffDN combined data from all integrations
    x = np.concatenate(x)
    y = np.concatenate(y)

    mod_init = Exponential1D(
        x_0=-2000.0,
        amplitude=2500.0,
        bounds={"amplitude": [100.0, 5000.0], "x_0": [-10000., 0.0]},
    ) + Polynomial1D(degree=2)

    fit = LevMarLSQFitter()

    findxs, = np.where((x < 55000.0) & (x > 0.0))
    mod = fit(mod_init, x[findxs], y[findxs], maxiter=10000)
    mod_x = np.linspace(0.0, 65000.0, num=100)
    mod_y = mod(mod_x)
    ax[2].plot(mod_x, mod_y, "k--", label="Exp1D+Poly1D")
    ax[2].plot(mod_x, mod[1](mod_x), "k-.", label="Poly1D only")

    # print(mod.param_names)
    # print(mod.parameters)

    # create the correction
    poly_mod = mod[1]
    # find the ramp that spans the largest range of DN
    ramp_minmax = rmax - rmin
    max_ramp_k, = np.where(ramp_minmax == np.amax(ramp_minmax))

    # observed data
    k = max_ramp_k[0]
    k1 = k * ngrps
    k2 = k1 + ngrps
    ydata = (hdu[0].data[k1:k2, pix_x, pix_y]).astype(float)

    # create clean versions of the data
    gindxs, = np.where((ydata > 0.0) & (ydata < 58000.))
    gdata = ydata[gindxs]
    aveDN = 0.5 * (gdata[:-1] + gdata[1:])

    # compute the expected 2pt diffs from the fit w/o the exponential
    #   need to use the real spacing between groups as the fit is done
    #   to the DN/group -> gotta get the spaceing right
    diff_exp = poly_mod(aveDN)
    diff_ideal = np.full((len(aveDN)), poly_mod.c0)
    # now create ramps from both
    DN_exp = np.cumsum(diff_exp)
    DN_ideal = np.cumsum(diff_ideal)
    cor = DN_ideal / DN_exp
    ax[3].plot(DN_exp, cor, label=f"Int #{k+1}")

    # fit the corretion to a polynomical
    cor_mod_init = Polynomial1D(degree=3)
    # fit_cor = LinearLSQFitter()
    cor_mod = fit(cor_mod_init, DN_exp, cor, maxiter=10000)
    # print(cor_mod)
    ax[3].plot(DN_exp, cor_mod(DN_exp), 'k--', label='Cor Poly1D')

    # apply the correction
    for k in range(nints):
        k1 = k * ngrps
        k2 = k1 + ngrps
        ydata = (hdu[0].data[k1:k2, pix_x, pix_y]).astype(float)

        # create clean versions of the data
        gindxs, = np.where((ydata > 0.0) & (ydata < 55000.))
        gdata = ydata[gindxs]
        ggnum = gnum[gindxs]

        # correct the ramps and plot
        ycor = np.interp(gdata, DN_exp, cor)
        gdata_cor = gdata * ycor
        ax[4].plot(ggnum, gdata_cor, label=f"Int #{k+1}")

        # plot the 2pt diffs versus average DN
        diffDN = np.diff(gdata_cor)
        aveDN = 0.5 * (gdata[:-1] + gdata[1:])
        ax[5].plot(aveDN, diffDN, label=f"Int #{k+1}")

        if k == 0:
            ax[5].set_ylim(0.9 * min(diffDN), 1.1 * max(diffDN))

    ax[5].plot(ax[5].get_xlim(), [poly_mod.c0, poly_mod.c0], "k--", label='c_0')

    # finish the plots
    ax[0].set_xlabel("group #")
    ax[0].set_ylabel("DN")
    ax[4].set_xlabel("group #")
    ax[4].set_ylabel("DN_cor")

    ax[1].set_xlabel("group #")
    ax[1].set_ylabel("DN/group")
    ax[5].set_xlabel("DN")
    ax[5].set_ylabel("DN_cor/group")

    ax[2].set_xlabel("DN")
    ax[2].set_ylabel("DN/group")

    ax[3].set_xlabel("DN")
    ax[3].set_ylabel("Mult Correction")

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
