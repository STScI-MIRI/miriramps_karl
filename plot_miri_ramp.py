import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

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
    filename = "Data/JPL8/M2_RSCD_Supplement/MIRI_5604_111_S_20180323-235653_SCE1.fits"
    hdu = fits.open(filename)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 8))
    # fig, ax = plt.subplots(ncols=2, figsize=(11, 8))

    pix_y, pix_x = args.pixel
    ngrps = hdu[0].header["NGROUPS"]
    nints = hdu[0].header["NINT"]

    # plot all integrations folded
    gnum = range(ngrps)
    for k in range(nints):
        k1 = k * ngrps
        k2 = k1 + ngrps
        ydata = (hdu[0].data[k1:k2, pix_x, pix_y]).astype(float)
        ax[0].plot(gnum, ydata, label=f"Int #{k}")

        # plot the 2pt diffs
        ax[1].plot(gnum[:-1], np.diff(ydata), label=f"Int #{k}")

        # plot the 2pt diffs versus average DN
        aveDN = 0.5 * (ydata[:-1] + ydata[1:])
        ax[2].plot(aveDN, np.diff(ydata), label=f"Int #{k}")

    ax[0].set_xlabel("group #")
    ax[0].set_ylabel("DN")

    ax[1].set_xlabel("group #")
    ax[1].set_ylabel("DN/group")

    ax[2].set_xlabel("DN")
    ax[2].set_ylabel("DN/group")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.suptitle(f"{filename}; Pixel ({pix_x}, {pix_y})")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_basename = f"plot_miri_ramp_{pix_x}_{pix_y}"
    if args.png:
        fig.savefig(out_basename)
    elif args.pdf:
        fig.savefig(out_basename)
    else:
        plt.show()
