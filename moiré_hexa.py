from hexalattice.hexalattice import *

for i in range(0, 301):
    hex_grid1, h_ax = create_hex_grid(nx=150,
                                      ny=150,
                                      rotate_deg=0,
                                      min_diam=0.288,
                                      crop_circ=18,
                                      do_plot=True)

    create_hex_grid(nx=75,
                    ny=75,
                    min_diam=0.604,
                    rotate_deg=-i / 10,
                    crop_circ=19,
                    do_plot=True,
                    h_ax=h_ax)

    plt.title("{}° rotation".format(-i / 10))
    plt.savefig("{}.pdf".format(-i / 10))
    plt.close()

