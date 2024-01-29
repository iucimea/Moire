from hexalattice.hexalattice import *

#for i in range(0, 301):
hex_grid1, h_ax = create_hex_grid(nx=100,
                                  ny=100,
                                  rotate_deg=0,
                                  min_diam=0.6,
                                  crop_circ=1000,
                                  edge_color='k',
                                  do_plot=True)

create_hex_grid(nx=150,
                ny=150,
                min_diam=0.5,
                rotate_deg=20,   # -i / 10,
                crop_circ=1000,
                h_ax=h_ax,
                edge_color='k',
                do_plot=True)

#plt.title("{}° rotation".format(-i / 10))
#plt.savefig("{}.pdf".format(-i / 10))
#plt.close()

plt.savefig("jems_10.pdf")
plt.show()
