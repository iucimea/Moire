from hexalattice.hexalattice import *

#for i in range(0, 301):
from hexalattice.hexalattice import *
import matplotlib.pyplot as plt

# Create the first hexagonal grid
hex_grid1, h_ax = create_hex_grid(nx=40,
                                  ny=40,
                                  rotate_deg=0,
                                  min_diam=0.6,
                                  crop_circ=1000,
                                  edge_color='b',
                                  do_plot=True)

# Create the second hexagonal grid
create_hex_grid(nx=40,
                ny=40,
                min_diam=0.6,
                rotate_deg=5,   # -i / 10,
                crop_circ=1000,
                h_ax=h_ax,
                edge_color='r',
                do_plot=True)


# Save and show the plot
plt.savefig("10.pdf")
plt.show()
