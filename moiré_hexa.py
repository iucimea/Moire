from hexalattice.hexalattice import *
import matplotlib.pyplot as plt

# Create the first hexagonal grid (yellow edges)
hex_grid1, h_ax = create_hex_grid(nx=20,
                                  ny=20,
                                  rotate_deg=0,
                                  min_diam=0.6,
                                  crop_circ=1000,
                                  edge_color='b',
                                  line_width=1.5,
                                  do_plot=True)

# Create the second hexagonal grid (blue edges)
create_hex_grid(nx=20,
                ny=20,
                min_diam=0.6,
                rotate_deg=5,
                crop_circ=1000,
                h_ax=h_ax,
                edge_color='r',
                line_width=1.5,  # blue
                do_plot=True)

# Save and show the plot
plt.savefig("5.pdf")
plt.show()
