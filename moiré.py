import math
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
import scipy.stats as st
from sklearn.datasets import make_blobs
from scipy.ndimage import gaussian_filter


# n_components = 3
# atoms, truth = make_blobs(n_samples=300, centers=n_components,
#                       cluster_std = [2, 1.5, 1],
#                       random_state=42)
#
# plt.scatter(atoms[:, 0], atoms[:, 1], s=50, c=truth)
# plt.title(f"Example of a mixture of {n_components} distributions")
# plt.xlabel("x")
# plt.ylabel("y")


for i in range(0, 1):
    hex_grid1, h_ax = create_hex_grid(nx=150,
                                      ny=150,
                                      rotate_deg=0,
                                      min_diam=0.288,
                                      crop_circ=20,
                                      # edge_color='b',
                                      do_plot=False)

    hex_grid2, dummy = create_hex_grid(nx=75,
                                   ny=75,
                                   min_diam=0.604,
                                   rotate_deg=0,  # i/10,
                                   crop_circ=20,
                                   # edge_color='r',
                                   do_plot=False,
                                   h_ax=h_ax)

atoms = np.concatenate((hex_grid1, hex_grid2), axis=0)

# Extract x and y
x = atoms[:, 0]
y = atoms[:, 1]

resolution = 50

# Define the borders
dx = max(x) - min(x)
dy = max(y) - min(y)

x_min = min(x) - 1
x_max = max(x) + 1
y_min = min(y) - 1
y_max = max(y) + 1

print(x_min, x_max, y_min, y_max)

# Create meshgrid
grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, grid_x.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
cfset = ax.contourf(grid_x, grid_y, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[x_min, x_max, y_min, y_max])
cset = ax.contour(grid_x, grid_y, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(grid_x, grid_y, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
ax.view_init(60, 35)

fig, ax = plt.subplots()
data = gaussian_filter(atoms, sigma=5)
plt.pcolormesh(data.T, cmap='inferno', shading='gouraud')
fig.canvas.draw()

def gauss(x1, x2, y1, y2):
    """
    Apply a Gaussian kernel estimation (2-sigma) to distance between points.

    Effectively, this applies a Gaussian kernel with a fixed radius to one
    of the points and evaluates it at the value of the euclidean distance
    between the two points (x1, y1) and (x2, y2).
    The Gaussian is transformed to roughly (!) yield 1.0 for distance 0 and
    have the 2-sigma located at radius distance.
    """
    return (
            (1.0 / (2.0 * math.pi))
            * math.exp(
        -1 * (3.0 * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / radius)) ** 2
            / 0.4)


def _kde(x, y):
    """
    Estimate the kernel density at a given position.

    Simply sums up all the Gaussian kernel values towards all points
    (pts_x, pts_y) from position (x, y).
    """
    return sum([
        gauss(x, px, y, py)
        # math.sqrt((x - px)**2 + (y - py)**2)
        for px, py in zip(pts_x, pts_y)
    ])


kde = np.vectorize(_kde)  # Let numpy care for applying our kde to a vector
z = kde(x, y)

xi, yi = np.where(z == np.amax(z))
max_x = grid_x[xi][0]
max_y = grid_y[yi][0]
print(f"{max_x:.4f}, {max_y:.4f}")

fig, ax = plt.subplots()
ax.axis('equal')
ax.pcolormesh(x, y, z, cmap='inferno', vmin=np.min(z), vmax=np.max(z))
fig.set_size_inches(4, 4)
fig.savefig('density.png', bbox_inches='tight')

fig, ax = plt.subplots()
ax.axis('equal')
ax.scatter(pts_x, pts_y, color='black', s=2)
ax.scatter(grid_x[xi], grid_y[yi], marker='+', color='red', s=200)
fig.set_size_inches(4, 4)
fig.savefig('marked.png', bbox_inches='tight')

plt.scatter(positions[:, 0], positions[:, 1], s=2, c='black')
# plt.hist2d(positions[:, 0], positions[:, 1], bins=80)
plt.show()
