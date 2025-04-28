import numpy as np
from pylab import meshgrid, cm, imshow, contour, clabel, figure, title, show
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage.feature import peak_local_max
import matplotlib.lines as mlines
from scipy.spatial.distance import cdist

x = np.linspace(-15, 15, 2000)
y = np.linspace(-15, 15, 2000)
X, Y = meshgrid(x, y)
center = (X.shape[1] // 2, Y.shape[0] // 2)

# Define the lattice constants for Au and Cr
a_Cr = .604 #nm 
a_Au = .288 #nm

# Create the real space lattices
def lattice(kx, ky, scale, theta, order):
    return (np.cos(scale* kx * np.cos(theta * np.pi / 180) + scale* ky * np.sin(theta * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((120 + theta) * np.pi / 180) + scale* ky * np.sin((120 + theta) * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((240 + theta) * np.pi / 180) + scale* ky * np.sin((240 + theta) * np.pi / 180))) ** order 

Au_real = lattice(X, Y, 2 * np.pi / a_Au, 20, 3)                
Cr_real = lattice(X, Y, 2 * np.pi / a_Cr, 20, 6)                
moire_real = Au_real * Cr_real

# Create the reciprocal space lattices
Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))
moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))

# Save the images in arrays for analysis with skimage
Au_reciprocal_array = np.abs(Au_reciprocal)
Cr_reciprocal_array = np.abs(Cr_reciprocal)
moire_reciprocal_array = np.abs(moire_reciprocal)

# Find Bragg spots in the arrays
Au_reciprocal_Bragg_spots = peak_local_max(Au_reciprocal_array, min_distance=10, threshold_abs=3000)
Cr_reciprocal_Bragg_spots = peak_local_max(Cr_reciprocal_array, min_distance=10, threshold_abs=3000)
moire_reciprocal_Bragg_spots = peak_local_max(moire_reciprocal_array, min_distance=10, threshold_abs=3000)

# plot Moiré pattern in real space
plt.figure(figsize=(8, 8))
plt.title('Real Space')
plt.imshow(moire_real, cmap=cm.jet)
plt.show()

# Calculate distances between all pairs of Bragg spots
distances = cdist(Au_reciprocal_Bragg_spots, Cr_reciprocal_Bragg_spots)

# Find the indices of the closest pairs
closest_pairs_indices = np.unravel_index(np.argsort(distances, axis=None)[:19], distances.shape)

# Extract the closest pairs of Bragg_spots
closest_Au_Bragg_spots = Au_reciprocal_Bragg_spots[closest_pairs_indices[0]]
closest_Cr_Bragg_spots = Cr_reciprocal_Bragg_spots[closest_pairs_indices[1]]

# Extract distances for the three sets of pairs
center_Bragg_distance = distances[closest_pairs_indices[0][0], closest_pairs_indices[1][0]]
first_closest_distances = [distances[closest_pairs_indices[0][i], closest_pairs_indices[1][i]] for i in range(1, 7)]
second_closest_distances = [distances[closest_pairs_indices[0][i], closest_pairs_indices[1][i]] for i in range(7, 13)]
third_closest_distances = [distances[closest_pairs_indices[0][i], closest_pairs_indices[1][i]] for i in range(13, 19)]

# Print the position of the central Bragg spots
print("\nCentral Bragg position in Au:")
print(closest_Au_Bragg_spots[0])

print("\nCentral Bragg position in Cr:")
print(closest_Cr_Bragg_spots[0])

# Print the distances
print("\nCentral Bragg distance:")
print(center_Bragg_distance)

print("\nFirst closest pair distance:")
print(first_closest_distances)

print("\nSecond six closest pairs distances:")
print(second_closest_distances)

print("\nThird six closest pairs distances:")
print(third_closest_distances)

# Compute the vectors for the closest pairs
first_closest_vectors = [closest_Au_Bragg_spots[i] - closest_Cr_Bragg_spots[i] for i in range(1, 7)]
second_closest_vectors = [closest_Au_Bragg_spots[i] - closest_Cr_Bragg_spots[i] for i in range(7, 13)]
third_closest_vectors = [closest_Au_Bragg_spots[i] - closest_Cr_Bragg_spots[i] for i in range(13, 19)]

# Print the vectors
print("\nFirst closest pair vectors:")
for vector in first_closest_vectors:
    print(vector)

print("\nSecond six closest pairs vectors:")
for vector in second_closest_vectors:
    print(vector)

print("\nThird six closest pairs vectors:")
for vector in third_closest_vectors:
    print(vector)

# Plot Au_reciprocal and Cr_reciprocal with Bragg_spots on the same image
plt.figure(figsize=(10, 10))
plt.title('Reciprocal Space')

# Overlay Au_reciprocal
plt.imshow(abs(Au_reciprocal), cmap=cm.magma, vmin=0, vmax=10000, alpha=0.5)

# Overlay Cr_reciprocal
plt.imshow(abs(Cr_reciprocal), cmap=cm.magma, vmin=0, vmax=10000, alpha=0.5)

# Plot yellow crosses for Au_reciprocal Bragg_spots
for peak in Au_reciprocal_Bragg_spots:
    plt.plot(peak[1], peak[0], 'b+', markersize=5, markeredgewidth=2)

# Plot red crosses for Cr_reciprocal Bragg_spots
for peak in Cr_reciprocal_Bragg_spots:
    plt.plot(peak[1], peak[0], 'r+', markersize=4, markeredgewidth=1)

# Central Bragg spots
plt.plot(closest_Au_Bragg_spots[0][1], closest_Au_Bragg_spots[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')
plt.plot(closest_Cr_Bragg_spots[0][1], closest_Cr_Bragg_spots[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')

# First six closest pairs
for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[1:7], closest_Cr_Bragg_spots[1:7]):
    plt.plot(Au_peak[1], Au_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')
    plt.plot(Cr_peak[1], Cr_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')

# Second six closest pairs
for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[7:13], closest_Cr_Bragg_spots[7:13]):
    plt.plot(Au_peak[1], Au_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')
    plt.plot(Cr_peak[1], Cr_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')

# Third six closest pairs
for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[13:19], closest_Cr_Bragg_spots[13:19]):
    plt.plot(Au_peak[1], Au_peak[0], 'co', markersize=6, markeredgewidth=1, markerfacecolor='none')
    plt.plot(Cr_peak[1], Cr_peak[0], 'co', markersize=6, markeredgewidth=1, markerfacecolor='none')

# Plot vectors connecting the closest Bragg spots
for i in range(len(first_closest_vectors)):  # Iterate over the actual size of the list
    vector = first_closest_vectors[i]
    plt.arrow(
        closest_Cr_Bragg_spots[i + 1][1], closest_Cr_Bragg_spots[i + 1][0],  # Start point (x, y)
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='magenta', width=0.2, head_width=2, length_includes_head=True
    )

for i in range(len(second_closest_vectors)):
    vector = second_closest_vectors[i]
    plt.arrow(
        closest_Cr_Bragg_spots[i + 7][1], closest_Cr_Bragg_spots[i + 7][0],  # Start point (x, y)
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='green', width=0.2, head_width=2, length_includes_head=True
    )

for i in range(len(third_closest_vectors)):
    vector = third_closest_vectors[i]
    plt.arrow(
        closest_Cr_Bragg_spots[i + 13][1], closest_Cr_Bragg_spots[i + 13][0],  # Start point (x, y)
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='cyan', width=0.2, head_width=2, length_includes_head=True
    )

# Plot the vectors at the origin
for i in range(1, 7):
    vector = first_closest_vectors[i - 1]  # Adjust index for first_closest_vectors
    plt.arrow(
        center[0], center[1],  # Start at origin
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='magenta', width=0.2, head_width=2, length_includes_head=True
    )

for i in range(7, 13):
    vector = second_closest_vectors[i - 7]  # Adjust index for second_closest_vectors
    plt.arrow(
        center[0], center[1],  # Start at origin
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='green', width=0.2, head_width=2, length_includes_head=True
    )

for i in range(13, 19):
    vector = third_closest_vectors[i - 13]  # Adjust index for third_closest_vectors
    plt.arrow(
        center[0], center[1],  # Start at origin
        vector[1],  # Delta x
        vector[0],  # Delta y
        color='cyan', width=0.2, head_width=2, length_includes_head=True
    )

# Create custom legend handles
Au_legend = mlines.Line2D([], [], color='blue', marker='+', linestyle='None', markersize=6, label='Au Bragg spots')
Cr_legend = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=6, label='Cr Bragg spots')
center_bragg_legend = mlines.Line2D([], [], color='w', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Center spots')
first_six_legend = mlines.Line2D([], [], color='m', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Six closest pairs')
second_six_legend = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Second six closest pairs')
third_six_legend = mlines.Line2D([], [], color='c', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Third six closest pairs')
first_six_vectors_legend = mlines.Line2D([], [], color='m', marker='>', linestyle='-', linewidth=2, label='First six closest pairs')
second_six_vectors_legend = mlines.Line2D([], [], color='g', marker='>', linestyle='-', linewidth=2, label='Second six closest pairs')
third_six_vectors_legend = mlines.Line2D([], [], color='c', marker='>', linestyle='-', linewidth=2, label='Third six closest pairs')

# Add the legend
plt.legend(handles=[Au_legend, Cr_legend, center_bragg_legend, first_six_legend, second_six_legend, third_six_legend, first_six_vectors_legend, second_six_vectors_legend, third_six_vectors_legend], loc='upper right', framealpha=0.5)
plt.show()

# Print the number of Bragg_spots found
print(f"Au_reciprocal_Bragg_spots: {len(Au_reciprocal_Bragg_spots)}")
print(f"Cr_reciprocal_Bragg_spots: {len(Cr_reciprocal_Bragg_spots)}")

# Create the moiré pattern from the selected difference vectors
# Modify the new_lattice function to accept a vector
def new_lattice(kx, ky, vector, order):
    scale = 2 * np.pi / np.linalg.norm(vector)  # Inverse of the length of the vector
    theta = np.arctan2(vector[0], vector[1]) * 180 / np.pi  # Angle in degrees
    return (np.cos(scale * kx * np.cos(theta * np.pi / 180) + scale * ky * np.sin(theta * np.pi / 180)) \
            * np.cos(scale * kx * np.cos((120 + theta) * np.pi / 180) + scale * ky * np.sin((120 + theta) * np.pi / 180)) \
            * np.cos(scale * kx * np.cos((240 + theta) * np.pi / 180) + scale * ky * np.sin((240 + theta) * np.pi / 180))) ** order

# Modify the new_lattice function to accept three vectors together
def new_lattice(kx, ky, vectors, order):
    lattice_sum = np.ones_like(kx)
    for vector in vectors:
        scale = np.linalg.norm(vector)
        theta = np.arctan2(vector[0], vector[1]) * 180 / np.pi  # Angle in degrees
        lattice_sum *= (np.cos(scale * kx * np.cos(theta * np.pi / 180) + scale * ky * np.sin(theta * np.pi / 180)) \
                * np.cos(scale * kx * np.cos((120 + theta) * np.pi / 180) + scale * ky * np.sin((120 + theta) * np.pi / 180)) \
                * np.cos(scale * kx * np.cos((240 + theta) * np.pi / 180) + scale * ky * np.sin((240 + theta) * np.pi / 180))) ** order
    return lattice_sum

moire_vectors= [first_closest_vectors[0], second_closest_vectors[0], third_closest_vectors[0]]

# Compute the moiré lattice
moire_lattice_computed = new_lattice(X, Y, moire_vectors, 1)

# Plot the computed moiré lattices
fig, axes = plt.subplots(1, 2, figsize=(32, 8))

# Original moiré pattern
axes[0].set_title('Original Moiré Lattice')
axes[0].imshow(moire_real, cmap=cm.jet)

# Combined moiré lattice
axes[1].set_title('Computed Moiré Lattice')
axes[1].imshow(moire_lattice_computed, cmap=cm.jet)

plt.show()