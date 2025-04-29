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

# Define the real space lattice function
def lattice(kx, ky, scale, theta, order):
    return (np.cos(scale* kx * np.cos(theta * np.pi / 180) + scale* ky * np.sin(theta * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((120 + theta) * np.pi / 180) + scale* ky * np.sin((120 + theta) * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((240 + theta) * np.pi / 180) + scale* ky * np.sin((240 + theta) * np.pi / 180))) ** order

def lattice_disloc(kx, ky, scale, theta, order):
    return (np.cos(scale* kx * np.cos(theta * np.pi / 180) + scale* ky * np.sin(theta * np.pi / 180) + np.angle(kx+1j*ky)) \
            * np.cos(scale* kx * np.cos((120 + theta) * np.pi / 180) + scale* ky * np.sin((120 + theta) * np.pi / 180) - np.angle(kx+1j*ky)) \
            * np.cos(scale* kx * np.cos((240 + theta) * np.pi / 180) + scale* ky * np.sin((240 + theta) * np.pi / 180))) ** order

# Create the real space Au lattice
theta_Au = 50.0  # degrees
Au_real = lattice(X, Y, 2 * np.pi / a_Au, theta_Au, 3)

# Create the reciprocal space Au lattice
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))

# Define the range of theta values
theta_range = np.linspace(20, 20, 1)

# Loop over the theta values
for theta_Cr in theta_range:
    print(f"Processing theta = {theta_Cr:.2f} degrees")

    # Recompute Cr_real with the current theta
    Cr_real = lattice(X, Y, 2 * np.pi / a_Cr, theta_Cr, 6)
    moire_real = Au_real * Cr_real

    # Create the reciprocal space lattices
    Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
    moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))

    # Save the images in arrays for analysis with skimage
    Au_reciprocal_array = np.abs(Au_reciprocal)
    Cr_reciprocal_array = np.abs(Cr_reciprocal)
    moire_reciprocal_array = np.abs(moire_reciprocal)

    # Find Bragg spots in the arrays
    Au_reciprocal_Bragg_spots = peak_local_max(Au_reciprocal_array, min_distance=10, threshold_abs=3000)
    Cr_reciprocal_Bragg_spots = peak_local_max(Cr_reciprocal_array, min_distance=10, threshold_abs=3000)
    moire_reciprocal_Bragg_spots = peak_local_max(moire_reciprocal_array, min_distance=10, threshold_abs=3000)

    # Print the number of Bragg_spots found
    print(f"Au_reciprocal_Bragg_spots: {len(Au_reciprocal_Bragg_spots)}")
    print(f"Cr_reciprocal_Bragg_spots: {len(Cr_reciprocal_Bragg_spots)}")
    print(f"moire_reciprocal_Bragg_spots: {len(moire_reciprocal_Bragg_spots)}")

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
    print("\nFirst closest pairs vectors:")
    for vector in first_closest_vectors:
        print(vector)

    print("\nSecond six closest pairs vectors:")
    for vector in second_closest_vectors:
        print(vector)

    print("\nThird six closest pairs vectors:")
    for vector in third_closest_vectors:
        print(vector)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot Au_reciprocal and Cr_reciprocal with Bragg_spots on the same image
    axes[0].set_title('Reciprocal Space')
    axes[0].set_xlabel('kx')
    axes[0].set_ylabel('ky')

    # Overlay Au_reciprocal
    axes[0].imshow(abs(Au_reciprocal), cmap=cm.magma, vmin=0, vmax=10000, alpha=0.5)

    # Overlay Cr_reciprocal
    axes[0].imshow(abs(Cr_reciprocal), cmap=cm.magma, vmin=0, vmax=10000, alpha=0.5)

    # Plot yellow crosses for Au_reciprocal Bragg_spots
    for peak in Au_reciprocal_Bragg_spots:
        axes[0].plot(peak[1], peak[0], 'y+', markersize=4, markeredgewidth=1)

    # Plot red crosses for Cr_reciprocal Bragg_spots
    for peak in Cr_reciprocal_Bragg_spots:
        axes[0].plot(peak[1], peak[0], 'r+', markersize=4, markeredgewidth=1)

    # Central Bragg spots
    axes[0].plot(closest_Au_Bragg_spots[0][1], closest_Au_Bragg_spots[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')
    axes[0].plot(closest_Cr_Bragg_spots[0][1], closest_Cr_Bragg_spots[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')

    # First six closest pairs
    for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[1:7], closest_Cr_Bragg_spots[1:7]):
        axes[0].plot(Au_peak[1], Au_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')
        axes[0].plot(Cr_peak[1], Cr_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')

    # Second six closest pairs
    for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[7:13], closest_Cr_Bragg_spots[7:13]):
        axes[0].plot(Au_peak[1], Au_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')
        axes[0].plot(Cr_peak[1], Cr_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')

    # Third six closest pairs
    for Au_peak, Cr_peak in zip(closest_Au_Bragg_spots[13:19], closest_Cr_Bragg_spots[13:19]):
        axes[0].plot(Au_peak[1], Au_peak[0], 'co', markersize=6, markeredgewidth=1, markerfacecolor='none')
        axes[0].plot(Cr_peak[1], Cr_peak[0], 'co', markersize=6, markeredgewidth=1, markerfacecolor='none')

    # Plot vectors connecting the closest Bragg spots
    for i in range(len(first_closest_vectors)):  # Iterate over the actual size of the list
        vector = first_closest_vectors[i]
        axes[0].arrow(
            closest_Cr_Bragg_spots[i + 1][1], closest_Cr_Bragg_spots[i + 1][0],  # Start point (x, y)
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='magenta', width=0.2, head_width=2, length_includes_head=True
        )

    for i in range(len(second_closest_vectors)):
        vector = second_closest_vectors[i]
        axes[0].arrow(
            closest_Cr_Bragg_spots[i + 7][1], closest_Cr_Bragg_spots[i + 7][0],  # Start point (x, y)
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='green', width=0.2, head_width=2, length_includes_head=True
        )

    for i in range(len(third_closest_vectors)):
        vector = third_closest_vectors[i]
        axes[0].arrow(
            closest_Cr_Bragg_spots[i + 13][1], closest_Cr_Bragg_spots[i + 13][0],  # Start point (x, y)
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='cyan', width=0.2, head_width=2, length_includes_head=True
        )

    # Plot the vectors at the origin
    for i in range(1, 7):
        vector = first_closest_vectors[i - 1]  # Adjust index for first_closest_vectors
        axes[0].arrow(
            center[0], center[1],  # Start at origin
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='magenta', width=0.2, head_width=2, length_includes_head=True
        )

    for i in range(7, 13):
        vector = second_closest_vectors[i - 7]  # Adjust index for second_closest_vectors
        axes[0].arrow(
            center[0], center[1],  # Start at origin
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='green', width=0.2, head_width=2, length_includes_head=True
        )

    for i in range(13, 19):
        vector = third_closest_vectors[i - 13]  # Adjust index for third_closest_vectors
        axes[0].arrow(
            center[0], center[1],  # Start at origin
            vector[1],  # Delta x
            vector[0],  # Delta y
            color='cyan', width=0.2, head_width=2, length_includes_head=True
        )

    # Create custom legend handles
    Au_legend = mlines.Line2D([], [], color='yellow', marker='+', linestyle='None', markersize=6, label='Au Bragg spots')
    Cr_legend = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=6, label='Cr Bragg spots')
    center_bragg_legend = mlines.Line2D([], [], color='w', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Center spots')
    first_six_legend = mlines.Line2D([], [], color='m', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Six closest pairs')
    second_six_legend = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Second six closest pairs')
    third_six_legend = mlines.Line2D([], [], color='c', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Third six closest pairs')
    first_six_vectors_legend = mlines.Line2D([], [], color='m', marker='>', linestyle='-', linewidth=2, label='First six closest pairs')
    second_six_vectors_legend = mlines.Line2D([], [], color='g', marker='>', linestyle='-', linewidth=2, label='Second six closest pairs')
    third_six_vectors_legend = mlines.Line2D([], [], color='c', marker='>', linestyle='-', linewidth=2, label='Third six closest pairs')

    # Add the legend
    axes[0].legend(handles=[Au_legend, Cr_legend, center_bragg_legend, first_six_legend, second_six_legend, third_six_legend], loc='upper right', framealpha=0.5)

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

    # Compute the moiré lattice
    moire_lattice_computed = new_lattice(X, Y, [first_closest_vectors[5], second_closest_vectors[5]], 2) #third_closest_vectors[0]


    # Plot Real Space
    axes[1].set_title(rf'Real Space ($\theta={theta_Au-theta_Cr:.2f}^\circ$)')
    axes[1].imshow(moire_real, cmap=cm.jet)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    # Plot Computed Moiré Lattice
    axes[2].set_title('Computed Moiré Lattice')
    axes[2].imshow(moire_lattice_computed, cmap=cm.jet)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"theta_{theta_Au-theta_Cr:.2f}.png")  # Save the figure with the current theta value
    plt.show()
