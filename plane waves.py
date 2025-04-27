import numpy as np
from pylab import meshgrid, cm, imshow, contour, clabel, figure, title, show
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from skimage.feature import peak_local_max
import matplotlib.lines as mlines
from scipy.spatial.distance import cdist


x = np.linspace(0, 20, 2000)
y = np.linspace(0, 20, 2000)
X, Y = meshgrid(x, y)

a_Cr = .604 #nm 
a_Au = .288 #nm

def lattice(kx, ky, scale, theta, order):
    return (np.cos(scale* kx * np.cos(theta * np.pi / 180) + scale* ky * np.sin(theta * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((120 + theta) * np.pi / 180) + scale* ky * np.sin((120 + theta) * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((240 + theta) * np.pi / 180) + scale* ky * np.sin((240 + theta) * np.pi / 180))) ** order 

# Create the real space lattices
Au_real = lattice(X, Y, 2 * np.pi / a_Au, 0, 3)                
Cr_real = lattice(X, Y, 2 * np.pi / a_Cr, 0, 6)                
moire_real = Au_real * Cr_real

# Create the reciprocal space lattices
Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))
moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))

# Save the images in arrays for analysis with skimage
Au_reciprocal_array = np.abs(Au_reciprocal)
Cr_reciprocal_array = np.abs(Cr_reciprocal)
moire_reciprocal_array = np.abs(moire_reciprocal)

# Find peaks in the arrays
Au_reciprocal_peaks = peak_local_max(Au_reciprocal_array, min_distance=10, threshold_abs=10000)
Cr_reciprocal_peaks = peak_local_max(Cr_reciprocal_array, min_distance=10, threshold_abs=10000)
moire_reciprocal_peaks = peak_local_max(moire_reciprocal_array, min_distance=10, threshold_abs=10000)

#plot Moiré pattern in real space
plt.figure(figsize=(8, 8))
plt.title('Real Space')
plt.imshow(moire_real, cmap=cm.jet)
plt.show()

# Calculate distances between all pairs of peaks
distances = cdist(Au_reciprocal_peaks, Cr_reciprocal_peaks)

# Find the indices of the closest pairs
closest_pairs_indices = np.unravel_index(np.argsort(distances, axis=None)[:13], distances.shape)

# Extract the closest pairs of peaks
closest_au_peaks = Au_reciprocal_peaks[closest_pairs_indices[0]]
closest_cr_peaks = Cr_reciprocal_peaks[closest_pairs_indices[1]]

# Plot Au_reciprocal and Cr_reciprocal with peaks on the same image
plt.figure(figsize=(10, 10))
plt.title('Reciprocal Space')

# Overlay Au_reciprocal
plt.imshow(abs(Au_reciprocal), cmap=cm.jet, vmin=0, vmax=10000, alpha=0.5)

# Overlay Cr_reciprocal
plt.imshow(abs(Cr_reciprocal), cmap=cm.cool, vmin=0, vmax=10000, alpha=0.5)

# Plot yellow circles for Au_reciprocal peaks
for peak in Au_reciprocal_peaks:
    plt.plot(peak[1], peak[0], 'y+', markersize=4, markeredgewidth=1)

for peak in Cr_reciprocal_peaks:
    plt.plot(peak[1], peak[0], 'r+', markersize=4, markeredgewidth=1)

# Center Bragg spots
plt.plot(closest_au_peaks[0][1], closest_au_peaks[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')
plt.plot(closest_cr_peaks[0][1], closest_cr_peaks[0][0], 'wo', markersize=6, markeredgewidth=1, markerfacecolor='none')

# First six closest pairs
for au_peak, cr_peak in zip(closest_au_peaks[1:7], closest_cr_peaks[1:7]):
    plt.plot(au_peak[1], au_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')
    plt.plot(cr_peak[1], cr_peak[0], 'mo', markersize=6, markeredgewidth=1, markerfacecolor='none')

# Second six closest pairs
for au_peak, cr_peak in zip(closest_au_peaks[7:], closest_cr_peaks[7:]):
    plt.plot(au_peak[1], au_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')
    plt.plot(cr_peak[1], cr_peak[0], 'go', markersize=6, markeredgewidth=1, markerfacecolor='none')

# Create custom legend handles
au_legend = mlines.Line2D([], [], color='yellow', marker='+', linestyle='None', markersize=6, label='Au Bragg spots')
cr_legend = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=6, label='Cr Bragg spots')
center_bragg_legend = mlines.Line2D([], [], color='w', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Center spots')
first_six_legend = mlines.Line2D([], [], color='m', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Six closest pairs')
second_six_legend = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Second six closest pairs')

# Add the legend
plt.legend(handles=[au_legend, cr_legend, center_bragg_legend, first_six_legend, second_six_legend], loc='upper right', framealpha=0.5)
plt.show()

# Print the number of peaks found
print(f"Au_reciprocal_peaks: {len(Au_reciprocal_peaks)}")
print(f"Cr_reciprocal_peaks: {len(Cr_reciprocal_peaks)}")
print(f"moire_reciprocal_peaks: {len(moire_reciprocal_peaks)}")


# fig = plt.figure(layout='constrained', figsize=(10, 10))
# fig.suptitle('Moire pattern and its reciprocal lattice', fontsize=16)
# ax_dict = fig.subplot_mosaic(
#     [['Au', 'Au_reciprocal'],
#      ['Cr', 'Cr_reciprocal'],
#      ['moire_real', 'moire_reciprocal']]
# )

# ax_dict['Au'].imshow(Au_real, cmap=cm.jet)
# ax_dict['Au'].set_title('Au')
# ax_dict['Au_reciprocal'].imshow(abs(Au_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
# ax_dict['Cr'].imshow(Cr_real, cmap=plt.cm.hot)
# ax_dict['Cr'].set_title('Cr')
# ax_dict['Cr_reciprocal'].imshow(abs(Cr_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
# ax_dict['moire_real'].imshow(moire_real, cmap=cm.jet)
# ax_dict['moire_real'].set_title('Moire')
# ax_dict['moire_reciprocal'].imshow(abs(moire_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
# ax_dict['moire_reciprocal'].axis('off')

# plt.show()