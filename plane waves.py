import numpy as np
from pylab import meshgrid, cm, imshow, contour, clabel, figure, title, show
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

x = np.linspace(0, 10, 1000)
y = np.linspace(0, 10, 1000)
X, Y = meshgrid(x, y)

a_Cr = .604 #nm 
a_Au = .288 #nm

def lattice(kx, ky, scale, theta, order):
    return (np.cos(scale* kx * np.cos(theta * np.pi / 180) + scale* ky * np.sin(theta * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((120 + theta) * np.pi / 180) + scale* ky * np.sin((120 + theta) * np.pi / 180)) \
            * np.cos(scale* kx * np.cos((240 + theta) * np.pi / 180) + scale* ky * np.sin((240 + theta) * np.pi / 180))) ** order 

Au_real = lattice(X, Y, 2 * np.pi / a_Au, 0, 5)                
Cr_real = lattice(X, Y, 2 * np.pi / a_Cr, 0, 5)                
moire_real = Au_real * Cr_real

Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))
moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))

fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(
    [['Au', 'Au_reciprocal'],
     ['Cr', 'Cr_reciprocal'],
     ['moire_real', 'moire_reciprocal']]
)

ax_dict['Au'].imshow(Au_real, cmap=cm.jet)
ax_dict['Au'].set_title('Au')
ax_dict['Au_reciprocal'].imshow(abs(Au_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['Cr'].imshow(Cr_real, cmap=plt.cm.hot)
ax_dict['Cr'].set_title('Cr')
ax_dict['Cr_reciprocal'].imshow(abs(Cr_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['moire_real'].imshow(moire_real, cmap=cm.jet)
ax_dict['moire_real'].set_title('Moire')
ax_dict['moire_reciprocal'].imshow(abs(moire_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['moire_reciprocal'].axis('off')

plt.show()
