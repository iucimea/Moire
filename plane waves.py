from pylab import meshgrid, cm, imshow, contour, clabel, figure, title, show
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch

x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = meshgrid(x, y)


def first_order(kx, ky, scale):
    return np.cos(kx * scale) * np.cos(-0.5 * kx * scale + np.sqrt(3) / 2 * ky * scale) * np.cos(
        -0.5 * kx * scale - np.sqrt(3) / 2 * ky * scale)


def second_order(kx, ky, scale):
    return (np.cos(kx * scale) * np.cos(-0.5 * kx * scale + np.sqrt(3) / 2 * ky * scale) * np.cos(
        -0.5 * kx * scale - np.sqrt(3) / 2 * ky * scale)) ** 2


def third_order(kx, ky, scale):
    return (np.cos(kx * scale) * np.cos(-0.5 * kx * scale + np.sqrt(3) / 2 * ky * scale) *
            np.cos(-0.5 * kx * scale - np.sqrt(3) / 2 * ky * scale)) ** 3


def first_order_rotated(kx, ky, scale):
    return np.cos(kx * np.cos(30 * np.pi / 180) * scale + ky * np.sin(30 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(150 * np.pi / 180) * scale + ky * np.sin(150 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(270 * np.pi / 180) * scale + ky * np.sin(270 * np.pi / 180) * scale)

def second_order_rotated(kx, ky, scale):
    return (np.cos(kx * np.cos(30 * np.pi / 180) * scale + ky * np.sin(30 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(150 * np.pi / 180) * scale + ky * np.sin(150 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(270 * np.pi / 180) * scale + ky * np.sin(270 * np.pi / 180) * scale)) ** 2

def third_order_rotated(kx, ky, scale):
    return (np.cos(kx * np.cos(30 * np.pi / 180) * scale + ky * np.sin(30 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(150 * np.pi / 180) * scale + ky * np.sin(150 * np.pi / 180) * scale) \
           * np.cos(kx * np.cos(270 * np.pi / 180) * scale + ky * np.sin(270 * np.pi / 180) * scale)) ** 3

a_Cr = 0.604
a_Au = 0.288

Au_real = second_order(X, Y, 2 * np.pi / a_Au)                      ### IMPORTANT ###
Cr_real = third_order_rotated(X, Y, 2 * np.pi / a_Cr)              ### IMPORTANT ###
moire_real = Au_real * Cr_real

Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))
moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))

fig = plt.figure(layout="constrained")
ax_dict = fig.subplot_mosaic(
    [#['Au', 'Au_reciprocal'],
     ['Cr', 'Cr_reciprocal'],
     ['moire_real', 'moire_reciprocal']]
)

#ax_dict['Au'].imshow(Au_real, cmap=cm.jet)
#ax_dict['Au'].set_title('Au')
#ax_dict['Au_reciprocal'].imshow(abs(Au_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['Cr'].imshow(Cr_real, cmap=plt.cm.hot)
ax_dict['Cr'].set_title('Cr')
ax_dict['Cr_reciprocal'].imshow(abs(Cr_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['moire_real'].imshow(moire_real, cmap=cm.jet)
ax_dict['moire_real'].set_title('Moire')
ax_dict['moire_reciprocal'].imshow(abs(moire_reciprocal), vmin=0, vmax=10000, cmap=cm.jet)
ax_dict['moire_reciprocal'].axis('off')

plt.show()
