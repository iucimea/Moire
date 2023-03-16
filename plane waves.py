import numpy as np
from pylab import meshgrid, cm, imshow, contour, clabel, figure, title, show
import matplotlib.pyplot as plt

x = np.linspace(-20, 20, 4000)
y = np.linspace(-20, 20, 4000)
X, Y = meshgrid(x, y)


def first_order(kx, ky, scale):
    return np.cos(kx*scale)*np.cos(-0.5*kx*scale+np.sqrt(3)/2*ky*scale)*np.cos(-0.5*kx*scale-np.sqrt(3)/2*ky*scale)


def second_order(kx, ky, scale):
    return (np.cos(kx*scale)*np.cos(-0.5*kx*scale+np.sqrt(3)/2*ky*scale)*np.cos(-0.5*kx*scale-np.sqrt(3)/2*ky*scale))**2


def third_order(kx, ky, scale):
    return (np.cos(kx*scale)*np.cos(-0.5*kx*scale+np.sqrt(3)/2*ky*scale)*np.cos(-0.5*kx*scale-np.sqrt(3)/2*ky*scale))**3


a_Cr = 0.604
a_Au = 0.288

Au_real = first_order(X, Y, 2*np.pi/a_Au)
Cr_real = first_order(X, Y, 2*np.pi/a_Cr)
moire_real = Au_real * Cr_real

Cr_reciprocal = np.fft.fftshift(np.fft.fft2(Cr_real))
Au_reciprocal = np.fft.fftshift(np.fft.fft2(Au_real))
moire_reciprocal = np.fft.fftshift(np.fft.fft2(moire_real))


# plt.imshow(Cr_real, cmap=cm.jet)
# plt.imshow(Au_real, cmap=cm.jet)
# plt.imshow(moire_real, cmap=cm.jet)

# plt.imshow(abs(Cr_reciprocal), cmap=cm.jet)
# plt.imshow(abs(Au_reciprocal), cmap=cm.jet)
plt.imshow(abs(moire_reciprocal), cmap=cm.jet)
