import numpy as np
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *
import matplotlib.animation as animation

hmpf = np.ones([4,4])
hmpf[2][1] = 0
imagelist = [hmpf*i*255./19. for i in range(20)]

fig = plt.figure() # make figure


# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = plt.imshow(imagelist[0], cmap=plt.get_cmap('jet'), vmin=0, vmax=255)

# function to update figure
def updatefig(j):
    # set the data in the axesimage object
    im.set_array(imagelist[j])
    # return the artists set
    return [im]
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(20),
                              interval=50, blit=True)
plt.show()