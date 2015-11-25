import mnisttk
from matplotlib import pyplot as plt
from matplotlib import cm


def show(data, index):
    img = data[index, :28*28].reshape((28, 28))
    plt.imshow(255 - img, cmap=cm.Greys_r)
    plt.show()


def show_distorted(data, index, sigma, alpha):
    dx, dy = mnisttk.create_distortion_maps((28, 28), sigma, alpha)
    img = mnisttk.displace(data[index, :28*28], dx, dy).reshape((28, 28))
    plt.imshow(255 - img, cmap=cm.Greys_r)
    plt.show()