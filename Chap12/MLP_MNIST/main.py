import os
import struct
import numpy as np
import matplotlib.pyplot as plt


'''
1. Ham doc du lieu mnist, gom 4 files
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte
- train-images-idx3-ubyte
- train-labels-idx1-ubyte

Du lieu nay gom 2 loai t10k va train
Con phia duoi la tuong tu nhau: 
Gom:
- *-images-idx3-ubyte 
- *-labels-idx1-ubyte

'''


def load_mnist(path, kind='train'):
    '''Load MNIST Data from path'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' %kind)
    image_path = os.path.join(path, '%s-images-idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        magic, num, rows, cols = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels





