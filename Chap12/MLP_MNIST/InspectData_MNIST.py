import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(path, kind='train'):
    '''Load MNIST data from path'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte'  %kind)
    image_path = os.path.join(path, '%s-images-idx3-ubyte' %kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        # images da duoc chuan hoa thanh dang [n_samples, n_features]
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - 5.) * 2

    return images, labels

print('Inspect data:')
X_train, y_train = load_mnist('./Data', kind='train')
print('Rows %d, columns: %d' %(X_train.shape[0], X_train.shape[1]))
print('Number of train labels: %d' %y_train.shape[0])

X_test, y_test= load_mnist('./Data', kind='t10k')
print('Rows: %d, columns: %d' %(X_test.shape[0], X_test.shape[1]))
print('Number of test labels: %d' %y_test.shape[0])

#-------------------------------------------------------
# VISUALIZE SOME EXAMPLES
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

# Ta co the xem 10 chu so mau
for i in range(10):
    img = X_train[y_train==i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# Ta cung co the xem cac bien the cua cung mot thang chu so
fig, ax = plt.subplots(nrows=10, ncols=4, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(40):
    img = X_train[y_train==5][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------
# SAVE NUMPY DATA FOR USED

# np.savez_compressed('./Data/mnist_scaled.npz',
#                     X_train=X_train,
#                     y_train=y_train,
#                     X_test=X_test,
#                     y_test=y_test)


