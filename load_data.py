import numpy as np
import struct
import matplotlib.pyplot as plt


def load_data(a,b):
    train_images_idx3_ubyte_file = 'MNIST_data/train-images.idx3-ubyte'
    train_labels_idx1_ubyte_file = 'MNIST_data/train-labels.idx1-ubyte'
    test_images_idx3_ubyte_file = 'MNIST_data/t10k-images.idx3-ubyte'
    test_labels_idx1_ubyte_file = 'MNIST_data/t10k-labels.idx1-ubyte'
    train_images = load_train_images(train_images_idx3_ubyte_file)
    train_labels = load_train_labels(train_labels_idx1_ubyte_file)
    test_images = load_test_images(test_images_idx3_ubyte_file)
    test_labels = load_test_labels(test_labels_idx1_ubyte_file)

    # The three-dimensional image matrix is changed to two-dimensional, and the label is transposed to a column vector
    train_data = np.zeros((784, a))
    for i in range(a):
        train_data[:, i] = train_images[i].flatten('F')
    train_data = np.transpose(train_data)  # a*784
    test_data = np.zeros((784, b))
    for i in range(b):
        test_data[:, i] = test_images[i].flatten('F')
    test_data = np.transpose(test_data)  # a*784

    train_labels = np.transpose(train_labels).reshape(1, 60000)
    train_labels = train_labels[0, 0:a].reshape(a, 1)  # a*1
    test_labels = np.transpose(test_labels).reshape(1, 10000)
    test_labels = test_labels[0, 0:b].reshape(b, 1)  # b*1
    return train_data, train_labels, test_data, test_labels


def load_train_images(train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(train_images_idx3_ubyte_file)


def load_train_labels(train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(train_labels_idx1_ubyte_file)


def load_test_images(test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(test_images_idx3_ubyte_file)

def load_test_labels(test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(test_labels_idx1_ubyte_file)


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()  # Read binary data
    # Parse the file header information, magic number, the number of pictures, the height and the width of each picture
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic num:%d, pic num: %d, pic size: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # Parse data set
    image_size = num_rows * num_cols
    # Retrieves the pointer position of the data in the cache. As you can see from the data structure described earlier, after reading the first four rows, the pointer position (offset position) points to 0016.
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # Parses the file header information, in turn for magic number and label number
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic num:%d, pic num: %d' % (magic_number, num_images))

    # parse data set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

