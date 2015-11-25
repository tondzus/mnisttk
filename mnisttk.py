import random

import ml
import struct
import numpy as np
from os.path import join
from math import floor, ceil
from operator import mul
from functools import reduce
from collections import namedtuple

from scipy.ndimage.filters import gaussian_filter


UNSIGNED_BYTE = b'\x08'
SIGNED_BYTE = b'\x09'
SHORT = b'\x0B'
INTEGER = b'\x0C'
FLOAT = b'\x0D'
DOUBLE = b'\x0E'


IdxHeader = namedtuple('IdxHeader', ['data_type', 'dimensions'])


class _IdxManipulator:
    type_dict = {
        UNSIGNED_BYTE: (np.uint8, 'B', 1),
        SIGNED_BYTE: (np.int8, 'b', 1),
        SHORT: (np.int16, 'h', 2),
        INTEGER: (np.int32, 'i', 4),
        FLOAT: (np.float32, 'f', 4),
        DOUBLE: (np.float64, 'd', 8),
        np.uint8: (UNSIGNED_BYTE, 'B'),
        np.int8: (SIGNED_BYTE, 'b'),
        np.int16: (SHORT, 'h'),
        np.int32: (INTEGER, 'i'),
        np.float32: (FLOAT, 'f'),
        np.float64: (DOUBLE, 'd'), }
    magic_number_struct = struct.Struct('>xxcb')


def decode(path):
    """Returns numpy ndarray parsed from idx file defined by path.
    """
    with open(path, 'rb')as fp:
        return IdxDecoder().read_matrix(fp.read())


def decode_file(fp):
    """Return numpy ndarray parsed from idx file. Reads file from position 0 and
    returns file pointer to original position.
    """
    original_position = fp.tell()
    fp.seek(0)
    matrix = IdxDecoder().read_matrix(fp.read())
    fp.seek(original_position)
    return matrix


def decode_bytes(byte_buffer):
    """Return numpy ndarray parsed from byte array.
    """
    return IdxDecoder().read_matrix(byte_buffer)


def encode(matrix, path):
    """Encodes numpy ndarray to file in path in idx format.
    """
    binary_data = IdxEncoder().write(matrix)
    with open(path, 'wb') as fp:
        fp.write(binary_data)


def encode_file(matrix, fp):
    """Encodes numpy ndarray into provided file in idx format. Does not return
    file to original state nor does it seek(0) before writting.
    """
    fp.write(IdxEncoder().write(matrix))


def encode_bytes(matrix):
    """Encodes numpy ndarray in idx format and returns result.
    """
    return IdxEncoder().write(matrix)


class IdxDecoder(_IdxManipulator):
    def _read_matrix_header(self, byte_buffer):
        """Reads magic number - that is first 4 bytes of file - and parse it
        into IdxHeader instance.
        """
        magic_number = byte_buffer[:4]
        data_type, dim_count = self.magic_number_struct.unpack(magic_number)
        dim_fmt = '>' + 'i' * dim_count
        dimensions_buffer = byte_buffer[4:4 + 4 * dim_count]
        dimensions = struct.unpack(dim_fmt, dimensions_buffer)
        return IdxHeader(data_type, dimensions)

    def _read_matrix_data(self, header, byte_buffer):
        """Reads and creates matrix data given correct header for them.
        """
        dt, sign, _ = self.type_dict[header.data_type]
        offset = 4 + 4 * len(header.dimensions)
        matrix = np.frombuffer(byte_buffer, dtype=dt, offset=offset)
        return matrix.reshape(header.dimensions).newbyteorder('>')

    def read_matrix(self, byte_buffer):
        """Reads next matrix from idx encoded file.
        """
        header = self._read_matrix_header(byte_buffer)
        return self._read_matrix_data(header, byte_buffer)


class IdxEncoder(_IdxManipulator):
    def _write_matrix_header(self, matrix):
        """Returns magic number and dimension bytes for given matrix.
        """
        dt, _ = self.type_dict[matrix.dtype.type]
        dim_fmt = '>' + 'i' * len(matrix.shape)
        magic_number = self.magic_number_struct.pack(dt, len(matrix.shape))
        dimensions = struct.pack(dim_fmt, *matrix.shape)
        return magic_number + dimensions

    def _write_matrix_data(self, matrix):
        """Returns matrix data bytes for given matrix.
        """
        return matrix.newbyteorder('>').tobytes()

    def write(self, matrix):
        """Returns idx encoded version of given matrix in bytes object.
        """
        header = self._write_matrix_header(matrix)
        data = self._write_matrix_data(matrix)
        return header + data


def load_train_data(path, extended=True):
    def classify(num):
        result = np.zeros(10)
        result[num] = 255.0
        return result

    prefix = 'extended' if extended else 'train'
    data = decode(join(path, prefix + '-images.idx3-ubyte'))
    labels_ = decode(join(path, prefix + '-labels.idx1-ubyte'))
    labels = np.asarray([classify(n) for n in labels_])
    available_data = np.zeros((len(data), 28*28+10), dtype=np.float32)
    available_data[:, :28*28] = data.reshape((len(data), 28*28))
    available_data[:, 28*28:] = labels
    ml.normalize(available_data, (0.0, 255.0))
    return available_data


def load_test_data(path):
    def classify(num):
        result = np.zeros(10)
        result[num] = 255.0
        return result

    data = decode(join(path, 't10k-images.idx3-ubyte'))
    labels_ = decode(join(path, 't10k-labels.idx1-ubyte'))
    labels = np.asarray([classify(n) for n in labels_])
    available_data = np.zeros((10000, 28*28+10), dtype=np.float32)
    available_data[:, :28*28] = data.reshape((10000, 28*28))
    available_data[:, 28*28:] = labels
    ml.normalize(available_data, (0.0, 255.0))
    return available_data


def extend_train_data(path, new_sample_count, sigma=2.0, alpha=8.0):
    data = decode(join(path, 'train-images.idx3-ubyte'))
    labels = decode(join(path, 'train-labels.idx1-ubyte'))
    data_shape = data.shape[0] + new_sample_count, data.shape[1], data.shape[2]

    new_data = np.zeros(data_shape, dtype=data.dtype)
    new_data[:data.shape[0]] = data
    new_label = np.zeros((labels.shape[0] + new_sample_count, ),
                         dtype=labels.dtype)
    new_label[:labels.shape[0]] = labels

    rnd = random.Random()
    offset, cur = len(data), 0
    for index in (rnd.randint(0, offset - 1) for _ in range(new_sample_count)):
        img_vector = np.zeros((28*28), dtype=np.float32)
        img_vector[:] = data[index].reshape(28*28)
        ml.normalize(img_vector, (0.0, 255.0))
        dx, dy = create_distortion_maps((28, 28), sigma, alpha)
        displaced_img = displace(img_vector, dx, dy)
        new_data[offset + cur, :, :] = np.round(displaced_img * 255).reshape((28, 28))
        new_label[offset + cur] = labels[index]
        cur += 1

    encode(new_data, join(path, 'extended-images.idx3-ubyte'))
    encode(new_label, join(path, 'extended-labels.idx1-ubyte'))


def interpolate_2d(up_left, up_right, down_left, down_right, dx, dy):
    up_val = up_left + (up_right - up_left) * dx
    down_val = down_left + (down_right - down_left) * dx
    #TODO y orientation test
    return up_val + (down_val - up_val) * dy


def create_distortion_maps(ndim, sigma, alpha):
    dx, dy = np.random.uniform(-1, 1, ndim), np.random.uniform(-1, 1, ndim)
    return alpha * gaussian_filter(dx, sigma), alpha * gaussian_filter(dy, sigma)


def displace(image, dx, dy):
    inimg = image.reshape((28, 28))
    output = np.zeros(28*28, dtype=np.float32)
    outimg = output.reshape((28, 28))
    for (x, y), _ in np.ndenumerate(inimg):
        newx, newy = x + dx[x, y], y + dy[x, y]
        if 0 <= floor(newx) < 28 and 0 <= ceil(newx) < 28 and 0 <= floor(newy) < 28 and 0 <= ceil(newy) < 28:
            outimg[x, y] = interpolate_2d(
                inimg[floor(newx), floor(newy)], inimg[ceil(newx), floor(newy)],
                inimg[floor(newx), ceil(newy)], inimg[ceil(newx), ceil(newy)],
                newx - floor(newx), newy - floor(newy)
            )
        else:
            outimg[x, y] = 0
    return output
