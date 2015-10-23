import os
import numpy as np
import struct
import pytest
import mnisttk
from io import BytesIO


@pytest.fixture(scope='module')
def ubyte_matrix(resources_root):
    return os.path.join(resources_root, 'ubyte_correct.idx')


def test_ubyte_matrix_header_parse(ubyte_matrix):
    data = open(ubyte_matrix, 'rb').read()
    parser = mnisttk.IdxDecoder()
    header = parser._read_matrix_header(data)

    assert header.data_type == mnisttk.UNSIGNED_BYTE
    assert header.dimensions == (2, 3)


def test_ubyte_matrix_data_parse(ubyte_matrix):
    data = open(ubyte_matrix, 'rb').read()
    parser = mnisttk.IdxDecoder()
    header = mnisttk.IdxHeader(mnisttk.UNSIGNED_BYTE, (2, 3))
    matrix = parser._read_matrix_data(header, data)

    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_matrix_file_load_helper(ubyte_matrix):
    matrix = mnisttk.decode(ubyte_matrix)
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_matrix_file_pointer_load_helper(ubyte_matrix):
    with open(ubyte_matrix, 'rb') as fp:
        matrix = mnisttk.decode_file(fp)
        assert matrix.dtype == np.uint8
        assert matrix.shape == (2, 3)
        assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_matrix_file_pointer_load_helper_file_position(ubyte_matrix):
    with open(ubyte_matrix, 'rb') as fp:
        fp.seek(10)
        matrix = mnisttk.decode_file(fp)
        assert fp.tell() == 10
        assert matrix.dtype == np.uint8
        assert matrix.shape == (2, 3)
        assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_matrix_file_pointer_load_helper_works_with_string_io(ubyte_matrix):
    with open(ubyte_matrix, 'rb') as fp:
        file_like_buffer = BytesIO(fp.read())
    matrix = mnisttk.decode_file(file_like_buffer)
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_matrix_byte_array_load_helper(ubyte_matrix):
    with open(ubyte_matrix, 'rb') as fp:
        byte_array = fp.read()
    matrix = mnisttk.decode_bytes(byte_array)
    assert matrix.dtype == np.uint8
    assert matrix.shape == (2, 3)
    assert np.all(matrix.reshape(6) == [1, 2, 3, 4, 5, 6])


def test_ubyte_matrix_header_encode():
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    encoder = mnisttk.IdxEncoder()
    header = encoder._write_matrix_header(matrix)
    assert header[2:] == b'\x08\x02\x00\x00\x00\x02\x00\x00\x00\x03'


def test_ubyte_matrix_data_encode():
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    encoder = mnisttk.IdxEncoder()
    data = encoder._write_matrix_data(matrix)
    assert data == struct.pack('>BBBBBB', 1, 2, 3, 4, 5, 6)


def test_ubyte_matrix_encode(ubyte_matrix):
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    encoded = mnisttk.IdxEncoder().write(matrix)
    with open(ubyte_matrix, 'rb') as fp:
        model = fp.read()
    assert encoded == model


def test_ubyte_matrix_encode_file_helper(request, ubyte_matrix):
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    new_idx_file = ubyte_matrix + '-test'

    def fin():
        try:
            os.remove(new_idx_file)
        except IOError:
            pass
    request.addfinalizer(fin)
    mnisttk.encode(matrix, new_idx_file)
    with open(ubyte_matrix, 'rb') as model, open(new_idx_file, 'rb') as test:
        written_data = test.read()
        # 1x4B magic number 2x4B dimensions 6x1B uint8 values
        assert len(written_data) == 18
        assert model.read() == written_data


def test_ubyte_matrix_encode_file_object_helper(ubyte_matrix):
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    bytes_buffer = BytesIO()
    mnisttk.encode_file(matrix, bytes_buffer)
    with open(ubyte_matrix, 'rb') as model:
        assert bytes_buffer.getvalue() == model.read()


def test_ubyte_matrix_encode_bytes_helper(ubyte_matrix):
    matrix = np.fromiter([1, 2, 3, 4, 5, 6], dtype=np.uint8).reshape((2, 3))
    encoded_data = mnisttk.encode_bytes(matrix)
    # 1x4B magic number 2x4B dimensions 6x1B uint8 values
    assert len(encoded_data) == 18
    with open(ubyte_matrix, 'rb') as model:
        assert model.read() == encoded_data
