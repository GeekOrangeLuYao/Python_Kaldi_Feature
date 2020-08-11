from base.io_functions import *
import numpy as np


def read_float_vec(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    vec_type = read_token(fd)
    print(f"\tType of the common vector: {vec_type}")
    if vec_type not in ["FV", "DV"]:
        raise RuntimeError(f"Unknown matrix type in Kaldi: {vec_type}")
    float_size = 4 if vec_type == 'FV' else 8
    float_type = np.float32 if vec_type == "FV" else np.float64
    dim = read_int32(fd)
    print(f"\tDim of the common vector: {dim}")
    vec_data = fd.read(float_size * dim)
    return np.fromstring(vec_data, dtype=float_type)


def read_int32_vec(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    vec_size = read_int32(fd)
    vec = np.array([read_int32(fd) for _ in range(vec_size)], dtype=np.int32)
    return vec


def read_common_mat(fd):
    mat_type = read_token(fd)
    print(f"\tType of the common matrix: {mat_type}")
    if mat_type not in ["FM", "DM"]:
        raise RuntimeError(f"Unknown matrix type in kaldi: {mat_type}")
    float_size = 4 if mat_type == 'FM' else 8
    float_type = np.float32 if mat_type == 'FM' else np.float64
    num_rows = read_int32(fd)
    num_cols = read_int32(fd)
    print(f"\tSize of the common matrix: {num_rows} x {num_cols}")
    mat_data = fd.read(float_size * num_cols * num_rows)
    mat = np.fromstring(mat_data, dtype=float_type)
    return mat.reshape(num_rows, num_cols)


def read_float_mat(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    peek_mat_type = peek_char(fd)
    if peek_mat_type == 'C' or peek_mat_type == 'S':
        # Kaldi have compress_matrix and sparse_matrix
        raise RuntimeError(f"Matrix type {peek_mat_type} Error")
    return read_common_mat(fd)


def read_float_mat_vec(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    peek_type = peek_char(fd, num_chars=2)
    if peek_type[-1] == "V":
        return read_float_vec(fd, direct_access=False)
    else:
        return read_float_mat(fd, direct_access=False)


def write_float_vec(fd, vec):
    if vec.dtype not in [np.float32, np.float64]:
        raise RuntimeError(f"Unsupported numpy dtype: {vec.dtype}")
    vec_type = 'FV' if vec.dtype == np.float32 else 'DV'
    write_token(fd, vec_type)
    if vec.ndim != 1:
        raise RuntimeError("write_float_vec accept 1D-vector only")
    dim = vec.size
    write_int32(fd, dim)
    fd.write(vec.tobytes())


def write_common_mat(fd, mat):
    if mat.dtype not in [np.float32, np.float64]:
        raise RuntimeError(f"Unsupported numpy dtype: {mat.dtype}")
    mat_type = 'FM' if mat.dtype == np.float32 else 'DM'
    write_token(fd, mat_type)
    num_rows, num_cols = mat.shape
    write_int32(fd, num_rows)
    write_int32(fd, num_cols)
    fd.write(mat.tobytes())


def write_float_mat_vec(fd, mat_or_vec):
    if isinstance(mat_or_vec, np.ndarray):
        if mat_or_vec.ndim == 2:
            write_common_mat(fd, mat_or_vec)
        else:
            write_float_vec(fd, mat_or_vec)
    else:
        raise TypeError(f"Unsupport type: {type(mat_or_vec)}")
