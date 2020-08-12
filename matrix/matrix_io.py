from base.io_functions import *
import numpy as np


def read_float_vec(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    vec_type = read_token(fd)
    # print(f"\tType of the common vector: {vec_type}")
    if vec_type not in ["FV", "DV"]:
        raise RuntimeError(f"Unknown matrix type in Kaldi: {vec_type}")
    float_size = 4 if vec_type == 'FV' else 8
    float_type = np.float32 if vec_type == "FV" else np.float64
    dim = read_int32(fd)
    # print(f"\tDim of the common vector: {dim}")
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
    # print(f"\tType of the common matrix: {mat_type}")
    if mat_type not in ["FM", "DM"]:
        raise RuntimeError(f"Unknown matrix type in kaldi: {mat_type}")
    float_size = 4 if mat_type == 'FM' else 8
    float_type = np.float32 if mat_type == 'FM' else np.float64
    num_rows = read_int32(fd)
    num_cols = read_int32(fd)
    # print(f"\tSize of the common matrix: {num_rows} x {num_cols}")
    mat_data = fd.read(float_size * num_cols * num_rows)
    mat = np.fromstring(mat_data, dtype=float_type)
    return mat.reshape(num_rows, num_cols)

def uncompress(cdata, cps_type, head):
    """
        In format CM(kOneByteWithColHeaders):
        PerColHeader, ...(x C), ... uint8 sequence ...
        first: get each PerColHeader pch for a single column
        then : using pch to uncompress each float in the column
        We load it seperately at a time

        In format CM2(kTwoByte):
        ...uint16 sequence...

        In format CM3(kOneByte):
        ...uint8 sequence...
    """
    min_val, prange, num_rows, num_cols = head
    # mat = np.zeros([num_rows, num_cols])
    # print(f'\tUncompress to matrix {num_rows} X {num_cols}')

    if cps_type == 'CM':
        # checking compressed data size, 8 is the sizeof PerColHeader
        assert len(cdata) == num_cols * (8 + num_rows)
        chead, cmain = cdata[:8 * num_cols], cdata[8 * num_cols:]
        # type uint16
        pch = np.fromstring(chead, dtype=np.uint16).astype(np.float32)
        pch = np.transpose(pch.reshape(num_cols, 4))
        pch = pch * prange / 65535.0 + min_val
        # type uint8
        uint8 = np.fromstring(cmain, dtype=np.uint8).astype(np.float32)
        uint8 = np.transpose(uint8.reshape(num_cols, num_rows))
        # precompute index
        le64_index = uint8 <= 64
        gt92_index = uint8 >= 193
        # le92_index = np.logical_not(np.logical_xor(le64_index, gt92_index))
        return np.where(le64_index,
                        uint8 * (pch[1] - pch[0]) / 64.0 + pch[0],
                        np.where(gt92_index,
                                (uint8 - 192) * (pch[3] - pch[2]) / 63.0 + pch[2],
                                (uint8 - 64) * (pch[2] - pch[1]) / 128.0 + pch[1]))
    else:
        if cps_type == 'CM2':
            inc = float(prange / 65535.0)
            uint_seq = np.fromstring(cdata, dtype=np.uint16).astype(np.float32)
        else:
            inc = float(prange / 255.0)
            uint_seq = np.fromstring(cdata, dtype=np.uint8).astype(np.float32)
        mat = min_val + uint_seq.reshape(num_rows, num_cols) * inc

    return mat


def read_compress_mat(fd):
    cps_type = read_token(fd)
    # print(f'\tFollowing matrix type: {cps_type}')
    head = struct.unpack('ffii', fd.read(16))
    # print(f'\tCompress matrix header: {head}')
    # 8: sizeof PerColHeader
    # head: {min_value, range, num_rows, num_cols}
    num_rows, num_cols = head[2], head[3]
    remain_size = 0
    if cps_type == 'CM':
        remain_size = num_cols * (8 + num_rows)
    elif cps_type == 'CM2':
        remain_size = 2 * num_rows * num_cols
    elif cps_type == 'CM3':
        remain_size = num_rows * num_cols
    else:
        throw_on_error(False, f'Unknown matrix compressing type: {cps_type}')
    # now uncompress it
    compress_data = fd.read(remain_size)
    mat = uncompress(compress_data, cps_type, head)
    return mat


def read_float_mat(fd, direct_access=False):
    if direct_access:
        expect_binary(fd)
    peek_mat_type = peek_char(fd)
    if peek_mat_type == 'C':
        return read_compress_mat(fd)
    elif peek_mat_type == 'S':
        # Kaldi have compress_matrix and sparse_matrix
        raise RuntimeError(f"Matrix type {peek_mat_type} Error")
    else:
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
