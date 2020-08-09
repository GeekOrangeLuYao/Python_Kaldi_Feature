"""
    Some basic matrix functions for matrix, we will use numpy more,

"""

import numpy as np
from scipy.fftpack import dct


def compute_dct_matrix(M: np.ndarray):
    # realization DCT Matrix
    # TODO: test the difference between this function and scipy.fftpack.dct
    K = M.shape[0]
    N = M.shape[1]

    assert K > 0
    assert N > 0

    normalizer = np.sqrt(1.0 / N)
    for j in range(N):
        M[0, j] = normalizer

    normalizer = np.sqrt(2.0 / N)
    for k in range(K):
        for n in range(N):
            M[k, n] = normalizer * np.cos(np.pi / N * (n + 0.5) * k)
    return M


def main():
    num_bins = 13
    M = np.zeros((num_bins, num_bins))
    M1 = compute_dct_matrix(M.copy())
    # M2 = dct(M.copy())

    print(M1)
    # print(M2)


if __name__ == '__main__':
    main()
