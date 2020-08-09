import numpy as np


def compute_power_spectrum(waveform: np.ndarray) -> np.ndarray:
    """
        The kaldi srfft is very different, which is not suitable for numpy
        Abandon this function
    """
    dim = waveform.shape[0]

    half_dim = dim // 2
    first_energy = waveform[0] ** 2
    last_energy = waveform[1] ** 2

    for i in range(1, half_dim):
        # TODO: check the i << 1 and i << 1 | 1
        real = waveform[i << 1]
        imag = waveform[i << 1 | 1]
        waveform[i] = real ** 2 + imag ** 2

    waveform[0] = first_energy
    waveform[half_dim] = last_energy

    return waveform
