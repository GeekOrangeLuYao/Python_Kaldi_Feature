"""
    feature_kernel.py
        - feature_window
        - mel_banks
        - feature_mfcc

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature.feature_window import FrameExtractionOptions
from torch_feature.feature_functions import init_preemph_matrix, init_window_function
from feature.feature_mfcc import MfccOptions
from matrix.matrix_functions import compute_dct_matrix
from feature.mel_computations import compute_lifter_coeffs


class Window(nn.Module):
    def __init__(self,
                 opts: FrameExtractionOptions,
                 requires_grad=False):
        super(Window, self).__init__()
        self.opts = opts


class StftFeature(nn.Module):
    def __init__(self,
                 opts: FrameExtractionOptions,
                 requires_grad=False):
        super(StftFeature, self).__init__()
        self.opts = opts

        enframe_basis, fft_basis = self._init_kernel()
        self.enframe = nn.Parameter(enframe_basis, requires_grad=requires_grad)
        self.fft = nn.Parameter(fft_basis, requires_grad=requires_grad)

    def _init_kernel(self):
        enframe_basis = np.eye(self.opts.get_win_size())
        enframe_basis = np.expand_dims(enframe_basis, 1)

        window = init_window_function(self.opts)
        preemph_matrix = init_preemph_matrix(self.opts.get_padded_window_size(), self.opts.preemph_coeff)
        fourier_basis = np.fft.rfft(preemph_matrix)[:self.opts.get_win_size()]
        real_kernel = np.real(fourier_basis)
        imag_kernel = np.imag(fourier_basis)
        kernel = np.concatenate([real_kernel, imag_kernel], 1)
        fft_basis = kernel.T * window

        return torch.from_numpy(enframe_basis.astype(np.float32)), torch.from_numpy(fft_basis.astype(np.float32))

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim == 2:
            inputs = torch.unsqueeze(inputs, 1)

        outputs = F.conv1d(inputs, self.enframe, stride=self.opts.get_win_shift())

        if self.opts.remove_dc_offset:
            mean = torch.mean(outputs, 1, keepdim=True)
            outputs -= mean

        outputs = torch.transpose(outputs, 1, 2)
        outputs = F.linear(outputs, self.fft, bias=None)
        outputs = torch.transpose(outputs, 1, 2)

        dim = self.opts.get_padded_window_size() // 2 + 1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim, :]

        mags = real ** 2 + imag ** 2
        return mags


class MfccFeature(nn.Module):
    """
        1. windows: remove_dc_offset -> preemph_coeff -> window_function
        2. stft: fft -> powerspectrum
        3. mel_banks
        4. dct_matrix -> cepstral_lifter
    """

    def __init__(self, opts: MfccOptions, requires_grad):
        super(MfccFeature, self).__init__()
        self.opts = opts
        num_bins = opts.mel_opts.num_bins
        dct_matrix = np.zeros((num_bins, num_bins))
        dct_matrix = compute_dct_matrix(dct_matrix)
        self.dct_matrix = nn.Parameter(torch.from_numpy(dct_matrix[:opts.num_ceps, :num_bins]),
                                       requires_grad=requires_grad)

        self.lifter_coeffs = None
        if opts.cepstral_lifter != 0.0:
            self.lifter_coeffs = np.zeros((opts.num_ceps,))
            self.lifter_coeffs = compute_lifter_coeffs(opts.cepstral_lifter, self.lifter_coeffs)
            self.lifter_coeffs = nn.Parameter(torch.from_numpy(self.lifter_coeffs), requires_grad=requires_grad)

        if opts.energy_floor > 0.0:
            self.log_energy_floor = np.log(opts.energy_floor)
