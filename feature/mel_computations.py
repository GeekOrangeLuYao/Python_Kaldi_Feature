from typing import List, Tuple
import numpy as np

from feature.feature_window import FrameExtractionOptions


def hz2mel(hz):
    return 1127 * np.log(1 + hz / 700.)


def mel2hz(mel):
    return 700 * (np.exp(mel / 1127.) - 1)


def compute_lifter_coeffs(Q, coeffs: np.ndarray) -> np.ndarray:
    for i in range(coeffs.shape[0]):
        coeffs[i] = 1.0 + 0.5 * Q * np.sin(np.pi * i / Q)
    return coeffs


class MelBanksOptions(object):

    def __init__(self,
                 num_bins=25,
                 low_freq=20,
                 high_freq=0):
        self.num_bins = num_bins
        self.low_freq = low_freq
        self.high_freq = high_freq


class MelBanks(object):
    def __init__(self,
                 opts: MelBanksOptions,
                 frame_opts: FrameExtractionOptions):
        # data domain:
        # center_freqs
        # bins
        num_bins = opts.num_bins
        assert num_bins >= 3, "Must have at least 3 mel bins"
        sample_freq = frame_opts.samp_freq
        window_length_padded = frame_opts.get_padded_window_size()

        assert window_length_padded % 2 == 0  # I think this is unnecessary
        num_fft_bins = window_length_padded / 2

        nyquist = 0.5 * sample_freq  # nyquist theory
        low_freq = opts.low_freq
        high_freq = opts.high_freq if opts.high_freq > 0 else nyquist + opts.high_freq

        if low_freq < 0.0 or low_freq >= nyquist or high_freq <= 0.0 or high_freq > nyquist or high_freq <= low_freq:
            raise ValueError(
                f"Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}")

        fft_bin_width = sample_freq / window_length_padded
        mel_low_freq = hz2mel(low_freq)
        mel_high_freq = hz2mel(high_freq)
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

        self.center_freqs = np.zeros((num_bins,))
        self.bins: List[Tuple[int, np.ndarray]] = list()  # self.bins = List(Tuple(int, List))

        for bin in range(num_bins):
            left_mel = mel_low_freq + bin * mel_freq_delta
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta

            self.center_freqs[bin] = mel2hz(center_mel)
            this_bin = np.zeros((num_fft_bins,))

            first_index = -1
            last_index = -1
            for i in range(num_fft_bins):
                freq = float(fft_bin_width * i)
                mel = hz2mel(freq)

                if left_mel < mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    this_bin[i] = weight
                    if first_index == -1:
                        first_index = i
                    last_index = i

            assert first_index != -1 and last_index >= first_index, "You may have set --num-mel-bins too large"
            self.bins.append((first_index, np.copy(this_bin[first_index: last_index])))

    def compute(self, power_spectrum: np.ndarray) -> np.ndarray:
        num_bins = len(self.bins)
        mel_energies_out = np.zeros((num_bins,))

        for i in range(num_bins):
            offset = self.bins[i][0]
            v = self.bins[i][1]
            mel_energies_out[i] = np.dot(v, power_spectrum[offset:])

        return mel_energies_out
