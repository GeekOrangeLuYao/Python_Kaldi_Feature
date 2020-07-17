import math
import logging

import numpy as np
from numpy.lib.type_check import _asscalar_dispatcher

from feature.feature_window import FrameExtractionOptions

class MelBanksOptions(object):
    def __init__(self,
                 num_bins = 25,
                 low_freq = 20,
                 high_freq = 0,
                 vltn_low = 100,
                 vltn_high = -500,
                 debug_mel = False) -> None:
        self.num_bins = num_bins
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.vltn_low = vltn_low
        self.vltn_high = vltn_high
        self.debug_mel = debug_mel

class MelBanks(object):

    def __init__(self,
                 opts: MelBanksOptions,
                 frame_opts: FrameExtractionOptions,
                 vtln_warp_factor) -> None:
        self.num_bins = opts.num_bins
        if self.num_bins < 3:
            logging.error("Must have at least 3 mel bins")
        
        sample_freq = frame_opts.samp_freq
        window_length_padded = frame_opts.get_padded_window_size()

        assert window_length_padded % 2 == 0
        num_fft_bins = window_length_padded / 2

        nyquist = 0.5 * sample_freq
        low_freq = opts.low_freq

        if opts.high_freq > 0.0:
            high_freq = opts.high_freq
        else:
            high_freq = nyquist + opts.high_freq
        
        if low_freq < 0.0 or low_freq >= nyquist or high_freq <= 0.0 or high_freq > nyquist or high_freq <= low_freq:
            logging.error(f"Bad values in options: low-freq {low_freq} and high-freq {high_freq} vs. nyquist {nyquist}")
        
        fft_bin_width = sample_freq / window_length_padded
        mel_low_freq = self.MelScale(low_freq)
        mel_high_freq = self.MelScale(high_freq)
        
        self.debug = opts.debug_mel
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (self.num_bins + 1)

        vtln_low = opts.vtln_low
        vtln_high = opts.vtln_high

        if vtln_high < 0.0:
            vtln_high += nyquist
        
        if vtln_warp_factor != 1.0 and \
           (vtln_low < 0.0 or vtln_low <= low_freq \
            or vtln_low >= high_freq \
            or vtln_high <= 0.0 or vtln_high >= high_freq \
            or vtln_high <= vtln_low):
           logging.error(f"Bad values in options: vtln-low {vtln_low} and vtln-high {vtln_high}, versus low-freq {low_freq} and high-freq {high_freq}")
        
        self.bins = []
        self.center_freqs = np.zeros((self.num_bins, ))

        for bin in range(self.num_bins):
            left_mel = mel_low_freq + bin * mel_freq_delta
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta

            if vtln_warp_factor != 1.0:
                left_mel = self.VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                                vtln_warp_factor, left_mel)
                center_mel = self.VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                                  vtln_warp_factor, center_mel)
                right_mel = self.VtlnWarpMelFreq(vtln_low, vtln_high, low_freq, high_freq,
                                                 vtln_warp_factor, right_mel)

            self.center_freqs[bin] = self.InverseMelScale(center_mel)

            this_bin = np.zeros((num_fft_bins, ))
            first_index = -1
            last_index = -1
            for i in range(num_fft_bins):
                freq = fft_bin_width * i
                mel = self.MelScale(freq)

                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel-mel) / (right_mel-center_mel)
                    this_bin[i] = weight
                    if first_index == -1:
                        first_index = i
                    last_index = i
            
            assert first_index != -1 and last_index >= first_index, "You may have set --num-mel-bins too large"

            size = last_index + 1 - first_index
            temp_vector = np.copy(this_bin[first_index: first_index + size])
            self.bins.append((first_index, temp_vector))
        
        if self.debug:
            for i in range(len(self.bins)):
                logging.info(f"bin {i}, offset = {self.bins[i][0]}, vec = {self.bins[i][1]}")

    @staticmethod
    def MelScale(self, freq):
        return 1127.0 * math.log(1. + freq / 700.)
    
    @staticmethod
    def InverseMelScale(self, mel_freq):
        return 700. * (math.exp(mel_freq / 1127.) - 1.)

    @staticmethod
    def VtlnWarpFreq(self,
                     vtln_low_cutoff,
                     vtln_high_cutoff,
                     low_freq,
                     high_freq,
                     vtln_warp_factor,
                     freq):
        if freq < low_freq or freq > high_freq:
            return freq
        
        assert vtln_low_cutoff > low_freq, "be sure to set the --vtln-low option higher than --low-freq"
        assert vtln_high_cutoff < high_freq, "be sure to set the --vtln-high option lower than --high-freq [or negative]"

        one = 1.0;
        l = vtln_low_cutoff * max(one, vtln_warp_factor)
        h = vtln_high_cutoff * min(one, vtln_warp_factor)
        scale = 1.0 / vtln_warp_factor;
        Fl = scale * l
        Fh = scale * h

        assert l > low_freq and h < high_freq

        scale_left = (Fl - low_freq) / (l - low_freq)
        scale_right = (high_freq - Fh) / (high_freq - h)

        if freq < l:
            return low_freq + scale_left * (freq - low_freq)
        elif freq < h:
            return scale * freq
        else:
            high_freq + scale_right * (freq - high_freq)

    @staticmethod
    def VtlnWarpMelFreq(self,
                        vtln_low_cutoff,
                        vtln_high_cutoff,
                        low_freq,
                        high_freq,
                        vtln_warp_factor,
                        mel_freq):
        return self.MelScale(self.VtlnWarpFreq(vtln_low_cutoff, vtln_high_cutoff,
                                          low_freq, high_freq,
                                          vtln_warp_factor, self.InverseMelScale(mel_freq)))
