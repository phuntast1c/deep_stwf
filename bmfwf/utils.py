import argparse
import datetime
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from glob import glob
from itertools import repeat
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import human_readable_ids as hri
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
import yaml
from clarity.utils.audiogram import Audiogram, Listener
from scipy import signal, stats
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


# def ensure_dir(dirname):
#     dirname = Path(dirname)
#     if not dirname.is_dir():
#         dirname.mkdir(parents=True, exist_ok=False)


# def inf_loop(data_loader):
#     """wrapper function for endless data loader."""
#     for loader in repeat(data_loader):
#         yield from loader


# @torch.jit.script
# def safe_sqrt(tensor, eps=EPS):
#     """
#     safer version of sqrt, adding EPS to all values to avoid nan
#     :param input:
#     :param kwargs:
#     :return:
#     """
#     return torch.sqrt(tensor + eps)


class STFTTorch:
    """
    class used to simplify handling of STFT & iSTFT
    """

    def __init__(
        self,
        frame_length=64,
        overlap_length=48,
        window=torch.hann_window,
        sqrt=True,
        normalized: bool = False,
        center: bool = True,
        fft_length=None,
        fft_length_synth=None,
        synthesis_window=None,
    ):
        self.frame_length = frame_length
        if fft_length is None:
            self.fft_length = frame_length
        else:
            self.fft_length = fft_length

        if fft_length_synth is None:
            self.fft_length_synth = fft_length
        else:
            self.fft_length_synth = fft_length_synth

        self.num_bins = int((self.fft_length / 2) + 1)
        self.overlap_length = overlap_length
        self.shift_length = self.frame_length - self.overlap_length
        self.sqrt = sqrt
        self.normalized = normalized
        self.center = center

        if type(window) is str:
            if window == "hann":
                window = torch.hann_window
            elif window == "hamming":
                window = torch.hamming_window
            elif window == "bartlett":
                window = torch.bartlett_window
            elif window == "blackman":
                window = torch.blackman_window
            else:
                raise ValueError("unknown window type!")
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif callable(window):
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif type(window) is torch.Tensor:
            self.window = window
        else:
            raise NotImplementedError()

        if self.sqrt:
            self.window = self.window.sqrt()

        if synthesis_window is None:
            self.synthesis_window = self.window
        else:
            self.synthesis_window = synthesis_window

    def get_stft(self, wave):
        if self.window.device != wave.device:
            # move to device
            self.window = self.window.to(device=wave.device)
        shape_orig = wave.shape
        if wave.ndim > 2:  # reshape required
            wave = wave.reshape(-1, shape_orig[-1])
        stft = torch.stft(
            wave,
            window=self.window,
            n_fft=self.fft_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=self.normalized,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )
        return stft.reshape((*shape_orig[:-1], *stft.shape[-2:]))

    def get_istft(self, stft, length=None):
        if self.synthesis_window.device != stft.device:
            # move to device
            self.synthesis_window = self.synthesis_window.to(stft.device)

        if stft.ndim == 3:  # batch x F x T
            istft = torch.istft(
                stft,
                window=self.synthesis_window,
                n_fft=self.fft_length_synth,
                hop_length=self.shift_length,
                win_length=self.frame_length,
                normalized=self.normalized,
                center=self.center,
                length=length,
                return_complex=False,
            )
        elif stft.ndim == 4:  # batch x M x F x T
            istft = torch.stack(
                [
                    torch.istft(
                        x,
                        window=self.synthesis_window,
                        n_fft=self.fft_length,
                        hop_length=self.shift_length,
                        win_length=self.frame_length,
                        normalized=self.normalized,
                        center=self.center,
                        length=length,
                        return_complex=False,
                    )
                    for x in stft
                ]
            )
        else:
            raise ValueError("unsupported STFT shape!")
        return istft


# def estimate_IBM(
#     X,
#     N,
#     threshold_unvoiced_speech=5,
#     threshold_voiced_speech=0,
#     threshold_unvoiced_noise=-10,
#     threshold_voiced_noise=-10,
#     #  low_cut=5,
#     high_cut=500,
# ):
#     """
#     taken from Heymann2017 (not really mentioned in their paper)
#     Estimate an ideal binary mask given the speech and noise spectra x and N.
#     :param X: speech signal in STFT domain with shape (frames, frequency-bins)
#     :param N: noise signal in STFT domain with shape (frames, frequency-bins)
#     :param threshold_unvoiced_speech:
#     :param threshold_voiced_speech:
#     :param threshold_unvoiced_noise:
#     :param threshold_voiced_noise:
#     :param low_cut: all values with frequency<low_cut are set to 0 in the
#         speech mask ans set to 1 in the noise mask
#     :param high_cut: all values with frequency>high_cut are set to 0 in the
#         speech mask ans set to 1 in the noise mask
#     :return: (speech mask, noise mask): tuple containing the two arrays,
#         which are the masks for  and N
#     """

#     def voiced_unvoiced_split_characteristic(iK):
#         """used by Heymann et al to determine the threshold values for the IBMs"""
#         # adjusted linearly to short frames using factor
#         factor = iK / 513
#         iK = int(iK)
#         split_bin = int(np.round(200 * factor))
#         transition_width = int(np.round(99 * factor))
#         fast_transition_width = int(np.round(5 * factor))
#         low_bin = int(np.round(4 * factor))
#         high_bin = int(np.round(500 * factor))

#         a = np.arange(0, transition_width)
#         a = a * np.pi / (transition_width - 1)
#         transition = 0.5 * (1 + np.cos(a))

#         b = np.arange(0, fast_transition_width)
#         b = b * np.pi / (fast_transition_width - 1)
#         fast_transition = 0.5 * (1 + np.cos(b))

#         transition_voiced_start = int(split_bin - transition_width / 2)
#         voiced = np.ones(iK)

#         # voiced
#         # high edge
#         voiced[
#             transition_voiced_start - 1 : (
#                 transition_voiced_start + transition_width - 1
#             )
#         ] = transition
#         voiced[transition_voiced_start - 1 + transition_width :] = 0

#         # low edge
#         voiced[0:low_bin] = 0
#         voiced[low_bin - 1 : (low_bin + fast_transition_width - 1)] = (
#             1 - fast_transition
#         )

#         # unvoiced
#         # low edge
#         unvoiced = np.ones(iK)
#         unvoiced[
#             transition_voiced_start - 1 : (
#                 transition_voiced_start + transition_width - 1
#             )
#         ] = 1 - transition
#         unvoiced[0:transition_voiced_start] = 0

#         # high edge
#         unvoiced[high_bin - 1 :] = 0
#         unvoiced[high_bin - 1 : (high_bin + fast_transition_width - 1)] = (
#             fast_transition
#         )
#         return voiced, unvoiced

#     iK = X.shape[0]
#     # also adjusted to work with Doerte's short frames
#     factor = iK / 513
#     # low_cut = 2
#     high_cut = int(np.round(high_cut * factor))
#     voiced, unvoiced = voiced_unvoiced_split_characteristic(iK)
#     threshold = threshold_voiced_speech * voiced + threshold_unvoiced_speech * unvoiced
#     threshold_new = (
#         threshold_unvoiced_noise * voiced + threshold_voiced_noise * unvoiced
#     )

#     xPSD = np.sqrt(X * X.conjugate())  # |x|^2 instead of |x|!!!

#     c = 10 ** (threshold / 10)
#     xPSD_threshold = xPSD / c[:, None]
#     c_new = 10 ** (threshold_new / 10)
#     xPSD_threshold_new = xPSD / c_new[:, None]

#     nPSD = np.sqrt(N * N.conjugate())

#     speechMask = xPSD_threshold > nPSD
#     # speechMask = (xPSD > nPSD)
#     speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005 * factor))
#     # speechMask[0:low_cut - 1] = 0
#     # speechMask[high_cut:] = 0

#     noiseMask = xPSD_threshold_new < nPSD
#     # noiseMask = (xPSD < nPSD)

#     noiseMask = np.logical_or(noiseMask, (xPSD_threshold_new < 0.005 * factor))
#     # noiseMask[0:low_cut - 1] = 1
#     # noiseMask[high_cut:] = 1

#     return speechMask.astype(np.float), noiseMask.astype(np.float)


# def estimate_naive_IRM(mX, mN):
#     """
#     estimate naive ideal ratio mask NOT regarding phase differences and without thresholds
#     """
#     return np.abs(mX) / np.abs(mN)


# def estimate_cIRMs(tX, tY, K=10, C=0.1):
#     """
#     defines the complex IRM as proposed by Williamson2016
#     :param tX: component STFT (speech / noise)
#     :param tY: noisy STFT
#     :param K: scaling factor / dynamic range
#     :param C: steepness factor
#     :return: cIRM corresponding to component x
#     """
#     Sr, Si = tX[..., 0], tX[..., 1]
#     Yr, Yi = tY[..., 0], tY[..., 1]
#     denom = Yr**2 + Yi**2
#     Mr = (Yr * Sr + Yi * Si) / denom
#     Mi = (Yr * Si - Yi * Sr) / denom

#     cIRMr = K * (1 - torch.exp(-C * Mr)) / (1 + torch.exp(-C * Mr))
#     cIRMi = K * (1 - torch.exp(-C * Mi)) / (1 + torch.exp(-C * Mi))
#     return torch.cat((cIRMr, cIRMi), dim=1)


# def decompress_cIRMs(cIRMX, cIRMN, K=10, C=0.1):
#     """
#     decompressed the cIRM{x,N} estimates
#     :param cIRMX:
#     :param cIRMN:
#     :param K:
#     :param C:
#     :return:
#     """
#     cIRMX = -1 / C * torch.log((K - cIRMX) / (K + cIRMX))
#     cIRMN = -1 / C * torch.log((K - cIRMN) / (K + cIRMN))
#     return cIRMX, cIRMN


# class CustomBatchBase:
#     def __init__(self):
#         pass

#     def pin_memory(self):
#         self.signals = {
#             key: val.pin_memory() if val is torch.Tensor else val
#             for key, val in self.signals.items()
#         }
#         return self

#     def cuda(self, device=None, non_blocking=True):
#         self.signals = {
#             key: val.cuda(device=device, non_blocking=non_blocking)
#             for key, val in self.signals.items()
#         }
#         return self

#     def to(self, device=None, dtype=None, non_blocking=True):
#         self.signals = {
#             key: val.to(device=device, dtype=dtype, non_blocking=non_blocking)
#             for key, val in self.signals.items()
#         }
#         return self


# @dataclass
# class CustomBatchSignalsMeta(CustomBatchBase):
#     signals: dict
#     meta: list

#     def __init__(self, batch: list) -> None:
#         super().__init__()

#         self.signals = default_collate(
#             [x[0] for x in batch],
#         )
#         self.meta = [x[1] for x in batch]


# def collate_fn_signals_meta(batch):
#     return CustomBatchSignalsMeta(batch)


# class CustomBatchSignalsMetaVariableLength(CustomBatchBase):
#     signals: dict
#     meta: list

#     def __init__(self, batch: list) -> None:
#         super().__init__()

#         keys = batch[0][0].keys()
#         self.signals = {}
#         for key in keys:
#             lst = [x[0][key].transpose(0, -1) for x in batch]
#             self.signals[key] = torch.nn.utils.rnn.pad_sequence(
#                 lst, batch_first=True
#             ).transpose(1, -1)
#         self.meta = [x[1] for x in batch]


# def collate_fn_signals_meta_variable_length(batch: list):
#     return CustomBatchSignalsMetaVariableLength(batch)


# def apply_batch_norm(norm, x):
#     return norm(x.permute(0, 2, 1)).permute(0, 2, 1)


# def apply_layer_norm(norm, x):
#     return norm(x)


# @torch.jit.script
# def trace(mat: torch.Tensor, keepdim: bool = False):
#     """
#     returns the trace of mat, taken over the last two dimensions
#     :param mat:
#     :return:
#     """
#     return torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


# @torch.jit.script
# def tik_reg(mat: torch.Tensor, reg: float = 0.001):
#     """
#     performs Tikhonov regularization
#     only modifies real part
#     mat: ... x M x 2
#     """
#     M = mat.shape[-2]
#     temp = ((reg * trace(mat.abs())) / M)[..., None, None] * torch.eye(
#         M, device=mat.device
#     )[None, None, ...]
#     return mat + torch.stack(
#         [temp, temp.new_zeros(size=(1,)).expand(temp.shape)], dim=-1
#     )


# def tik_reg(mat: torch.Tensor, reg: float = 1e-3, eps: float = EPS):
#     """
#     performs Tikhonov regularization
#     mat: ... x M x M
#     """
#     M = mat.shape[-2]
#     return (
#         mat
#         + ((reg * trace(mat.abs() + eps)) / M)[..., None, None]
#         * torch.eye(M, device=mat.device)[None, None, ...]
#     )


# # @torch.jit.script
# def return_tik_reg_factor(vec: torch.Tensor, reg: float = 0.001):
#     """
#     return the regularization constant that would be used in Tikhonov regularization
#     instead of computing tr of a matrix, take a vector representation (for rank-1 matrix structure)
#     """
#     iAdj = vec.shape[-2]
#     return (reg / iAdj) * (vec.abs().pow(2)).sum(-2, keepdim=True)


# # @torch.jit.script
# # def complex_tensor_outer_product(y):
# #     """
# #     outer product over complex tensors y (... x M x 2)
# #     """
# #     len_y = y.shape[-2]  # size of extra dim
# #     yyh = torch.empty(size=y.shape[:-1] + (len_y, 2), device=y.device)
# #     yyh[..., 0] = torch.einsum("...i,...j->...ij", y[..., 0], y[..., 0]) + torch.einsum(
# #         "...i,...j->...ij", y[..., 1], y[..., 1]
# #     )
# #     yyh[..., 1] = torch.einsum("...i,...j->...ij", y[..., 1], y[..., 0]) - torch.einsum(
# #         "...i,...j->...ij", y[..., 0], y[..., 1]
# #     )
# #     return yyh


def filter_minimum_gain(
    w,
    y,
    minimum_gain: float,
    mask=None,
    binaural: bool = False,
    ref_channels: tuple = None,
    keep_phase_enhanced: bool = False,
):
    if binaural:
        Y = torch.cat(
            [
                y[..., ref_channels[0], 0],
                y[..., ref_channels[1], 0],
            ],
            dim=1,
        )
    else:
        Y = y[..., -1, 0]

    output = (hermitian(w) @ y)[..., 0, 0]

    if mask is None:
        if binaural:
            mask = output.abs() < (minimum_gain * Y).abs()
        else:
            mask = output[:, :1].abs() < (minimum_gain * Y).abs()

    if keep_phase_enhanced:
        output[mask] = minimum_gain * Y[mask].abs() * output[mask] / output[mask].abs()
    else:
        output[mask] = minimum_gain * Y[mask]

    return output, mask


# def rms(x):
#     """
#     compute rms, i.e., root-mean-square of vector x
#     """
#     return np.sqrt(np.mean(np.square(x)))


# def db2pow(fdB: float) -> float:
#     """converts dB to power"""
#     return 10 ** (fdB / 10)


def db2mag(fdB: float) -> float:
    """converts dB to magnitude"""
    return 10 ** (fdB / 20)


# def filter_minimum_gain_like(
#     G_min,
#     w,
#     y,
#     alpha=None,
#     k=10.0,
#     binaural=True,
#     ref_channels: tuple = None,
# ):
#     """
#     approximate a minimum gain operation as
#     speech_estimate = alpha w^H y + (1 - alpha) G_min Y,
#     where alpha = 1 / (1 + exp(-2 k x)), x = w^H y - G_min Y
#     inputs:
#         - G_min:    minimum gain, float
#         - w:        complex-valued filter coefficients, ... x L x N x 2
#         - y:        buffered and stacked input, ... x L x N x 2
#         - k:        scaling in tanh-like function
#     outputs:
#         - minimum gain-filtered output
#         - (optional) alpha
#     """
#     filtered_input = (hermitian(w) @ y)[..., 0, 0]
#     if binaural:
#         Y = y[..., ref_channels, 0]
#     else:
#         Y = y[..., -1, 0]
#     return minimum_gain_like(G_min, Y, filtered_input, alpha, k)


# def minimum_gain_like(G_min, Y, filtered_input, alpha=None, k=10.0):
#     if alpha is None:
#         alpha = 1.0 / (
#             1.0 + torch.exp(-2 * k * (filtered_input.abs() - (G_min * Y).abs()))
#         )
#     output = alpha * filtered_input + (1 - alpha) * G_min * Y

#     return output, alpha


# def time_to_smoothing_constant(time_constant, shift_length, fs=16000):
#     """convert time constant to smoothing constant"""
#     return np.exp(-shift_length / (fs * time_constant))


# def smoothing_to_time_constant(smoothing_constant, shift_length, fs=16000):
#     """convert smoothing constant to time constant"""
#     return -shift_length / (fs * np.log(smoothing_constant))


# def pad_unfold_stack(
#     tensor: torch.tensor, filter_length: Union[torch.Tensor, float, int]
# ):
#     """
#     pad tensor such that after unfolding the number of time steps stays the same,
#     then unfold, and finally stack real and imaginary component into last dimension
#     tensor: ... x 2*frequency_bins x time_steps
#     """
#     frequency_bins = int(tensor.shape[-2] / 2)
#     output = F.pad(tensor, pad=[filter_length - 1, 0]).unfold(
#         dimension=-1, size=filter_length, step=1
#     )
#     output = torch.stack(
#         [output[:, :frequency_bins], output[:, frequency_bins:]], dim=-1
#     )
#     return output


# def save_wave(data, filename, fs=16000, normalize=False):
#     """optionally normalize with rms and save data under filename using soundfile"""
#     if data.ndim == 1:
#         data = data.unsqueeze(0)
#     if normalize:
#         denominator = (data**2).mean().sqrt()
#         data = data * (0.1 / denominator)
#         max_val = data.abs().max()
#         if max_val > 1:
#             data = data / max_val
#     ta.save(uri=filename, src=data.cpu(), sample_rate=fs, bits_per_sample=32)


# def get_measure_enhanced_noisy(output, signals, measure, **kwargs):
#     """
#     compute difference of measure on enhanced and noisy
#     """
#     try:
#         result_enhanced = measure(output, signals, noisy=False, **kwargs)
#         result_noisy = measure(output, signals, noisy=True, **kwargs)
#     except KeyError:
#         result_enhanced = np.nan
#         result_noisy = np.nan

#     return result_enhanced, result_noisy


# class MyDataParallel(torch.nn.DataParallel):
#     """
#     allow nn.DataParallel to call model's attributes
#     """

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.module, name)


# def frequency_weighting(
#     weighting_type: str, frequencies: Union[np.ndarray, torch.Tensor, float]
# ):
#     """
#     compute A-, B-, or C-weighting according to IEC/CD 1672 norm
#     follows equations from http://www.cross-spectrum.com/audio/weighting.html
#     inputs:
#         weighting_type: 'A', 'B', or 'C'
#         frequencies: vector of frequencies to compute weights for
#     output: frequency weights, same shape as frequencies
#     """

#     if weighting_type.lower() == "a":
#         return _a_weight(frequencies)
#     elif weighting_type.lower() == "b":
#         return _b_weight(frequencies)
#     elif weighting_type.lower() == "c":
#         return _c_weight(frequencies)
#     else:
#         raise ValueError


# def _c_weight(f):
#     return 12200.0**2.0 * f**2.0 / ((f**2.0 + 20.6**2.0) * (f**2.0 + 12200.0**2.0))


# def _b_weight(f):
#     return _c_weight(f) * f / ((f**2.0 + 158.5**2.0) ** 0.5)


# def _a_weight(f):
#     return _b_weight(f) * f / ((f**2.0 + 737.9**2.0) ** 0.5)


# def toeplitz(c, r=None):
#     """
#     construct a Toeplitz matrix from column c and row r
#     Args:
#         c ([torch.tensor]): ... x n x 2
#         r ([torch.tensor]): ... x n x 2
#     Returns:
#     Hermitian Toeplitz matrix of shape ... x n x n x 2
#     """
#     raise NotImplementedError("adapt to changes!")
#     if r is None:
#         r = complex_tensor_conj(c)
#     else:
#         assert r.shape == c.shape
#     c = c.flip((-2,))
#     vals = torch.cat([c, r[..., 1:, :]], dim=-2)  # 2n-1 x 2
#     N = vals.shape[-2]
#     n = c.shape[-2]
#     A = c.new_empty(c.shape[:-1] + (n, 2), requires_grad=True)
#     for nn in torch.arange(N):
#         # get diagonal indices
#         if nn < n:  # lower triangle
#             idx1 = torch.arange(n - 1 - nn, n)
#             idx2 = torch.arange(nn + 1)
#         else:  # upper diagonal excluding diagonal
#             idx1 = torch.arange(0, N - nn)
#             idx2 = torch.arange(nn - n + 1, n)
#         A[..., idx1, idx2, :] = vals[..., nn, :].unsqueeze(-2).clone()
#     return A


# # @torch.jit.script
# def vandermonde_balanced(x: torch.Tensor):
#     """generate balanced Vandermonde matrix from input tensor x

#     Args:
#         x (torch.tensor): desired phase; ... x n x 1

#     Returns:
#         [type]: ... x n x n x 2
#     """
#     n = x.shape[-2]
#     exp_input = 1j * x * torch.arange(n, device=x.device)
#     return torch.exp(exp_input)


# def get_hermitian_toeplitz(x: torch.tensor, ensure_psd=False):
#     """generate n x n Hermitian Toeplitz matrix from 2n real coefficients
#     exploits balanced Vandermonde factorization
#     note: the elements used for the diagonal matrix need to be positive for the resulting matrix to be psd
#     thus, if desired, softplus is applied to the corresponding input elements
#     Args:
#         x (torch.tensor): ... x 2n x 1
#         ensure_psd (bool): whether or not to ensure that the resulting matrix is psd
#     """
#     # get Vandermonde and diagonal matrices
#     m = x.shape[-2]
#     n = int((m + 1) / 2)
#     vand_balanced = vandermonde_balanced(x[..., :n, :])
#     if ensure_psd:
#         diag = torch.diag_embed(F.softplus(x[..., n:, 0])) + 0j
#     else:
#         diag = torch.diag_embed(x[..., n:, 0]) + 0j
#     # multiply to get Hermitian psd Toeplitz matrix
#     return hermitian(vand_balanced) @ diag @ vand_balanced


def vector_to_Hermitian(vec: torch.Tensor) -> torch.Tensor:
    """
    this function constructs a ... x N x N-dim. complex Hermitian matrix from a vector of N**2 independent real-valued components
    broadcasting enabled
    input:
        vec: ... x N**2
    output:
        mat: ... x N x N
    """
    N = int(np.sqrt(vec.shape[-1]))

    # init and get indices
    tril = np.tril_indices(N, 0)
    tril2 = np.tril_indices(N, -1)  # above main diagonal
    cut = tril[0].shape[0]

    # choose elements that will sit on diagonal, then force them to be non-negative
    # diag_elements = (0, N+1, 2(N+1), ...)
    diag_elements = [
        idx for idx, (el1, el2) in enumerate(zip(tril[0], tril[1])) if el1 == el2
    ]
    vec[..., diag_elements] = F.softplus(vec[..., diag_elements])

    mat = vec.new_zeros(size=vec.shape[:-1] + (N, N), dtype=torch.cfloat)

    mat.real[(...,) + tril] = vec[..., :cut]
    mat.imag[(...,) + tril2] = vec[..., cut:]

    # A = B @ B^H is Hermitian
    mat = mat @ hermitian(mat)
    return mat


# class Bunch(object):
#     """
#     transforms adict into an object such that adict values can be accessed using "." notation
#     """

#     def __init__(self, adict) -> None:
#         self.__dict__.update(adict)


# def solve_l2_reg(
#     B: torch.Tensor, A: torch.Tensor, reg: float, out: torch.Tensor = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """solve the LS problem ||A X - B||^2_2 + ||reg * I X||^2_2

#     Args:
#         input (torch.Tensor): [description]
#         A (torch.Tensor): [description]
#         reg (float): [description]
#         out (torch.Tensor, optional): [description]. Defaults to None.

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: [description]
#     """
#     dim_A = A.shape[-2]
#     eye = torch.eye(dim_A, dtype=A.dtype, device=A.device)
#     A_solve = A.transpose(-2, -1) @ A + reg**2 * eye.transpose(-2, -1) @ eye
#     B_solve = A.transpose(-2, -1) @ B
#     return B_solve.solve(A_solve)


# class UnfoldFoldHelper:
#     """
#     accept a batched time-domain signal, B x L, and handle unfolding and folding operations
#     """

#     def __init__(self, size: int, step: int) -> None:
#         self.unfold_params = {"kernel_size": (1, size), "stride": (1, step)}
#         self.unfold = partial(F.unfold, **self.unfold_params)
#         self.fold = partial(F.fold, **self.unfold_params)

#     def get_unfolded(self, tensor: torch.Tensor) -> torch.Tensor:
#         return self.unfold(tensor[:, None, None, :]).transpose(-2, -1).contiguous()

#     def get_folded(self, tensor: torch.Tensor, output_size: int) -> torch.Tensor:
#         return self.fold(tensor.transpose(-2, -1), output_size=(1, output_size))[
#             :, 0, 0
#         ]


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ("yes", "true", "t", "y", "1"):
#         return True
#     elif v.lower() in ("no", "false", "f", "n", "0"):
#         return False
#     else:
#         raise argparse.ArgumentTypeError("Boolean value expected.")


# def str2int_or_float(s: str):
#     """convert a string into either an integer or a float, depending on its value

#     Args:
#         s ([str]): string
#     """
#     if type(s) in [int, float]:
#         return s
#     else:
#         return int(s) if (float(s) % 1.0) == 0.0 else float(s)


# def convert_smoothing_constant(smoothing1: float, shift1: int, shift2: int) -> float:
#     """convert smoothing constant to new STFT framework

#     Args:
#         smoothing1 (float): [original smoothing constant]
#         shift1 (int): [original frame shift]
#         shift2 (int): [new frame shift]

#     Returns:
#         float: [description]
#     """
#     return smoothing1 ** (shift2 / shift1)


def hermitian(X: torch.Tensor) -> torch.Tensor:
    return X.mH  # now with native support (which should do exactly the same...)


# def normalize_level(t: torch.Tensor, target_level: float = -20):
#     std = t.std()
#     factor = 10 ** (target_level / 20) / (std + EPS)
#     return t * factor, factor


# def plt_spectrogram(
#     waveform: torch.Tensor, sample_rate=16000, title="Spectrogram", xlim=None, log=True
# ):
#     f, t, Sxx = signal.spectrogram(waveform.numpy(), fs=sample_rate)
#     F, T = np.meshgrid(t, f)
#     plt.figure()
#     plt.pcolormesh(F, T, np.log10(Sxx + np.finfo(np.float).eps), cmap="magma")
#     return plt


# def get_checkpoint_path(
#     str_proj_id: str, last: bool, dir_path: Optional[str] = None
# ) -> str:
#     """
#     get checkpoint path from project name and I
#     just takes the first checkpoint in that folder it can find!
#     """
#     if dir_path is None:
#         dir_path = os.path.dirname(os.path.realpath(__file__))

#     _, project, id = str_proj_id.split("/")

#     pth = os.path.join(dir_path, "saved", "**", "*-" + id, "..", "..")
#     fns = retrieve_checkpoint_paths(pth, last)
#     if len(fns) == 0:  # also try the other pattern
#         pth = os.path.join(dir_path, "saved", "*-" + id, "**")
#         fns = retrieve_checkpoint_paths(pth, last)

#     print(f"found {len(fns)} checkpoints in {pth}!")
#     return fns[0]


# def retrieve_checkpoint_paths(pth: str, last: bool):
#     if last:
#         pth = os.path.join(
#             pth,
#             "last.ckpt",
#         )
#     else:
#         pth = os.path.join(
#             pth,
#             "epoch*.ckpt",
#         )

#     fns = glob(pth, recursive=True)
#     return fns


# def get_results_paths(str_proj_id: str, dir: Optional[str] = None) -> list:
#     """
#     get results path(s) from run ID
#     """
#     _, _, wandb_id = str_proj_id.split("/")
#     if dir is None:
#         wandb_dir = (
#             "/home/marvint/dr/code/deep_learning_projects/clarity_challenge/wandb/"
#         )
#     else:
#         wandb_dir = os.path.join(dir, "wandb")
#     return natsort.natsorted(
#         glob(os.path.join(wandb_dir, f"*{wandb_id}", "files", "test", "*.results"))
#     )


# def get_results_csv_from_wandb_run_path(
#     str_proj_id: str, dir: Optional[str] = None
# ) -> str:
#     _, _, wandb_id = str_proj_id.split("/")
#     if dir is None:
#         dir_path = os.path.dirname(os.path.realpath(__file__))
#     else:
#         dir_path = dir
#     pth = os.path.join(dir_path, "saved", "**", f"*-{wandb_id}", "..", "..")
#     return natsort.natsorted(glob(os.path.join(pth, "test", "*.csv"), recursive=True))


# def get_results_csv_from_default_root_dir(
#     root_dir: str, dir: Optional[str] = None
# ) -> str:
#     return natsort.natsorted(
#         glob(os.path.join(root_dir, "test", "*.csv"), recursive=True)
#     )


# def get_cfg_path(path_info: str, dir: Optional[str] = None) -> str:
#     # first try new folder structure
#     cfg_path = glob(os.path.join(path_info, "config.yaml"), recursive=True)
#     if len(cfg_path) > 0:
#         return cfg_path[0]
#     else:
#         # second, try deprecated folder structure
#         _, _, wandb_id = path_info.split("/")
#         if dir is None:
#             wandb_dir = os.path.dirname(os.path.realpath(__file__))
#         else:
#             wandb_dir = dir
#         wandb_dir = os.path.join(
#             wandb_dir, "saved", "**", f"*-{wandb_id}", "files", "config.yaml"
#         )
#         cfg_path = glob(wandb_dir, recursive=True)
#         if len(cfg_path) > 0:
#             return cfg_path[0]
#         else:
#             cfg_path = get_cfg_path_deprecated(path_info, dir)
#             if len(cfg_path) > 0:
#                 return cfg_path[0]
#             else:
#                 raise ValueError(f"no config file found at {wandb_dir}!")


# def get_cfg_path_deprecated(str_proj_id: str, dir) -> str:
#     """
#     get results path(s) from run ID
#     """
#     _, _, id = str_proj_id.split("/")
#     # go to folder, look up corresponding config path
#     if dir is None:
#         wandb_dir = (
#             "/home/marvint/dr/code/deep_learning_projects/clarity_challenge/wandb/"
#         )
#     else:
#         wandb_dir = os.path.join(dir, "wandb")
#     return glob(os.path.join(wandb_dir, f"*{id}", "files", "config.yaml"))


# def principal_eigenvector(t: torch.Tensor) -> torch.Tensor:
#     # perform EVD
#     L, V = torch.linalg.eig(t)
#     # get vector corresponding to largest eigenvalue
#     _, idx_max = L.abs().sort(dim=-1, descending=True)
#     idx_max = idx_max[..., :1, None]
#     idx_max = idx_max.repeat((idx_max.ndim - 2) * (1,) + (L.shape[-1], 1))
#     return V.gather(dim=-1, index=idx_max)


# def my_explode(df: pd.DataFrame, measures: list, val_name: str = "improvement"):
#     """if Dataframe is saved as all measures per index, this function can be used to
#     separate the measures into different indices, and save the respective values under a new name

#     Args:
#         df (pd.DataFrame): [description]
#         measures (list): [description]
#         val_name (str, optional): [description]. Defaults to "improvement".

#     Returns:
#         [type]: [description]
#     """
#     df_out = pd.DataFrame()
#     for measure in measures:
#         df_tmp = df.copy()
#         df_tmp[val_name] = df_tmp[measure]
#         df_tmp = df_tmp.drop(columns=measures)
#         df_tmp["measure"] = measure
#         df_out = pd.concat([df_out, df_tmp], ignore_index=True)
#     return df_out


# def compute_significances(
#     df: pd.DataFrame, col_category: str, col_val: str, pmax: float = 0.05
# ):
#     for key1 in df[col_category].unique():
#         for key2 in df[col_category].unique():
#             if key1 != key2:
#                 statistic, pvalue = stats.ttest_ind(
#                     a=df.loc[(df[col_category] == key1), col_val],
#                     b=df.loc[(df[col_category] == key2), col_val],
#                     equal_var=False,
#                     alternative="greater",
#                 )
#                 if pvalue < pmax:
#                     print(
#                         f"{key1} significantly outperforms {key2} in terms of {col_val}! p = {pvalue:.4f}"
#                     )


# def get_results_dfs(path_info: str, dir: Optional[str] = None) -> pd.DataFrame:
#     # load results
#     # use one of the load functions: new, deprecated, deprecated_deprecated
#     results_paths = get_results_csv_from_default_root_dir(path_info, dir)
#     if len(results_paths) == 0:
#         results_paths = get_results_csv_from_wandb_run_path(path_info, dir)
#         if len(results_paths) == 0:
#             results_paths = get_results_paths(path_info, dir)
#             if len(results_paths) == 0:
#                 print(
#                     f"did not even find any results files for {path_info}; returning empty dataframe..."
#                 )
#                 return pd.DataFrame()

#     if ".results" in results_paths[0]:
#         dfs = [pd.read_json(results_path) for results_path in results_paths]
#     elif ".csv" in results_paths[0]:
#         dfs = [pd.read_csv(results_path) for results_path in results_paths]
#     else:
#         raise ValueError(f"unknown file type for {results_paths[0]}")

#     for idx, df in enumerate(dfs):
#         # compute some additional information
#         df["test set idx"] = idx

#     # load corresponding config file
#     cfg_path = get_cfg_path(path_info, dir)
#     with open(cfg_path, "r") as f:
#         cfg = yaml.full_load(f)
#     try:
#         cfg_values = {x: cfg[x]["value"] for x in cfg.keys() if type(cfg[x]) is dict}
#         # exclude dicts and lists from config values
#         cfg_to_transfer = {
#             key: val for key, val in cfg_values.items() if type(val) not in [dict, list]
#         }
#         dfs = [
#             pd.concat(
#                 [df, pd.DataFrame(cfg_to_transfer, index=np.arange(len(df)))], axis=1
#             )
#             for df in dfs
#         ]
#     except KeyError:  # using new saving format
#         cfg_to_transfer = flatten_dict(cfg)
#         dfs = [
#             pd.concat([df, pd.DataFrame(cfg_to_transfer, index=[0])], axis=1)
#             for df in dfs
#         ]

#     return pd.concat(dfs, axis=0, ignore_index=True)


# def flatten_dict(d, parent_key="", sep="_"):
#     """
#     This function flattens a nested dictionary by concatenating keys separated by a separator.
#     """
#     items = []
#     if isinstance(d, dict):
#         for k, v in d.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else k
#             if isinstance(v, dict):
#                 items.extend(flatten_dict(v, new_key, sep=sep).items())
#             elif not isinstance(v, Iterable):
#                 items.append((new_key, v))
#     else:
#         items.append((parent_key, d))
#     return dict(items)


# def plot_get_size(width="single", fraction=1.0, subplots=(1, 1)):
#     """Set figure dimensions to avoid scaling in LaTeX.

#     Parameters
#     ----------
#     width: float
#             Document textwidth or columnwidth in inches
#     fraction: float, optional
#             Fraction of the width the figure should occupy
#     Returns
#     -------
#     fig_dim: tuple
#             Dimensions of figure in inches
#     """
#     # Width of figure (in pts)
#     if width == "single":
#         width = 3.5
#     elif width == "double":
#         width = 7.16
#     else:
#         raise ValueError(f"unknown width {width}!")

#     fig_width = width * fraction

#     # Golden ratio to set aesthetic figure height
#     # https://disq.us/p/2940ij3
#     golden_ratio = (5**0.5 - 1) / 2
#     fig_height = fig_width * golden_ratio * (subplots[0] / subplots[1])

#     return fig_width, fig_height


# def pad(t: torch.Tensor, pad: list, dim: int) -> torch.Tensor:
#     """
#     easier implementation of padding...
#     """
#     shape_pre = list(t.shape)
#     shape_pre[dim] = pad[0]
#     shape_post = list(t.shape)
#     shape_post[dim] = pad[1]
#     pad_pre = t.new_zeros(shape_pre)
#     pad_post = t.new_zeros(shape_post)
#     return torch.cat([pad_pre, t, pad_post], dim=dim)


# def active_rms(
#     clean, noise, fs=16000, energy_thresh=0.13, estimate_activity: bool = True
# ):
#     """Returns the clean and noise RMS of the noise calculated only in the active portions"""
#     # use activity detector to detect joint activity of speech and noise
#     # then compute rms based on this set
#     if estimate_activity:
#         _, indices_active_clean = activitydetector(
#             clean, fs, return_indices=True, energy_thresh=energy_thresh
#         )
#         _, indices_active_noise = activitydetector(
#             noise, fs, return_indices=True, energy_thresh=energy_thresh
#         )
#         indices_active_joint = indices_active_clean & indices_active_noise
#         if not np.any(
#             indices_active_joint
#         ):  # impulsive noise... consider noise activity
#             indices_active_joint[:] = indices_active_noise
#         clean_rms = (clean[indices_active_joint] ** 2).mean() ** 0.5 + EPS
#         noise_rms = (noise[indices_active_joint] ** 2).mean() ** 0.5 + EPS
#     else:
#         clean_rms = (clean**2).mean() ** 0.5 + EPS
#         noise_rms = (noise**2).mean() ** 0.5 + EPS
#     return clean_rms, noise_rms


# def activitydetector(
#     audio, fs=16000, energy_thresh=0.13, target_level=-25, return_indices=False
# ):
#     """Return the percentage of the time the audio signal is above an energy threshold"""

#     audio = normalize(audio, target_level)
#     window_size = 50  # in ms
#     window_samples = int(fs * window_size / 1000)
#     sample_start = 0
#     cnt = 0
#     prev_energy_prob = 0
#     active_frames = 0

#     a = -1
#     b = 0.2
#     alpha_rel = 0.05
#     alpha_att = 0.8

#     if return_indices:
#         indices_active = np.zeros(audio.shape, dtype=bool)

#     while sample_start < len(audio):
#         sample_end = min(sample_start + window_samples, len(audio))
#         audio_win = audio[sample_start:sample_end]
#         frame_rms = 20 * np.log10(sum(audio_win**2) + EPS)
#         frame_energy_prob = 1.0 / (1 + np.exp(-(a + b * frame_rms)))

#         if frame_energy_prob > prev_energy_prob:
#             smoothed_energy_prob = frame_energy_prob * alpha_att + prev_energy_prob * (
#                 1 - alpha_att
#             )
#         else:
#             smoothed_energy_prob = frame_energy_prob * alpha_rel + prev_energy_prob * (
#                 1 - alpha_rel
#             )

#         if smoothed_energy_prob > energy_thresh:
#             active_frames += 1
#             if return_indices:
#                 indices_active[sample_start:sample_end] = True

#         prev_energy_prob = frame_energy_prob
#         sample_start += window_samples
#         cnt += 1

#     perc_active = active_frames / cnt

#     return (perc_active, indices_active) if return_indices else perc_active


# def normalize(audio, target_level=-25):
#     """Normalize the signal to the target level"""
#     rms = (audio**2).mean() ** 0.5
#     scalar = 10 ** (target_level / 20) / (rms + EPS)
#     audio = audio * scalar
#     return audio


# def get_config(pth_ckpt: str):
#     try:
#         cfg = torch.load(pth_ckpt)["hyper_parameters"]["config"]
#     except (KeyError, FileNotFoundError):
#         pth_ckpt = os.path.split(pth_ckpt)[0]
#         pth_ckpt = pth_ckpt.replace("/saved/", "/wandb/")
#         pth_ckpt = pth_ckpt.replace("/checkpoints", "")
#         pth_ckpt = os.path.join(pth_ckpt, "config.yaml")
#         print(f"did not find config in checkpoint. loading from {pth_ckpt} instead.")
#         with open(pth_ckpt, "r") as f:
#             cfg_loaded = yaml.safe_load(f)
#         cfg = {
#             key.replace("config/", ""): val["value"]
#             for key, val in cfg_loaded.items()
#             if "wandb" not in key
#         }
#         for key, val in cfg.items():
#             if val == "None":
#                 cfg[key] = None
#     return cfg


# def return_unique_filepath_from_glob(glob_str: str) -> str:
#     paths_matched = glob(glob_str, recursive=True)
#     assert len(paths_matched) == 1
#     return paths_matched[0]


def dcn(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def get_save_dir():
    path_current_file = os.path.dirname(os.path.abspath(__file__))
    str_year_month_day = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        path_current_file,
        "saved",
        f"{str_year_month_day}_" + hri.get_new_id().lower().replace(" ", "-"),
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# def view_as_complex_dim(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     # Check that the dimension to be converted is valid.
#     assert abs(dim) <= len(input_tensor.shape), (
#         f"Dimension {dim} is out of bounds for tensor of shape {input_tensor.shape}."
#     )

#     # Check that the dimension to be converted is of size 2.
#     assert input_tensor.shape[dim] == 2, (
#         f"Dimension {dim} of tensor must be of size 2, but got {input_tensor.shape[dim]}."
#     )

#     # Convert the real and imaginary parts to a complex tensor.
#     return torch.complex(input_tensor.select(dim, 0), input_tensor.select(dim, 1))


# def clone_tensors(tuples):
#     """Clone a tuple, list, or dict of tensors.

#     Args:
#         tuples (tuple, list, dict, or torch.Tensor): The input to clone.

#     Returns:
#         tuple, list, dict, or torch.Tensor: A clone of the input.
#     """
#     if isinstance(tuples, torch.Tensor):
#         return tuples.clone()
#     elif isinstance(tuples, tuple):
#         return tuple(clone_tensors(t) for t in tuples)
#     elif isinstance(tuples, list):
#         return [clone_tensors(t) for t in tuples]
#     elif isinstance(tuples, dict):
#         return {k: clone_tensors(v) for k, v in tuples.items()}
#     else:
#         return tuples


# def identity(x):
#     return x


# def _plot_example(t: torch.Tensor, line: str, idx: int):
#     # t: T x F
#     fig, ax = plt.subplots()
#     img = ax.matshow(t.T.detach().cpu().numpy(), aspect="auto")
#     fig.colorbar(img, ax=ax)
#     fig.savefig(f"tmp/{idx}_{line}.png")


# def repackage_hidden(h):
#     """
#     Wraps hidden states in new tensor, to detach them from their history.
#     """

#     if type(h) == torch.tensor:
#         return torch.tensor(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)


# def helper_haspi_hasqi_v2_be(
#     fn: callable, fs: int, ref: torch.Tensor, deg: torch.Tensor
# ) -> float:
#     # separate into left and right before feeding
#     refl, refr = ref[..., 0, :], ref[..., 1, :]
#     estl, estr = deg[..., 0, :], deg[..., 1, :]

#     if refl.ndim == 1:  # batch x M x samples
#         refl, refr = refl[None, ...], refr[None, ...]
#         estl, estr = estl[None, ...], estr[None, ...]
#     assert refl.shape == refr.shape == estl.shape == estr.shape

#     # also requires left and right listener audiograms...
#     audiogram_cfs = np.asarray([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
#     audiogram_levels_no_hl = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
#     audiogram_left = Audiogram(levels=audiogram_levels_no_hl, frequencies=audiogram_cfs)
#     audiogram_right = Audiogram(
#         levels=audiogram_levels_no_hl, frequencies=audiogram_cfs
#     )
#     listener = Listener(audiogram_left, audiogram_right)

#     result = [
#         fn(
#             xl,
#             xr,
#             yl,
#             yr,
#             sample_rate=fs,
#             listener=listener,
#         )
#         for xl, xr, yl, yr in zip(refl, refr, estl, estr)
#     ]
#     return sum(result) / len(result)


# def normalize_to_active_rms(signal, target_db, fs, energy_thresh=0.13):
#     """
#     Normalizes the input signal to have the specified active RMS level based on active regions.
#     """
#     active_indices = [
#         activitydetector(x, fs, return_indices=True, energy_thresh=energy_thresh)[1]
#         for x in signal
#     ]

#     rmss = []
#     for x, indices in zip(signal, active_indices):
#         if indices.sum() == 0:
#             raise ValueError("active_indices contain no active samples")
#         rmss.append(x[indices].pow(2).mean().sqrt())

#     rms = sum(rmss) / len(rmss)
#     if rms == 0:
#         raise ValueError("RMS of active regions is zero, cannot normalize")

#     target_rms = 10 ** (target_db / 20)
#     return target_rms / rms


# def scale_and_mix_signals(
#     clean: torch.Tensor,
#     noise: torch.Tensor,
#     snr_desired: float,
#     target_level_desired: float,
#     fs: int,
#     indices_ref: torch.Tensor,
#     activitydetector: callable = activitydetector,
# ):
#     """
#     Scale a clean speech signal to a specific active RMS level, add noise at a given SNR,
#     and prevent clipping in the resulting signal.

#     Args:
#         clean (torch.Tensor): Clean speech signal.
#         noise (torch.Tensor): Noise signal.
#         snr_desired (float): Desired SNR in dB.
#         target_levels (tuple): Range (min, max) for target active RMS levels (in dB).
#         fs (int): Sampling rate of the signals.
#         indices_ref (torch.Tensor): Reference indices for multi-channel or segment-based operations.
#         activitydetector (callable): Function to detect active regions. Should return active indices.

#     Returns:
#         torch.Tensor: Clean signal scaled.
#         torch.Tensor: Noise signal scaled.
#         torch.Tensor: Resulting noisy signal.
#     """
#     # Clone signals to prevent modification of original inputs
#     clean = clean.clone()
#     noise = noise.clone()

#     # 1. Normalize clean signal to target level using active regions
#     _, indices_active_clean = activitydetector(
#         clean[indices_ref[0]], fs, return_indices=True
#     )

#     rms_clean = (
#         clean[indices_ref[:, None], indices_active_clean].pow(2).mean(1).sqrt().mean()
#     )  # average across reference channels
#     current_level_db = 20 * torch.log10(rms_clean + EPS)
#     scale_fac = 10 ** ((target_level_desired - current_level_db) / 20)
#     clean *= scale_fac

#     # Ensure no clipping
#     max_val = clean.abs().max()
#     if max_val > 1:
#         clean /= 1.1 * max_val

#     # 2. Mix clean and noise signals using jointly active regions
#     _, indices_active_noise = activitydetector(
#         noise[indices_ref[0]], fs, return_indices=True
#     )
#     indices_active_both = indices_active_clean & indices_active_noise
#     if indices_active_both.sum() == 0:
#         print("No active regions found in both signals! Using complete signals.")
#         indices_active_both[:] = True

#     rms_clean = (
#         clean[indices_ref[:, None], indices_active_both].pow(2).mean(1).sqrt().mean()
#     )
#     rms_noise = (
#         noise[indices_ref[:, None], indices_active_both].pow(2).mean(1).sqrt().mean()
#     )

#     snr_linear = 10 ** (snr_desired / 20)
#     scale_noise = (rms_clean / (rms_noise + EPS)) / snr_linear
#     noise *= scale_noise

#     # 3. Mix signals
#     noisy = clean + noise

#     # 4. Ensure no clipping
#     max_val = noisy.abs().max()
#     if max_val > 1:
#         max_val *= 1.1
#         clean /= max_val
#         noise /= max_val
#         noisy /= max_val

#     return clean, noise, noisy


# def format_memory(memory_bytes):
#     if memory_bytes < 1e6:
#         return f"{memory_bytes / 1e3:.2f} KB"
#     elif memory_bytes < 1e9:
#         return f"{memory_bytes / 1e6:.2f} MB"
#     else:
#         return f"{memory_bytes / 1e9:.2f} GB"


# def format_flops(flops):
#     if flops < 1e6:
#         return f"{flops / 1e3:.2f} KFLOPS"
#     elif flops < 1e9:
#         return f"{flops / 1e6:.2f} MFLOPS"
#     else:
#         return f"{flops / 1e9:.2f} GFLOPS"


# def format_parameters(params):
#     if params < 1e6:
#         return f"{params / 1e3:.2f} K"
#     elif params < 1e9:
#         return f"{params / 1e6:.2f} M"
#     else:
#         return f"{params / 1e9:.2f} B"


# @contextmanager
# def suppress_stdout():
#     # Save the current stdout
#     original_stdout = sys.stdout
#     # Redirect stdout to null
#     sys.stdout = open(os.devnull, "w")
#     try:
#         yield
#     finally:
#         # Restore original stdout
#         sys.stdout.close()
#         sys.stdout = original_stdout


# def compute_batch_metrics(
#     preds: torch.Tensor, target: torch.Tensor, metric_fn
# ) -> torch.Tensor:
#     """
#     Computes a metric for each batch and channel (if applicable) of predictions and targets using the provided metric function.

#     Args:
#         preds (torch.Tensor): The predicted values. Can be a 2D or 3D tensor.
#         target (torch.Tensor): The target values. Must have the same shape as preds.
#         metric_fn (callable): A function that computes the metric given a reference (target) and a degraded (predicted) tensor.

#     Returns:
#         torch.Tensor: A tensor containing the computed metric values for each batch (and channel if preds is 3D).
#     """
#     vals = []
#     for batch_idx in torch.arange(preds.shape[0]):
#         if preds.ndim == 3:
#             vals_batch_idx = []
#             for channel_idx in torch.arange(preds.shape[1]):
#                 vals_batch_idx.append(
#                     metric_fn(
#                         ref=target[batch_idx, channel_idx],
#                         deg=preds[batch_idx, channel_idx],
#                     )
#                 )
#             vals.append(torch.as_tensor(vals_batch_idx).mean())
#         else:
#             vals.append(
#                 metric_fn(
#                     ref=target[batch_idx],
#                     deg=preds[batch_idx],
#                 )
#             )
#     return torch.as_tensor(vals)
