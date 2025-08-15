import datetime
import math
import os

import human_readable_ids as hri
import numpy as np
import torch
from torch.nn import functional as F

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


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


def db2mag(fdB: float) -> float:
    """converts dB to magnitude"""
    return 10 ** (fdB / 20)


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


def hermitian(X: torch.Tensor) -> torch.Tensor:
    return X.mH  # now with native support (which should do exactly the same...)


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
