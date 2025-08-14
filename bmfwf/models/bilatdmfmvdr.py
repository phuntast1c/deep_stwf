import math
from typing import Union

import torch

from . import BDMFMVDR, BaseLitModel

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class BilatDMFMVDR(BaseLitModel):
    """
    binaural deep MFMVDR filter
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        batch_size: int = 4,
        loss: str = "MagnitudeAbsoluteError",
        metrics_test: Union[
            tuple, str
        ] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,DNSMOS,SISDR",
        metrics_val: Union[tuple, str] = "",
        frame_length: int = 128,
        shift_length: int = 32,
        filter_length: int = 5,
        layer: int = 6,
        stack: int = 2,
        kernel: int = 3,
        hidden_dim: int = None,
        fs: int = 16000,
        num_channels: int = 1,
        minimum_gain: float = -20.0,
        window_type: str = "hann",
        interaural_rtf: str = "False",
        noise_stcm_left_and_right: bool = False,
        **kwargs,
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="BilatDMFMVDR",
            **kwargs,
        )
        self.frame_length = frame_length
        self.shift_length = shift_length
        self.filter_length = filter_length
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.hidden_dim = hidden_dim
        self.fs = fs
        self.num_channels = num_channels
        self.minimum_gain = minimum_gain
        self.window_type = window_type
        self.loss = loss
        self.interaural_rtf = interaural_rtf
        self.binaural = False
        self.noise_stcm_left_and_right = noise_stcm_left_and_right

        self.frequency_bins = self.frame_length // 2 + 1
        if self.omit_first_last_bin:
            self.frequency_bins -= 2

        self.model_left = BDMFMVDR(
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            frame_length=frame_length,
            shift_length=shift_length,
            filter_length=filter_length,
            layer=layer,
            stack=stack,
            kernel=kernel,
            hidden_dim=hidden_dim,
            fs=fs,
            num_channels=num_channels,
            minimum_gain=minimum_gain,
            window_type=window_type,
            interaural_rtf=interaural_rtf,
            binaural=False,
            noise_stcm_left_and_right=noise_stcm_left_and_right,
            **kwargs,
        )
        self.model_right = BDMFMVDR(
            learning_rate=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            frame_length=frame_length,
            shift_length=shift_length,
            filter_length=filter_length,
            layer=layer,
            stack=stack,
            kernel=kernel,
            hidden_dim=hidden_dim,
            fs=fs,
            num_channels=num_channels,
            minimum_gain=minimum_gain,
            window_type=window_type,
            interaural_rtf=interaural_rtf,
            binaural=False,
            noise_stcm_left_and_right=noise_stcm_left_and_right,
            **kwargs,
        )

        self.num_params = self.count_parameters()

        self.save_hyperparameters()

    def forward_(self, batch):
        # extract left and right batches
        batch_left, batch_right = {}, {}
        for key, val in batch.items():
            if val.shape[1] == 2 * self.num_channels:
                batch_left[key] = val[:, : self.num_channels, :]
                batch_right[key] = val[:, self.num_channels :, :]
            elif val.shape[1] == 2:
                batch_left[key] = val[:, :1, :]
                batch_right[key] = val[:, 1:, :]
            else:
                raise ValueError("Invalid shape")
        out = {}
        output_left = self.model_left.forward_(batch_left)
        output_right = self.model_right.forward_(batch_right)
        for key in output_left.keys():
            out[key] = torch.cat([output_left[key], output_right[key]], dim=1)
        return out
