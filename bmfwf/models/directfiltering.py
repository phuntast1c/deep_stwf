import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import building_blocks as bb
from .. import utils
from . import BaseLitModel


EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class DirectFiltering(BaseLitModel):
    """
    binaural complex-valued filtering over multiple frames & microphones
    architecture inspired ConvTasNet
    algorithm inspired by Deep Filtering approach
    """

    def __init__(
        self,
        learning_rate: float = 0.0003,
        batch_size: int = 8,
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
        trunc_val: float = None,
        window_type: str = "hann",
        interaural_rtf: str = "False",
        **kwargs,
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            loss=loss,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="DirectFiltering",
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
        self.trunc_val = trunc_val
        self.window_type = window_type
        self.interaural_rtf = interaural_rtf

        self.binaural = True

        self.set_parameters()
        self.num_params = self.count_parameters()

        self.save_hyperparameters()

    def set_parameters(self):
        if self.interaural_rtf == "False":
            self.interaural_rtf = False
        else:
            self.interaural_rtf = self.interaural_rtf

        self.frequency_bins = int(self.frame_length / 2) - 1

        self.minimum_gain = utils.db2mag(self.minimum_gain)

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=self.window_type,
            sqrt=self.window_type == "hann",
        )

        input_size_factor = 3 * self.num_channels

        if self.hidden_dim is None:
            self.hidden_dim = int(((1 + self.binaural) * self.frequency_bins) / 4)
            print(f"setting hidden_dim to {self.hidden_dim}")

        self.tcn_params = {
            "hidden_dim": self.hidden_dim,
            "layer": self.layer,
            "stack": self.stack,
            "kernel": self.kernel,
        }
        # determine output size

        self.input_size = input_size_factor * (1 + self.binaural) * self.frequency_bins
        self.output_size = self.frequency_bins * self.filter_length * self.num_channels

        self.output_size *= 4  # spatio-temporal filter
        self.output_size *= 2  # complex-valued

        if self.interaural_rtf:
            if self.interaural_rtf == "global":
                self.output_size = (
                    self.frequency_bins
                    * 2  # complex
                    * (self.filter_length + 2 * self.num_channels - 1)
                )
            elif self.interaural_rtf == "ipsilateral":
                self.output_size = (
                    self.frequency_bins
                    * 2  # complex
                    * 2
                    * (2 * self.filter_length + self.num_channels - 1)
                )
            else:
                raise ValueError(f"unknown value interaural_rtf={self.interaural_rtf}")

        # handle standard values of trunc_val
        if self.trunc_val is None:
            self.trunc_vals = (-1.0, 1.0)
        elif self.trunc_val == 0.0:
            self.trunc_vals = None
        else:
            self.trunc_vals = (-self.trunc_val, self.trunc_val)

        self.estimator = bb.TCNEstimator(
            input_dim=self.input_size,
            output_dim=self.output_size,
            BN_dim=self.tcn_params["hidden_dim"],  # should be named BN_dim...
            hidden_dim=4 * self.tcn_params["hidden_dim"],
            stack=self.tcn_params["stack"],
            layer=self.tcn_params["layer"],
            kernel=self.tcn_params["kernel"],
        )

        self.batchnorm1d_noisy = nn.BatchNorm1d(  # used for feature normalization
            num_features=(1 + self.binaural) * self.num_channels * self.frequency_bins,
        )

        self.ref_indices = (
            self.filter_length - 1,
            self.filter_length * (self.num_channels + 1) - 1,
        )

        self.receptive_field = self.estimator.receptive_field

    def forward_(self, batch):
        noisy = batch["input"]
        num_samples = noisy.shape[-1]
        batch_size = noisy.shape[0]
        noisy = torch.stack([self.stft.get_stft(x) for x in noisy])

        # use (log) magnitude and phase spectra
        noisy_mag = torch.cat(
            [(noisy[:, x, 1:-1].abs() + EPS) for x in torch.arange(noisy.shape[1])],
            dim=1,
        )
        if self.use_log:
            noisy_mag = noisy_mag.log10()

        noisy_phase_cos = torch.cat(
            [noisy[:, x, 1:-1].angle().cos() for x in torch.arange(noisy.shape[1])],
            dim=1,
        )
        noisy_phase_sin = torch.cat(
            [noisy[:, x, 1:-1].angle().sin() for x in torch.arange(noisy.shape[1])],
            dim=1,
        )

        if self.use_batchnorm:
            # apply batch normalization before computing any further
            noisy_mag = self.batchnorm1d_noisy(noisy_mag)

        features_cat = torch.cat([noisy_mag, noisy_phase_cos, noisy_phase_sin], dim=1)

        # binaural multi-frame signal vector
        noisy_signalmodel = F.pad(noisy, pad=[self.filter_length - 1, 0]).unfold(
            dimension=-1, size=self.filter_length, step=1
        )
        # concatenate binaural signals
        noisy_signalmodel = torch.cat(
            [noisy_signalmodel[:, x] for x in torch.arange(noisy_signalmodel.shape[1])],
            dim=-1,
        )

        if self.trunc_vals is not None:
            scale = (self.trunc_vals[1] - self.trunc_vals[0]) / 2
            offset = self.trunc_vals[0] + scale
            filters = offset + scale * torch.tanh(self.estimator(features_cat))
        else:
            filters = self.estimator(features_cat)

        # vary filter shape according to mode
        reshape_size = self.get_reshape_size(batch_size)

        # permute_size = 0, 1, 4, 2, 3
        filters = filters.reshape(reshape_size)  # .permute(permute_size)
        filters = torch.view_as_complex(filters.transpose(-2, -1).contiguous())
        filters = filters.transpose(-2, -1)
        filters = self.apply_simplifications(filters)

        filters = filters.unsqueeze(-1)

        noisy_signalmodel = noisy_signalmodel.unsqueeze(1).unsqueeze(-1)

        # reintroduce first and last frequency bin (filtering with unity)
        filters_shape = list(filters.shape)
        filters_shape[-4] = 1
        filters = torch.cat(
            [
                filters.new_zeros(size=filters_shape),
                filters,
                filters.new_zeros(size=filters_shape),
            ],
            dim=-4,
        )

        # set ref indices to 1.0 for DC and Nyquist frequency bins
        filters[:, 0, 0, ..., self.ref_indices[0], :] = 1.0
        filters[:, 0, -1, ..., self.ref_indices[0], :] = 1.0
        if self.binaural:
            filters[
                :,
                1,
                0,
                ...,
                self.ref_indices[1],
                :,
            ] = 1.0
            filters[
                :,
                1,
                -1,
                ...,
                self.ref_indices[1],
                :,
            ] = 1.0

        filter_inputs = {"input": noisy_signalmodel}

        output = {}
        for key, val in filter_inputs.items():
            if self.training:
                output[key + "_proc_stft"] = (utils.hermitian(filters) @ val)[..., 0, 0]
            else:
                if key == "input":
                    (
                        output[key + "_proc_stft"],
                        mask,
                    ) = utils.filter_minimum_gain(
                        filters,
                        val,
                        self.minimum_gain,
                        binaural=self.binaural,
                        ref_channels=self.ref_indices,
                    )
                else:
                    (
                        output[key + "_proc_stft"],
                        mask,
                    ) = utils.filter_minimum_gain(
                        filters,
                        val,
                        self.minimum_gain,
                        binaural=self.binaural,
                        ref_channels=self.ref_indices,
                        mask=mask,
                    )

        tmp = {}
        for key, val in output.items():
            tmp[key.replace("_stft", "")] = self.stft.get_istft(val, length=num_samples)
        output = {**output, **tmp}

        return output

    def apply_simplifications(self, filters):
        if self.interaural_rtf == "global":
            interaural_rtf, filters = filters[:, 0].tensor_split(
                (2 * self.num_channels - 1,), dim=-1
            )
            filters = torch.cat(
                [filters]
                + [
                    filters * interaural_rtf[..., None, x]
                    for x in torch.arange(interaural_rtf.shape[-1])
                ],
                dim=-1,
            )

            filters = torch.stack(
                [
                    filters,
                    filters / interaural_rtf[..., None, -self.num_channels],
                ],
                dim=1,
            )

        elif self.interaural_rtf == "ipsilateral":
            interaural_rtf, filters = filters[:, 0].tensor_split(
                (2 * (self.num_channels - 1),), dim=-1
            )
            filters = filters.unfold(
                dimension=-1,
                size=self.filter_length - 1,
                step=self.filter_length - 1,
            ).transpose(-2, -1)
            filters = torch.stack(
                [
                    torch.cat(
                        [filters[..., 0, :]]
                        + [
                            filters[..., 0, :] * interaural_rtf[..., x : x + 1]
                            for x in torch.arange(self.num_channels - 1)
                        ]
                        + [filters[..., 1, :]]
                        + [
                            filters[..., 1, :] * interaural_rtf[..., x : x + 1]
                            for x in torch.arange(
                                self.num_channels - 1,
                                2 * (self.num_channels - 1),
                            )
                        ],
                        dim=-1,
                    ),
                    torch.cat(
                        [filters[..., 2, :]]
                        + [
                            filters[..., 2, :] * interaural_rtf[..., x : x + 1]
                            for x in torch.arange(self.num_channels - 1)
                        ]
                        + [filters[..., 3, :]]
                        + [
                            filters[..., 3, :] * interaural_rtf[..., x : x + 1]
                            for x in torch.arange(
                                self.num_channels - 1,
                                2 * (self.num_channels - 1),
                            )
                        ],
                        dim=-1,
                    ),
                ],
                dim=1,
            )

        return filters

    def get_reshape_size(self, batch_size):
        if self.interaural_rtf == "global":
            reshape_size = (
                batch_size,
                1,
                self.frequency_bins,
                self.filter_length + 2 * self.num_channels - 1,
                2,  # complex values
                -1,
            )
        elif self.interaural_rtf == "ipsilateral":
            if self.mode == "spatio_temporal":
                reshape_size = (
                    batch_size,
                    1,
                    self.frequency_bins,
                    2 * (2 * self.filter_length + self.num_channels - 1),
                    2,  # complex values
                    -1,
                )
        else:
            reshape_size = (
                batch_size,
                2,
                self.frequency_bins,
                self.filter_length * self.num_channels * 2,
                1 + self.complex_valued,
                -1,
            )

        return reshape_size
