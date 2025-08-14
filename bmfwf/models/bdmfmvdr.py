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


class BDMFMVDR(BaseLitModel):
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
            model_name="BDMFMVDR",
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
        self.noise_stcm_left_and_right = noise_stcm_left_and_right

        self.frequency_bins = self.frame_length // 2 + 1
        self.frequency_bins -= 2

        self.binaural = True
        self.normalize_gamma = False

        if self.hidden_dim is None:
            self.hidden_dim = int(((1 + self.binaural) * self.frequency_bins) / 4)
            print(f"setting hidden_dim to {self.hidden_dim}")

        self.dnn_params = {
            "hidden_dim": self.hidden_dim,
            "layer": self.layer,
            "stack": self.stack,
            "kernel": self.kernel,
        }

        self.set_parameters()
        self.num_params = self.count_parameters()

        self.save_hyperparameters()

    def set_parameters(self):
        self.minimum_gain = utils.db2mag(self.minimum_gain)

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=self.window_type,
            sqrt=self.window_type == "hann",
        )

        # correlation matrix estimators
        if self.feature_representation == "mag_phase":
            input_size_factor = 3 * self.num_channels

        if self.hidden_dim is None:
            self.hidden_dim = int(((1 + self.binaural) * self.frequency_bins) / 4)

        self.output_size_phi = (
            self.filter_length * (1 + self.binaural) * self.num_channels
        ) ** 2 * self.frequency_bins

        if self.noise_stcm_left_and_right:
            self.output_size_phi *= 2

        self.Phin_estimator = bb.TCNEstimator(
            input_dim=input_size_factor * (1 + self.binaural) * self.frequency_bins,
            output_dim=self.output_size_phi,
            BN_dim=self.dnn_params["hidden_dim"],
            hidden_dim=4 * self.dnn_params["hidden_dim"],
            layer=int(self.layer),
            stack=int(self.stack),
            kernel=int(self.kernel),
        )

        # speech IFC / a-priori SNR estimators
        self.independent_speech_psd_estimates = (
            self.binaural and self.interaural_rtf != "global"
        )

        if self.interaural_rtf == "global":
            self.output_size_gamma = (
                (
                    self.filter_length
                    - (not self.normalize_gamma)
                    + (1 + self.binaural) * self.num_channels
                    - 1
                )
                * 2
                + self.use_mwf * (1 + self.independent_speech_psd_estimates)
            ) * self.frequency_bins
        elif self.interaural_rtf == "ipsilateral":
            self.output_size_gamma = (
                (
                    2
                    * (1 + self.binaural)
                    * (self.filter_length - (not self.normalize_gamma))
                    + (1 + self.binaural) * (self.num_channels - 1)
                )
                * 2
                + self.use_mwf * (1 + self.independent_speech_psd_estimates)
            ) * self.frequency_bins
        else:  # none
            self.output_size_gamma = (
                (
                    (1 + self.binaural) ** 2
                    * self.num_channels
                    * (self.filter_length - (not self.normalize_gamma))
                )
                * 2
                + self.use_mwf * (1 + self.independent_speech_psd_estimates)
            ) * self.frequency_bins

        self.speech_ifc_estimator = bb.TCNEstimator(
            input_dim=input_size_factor * (1 + self.binaural) * self.frequency_bins,
            output_dim=self.output_size_gamma,
            BN_dim=self.dnn_params["hidden_dim"],
            hidden_dim=4 * self.dnn_params["hidden_dim"],
            layer=int(self.layer),
            stack=int(self.stack),
            kernel=int(self.kernel),
        )

        self.batchnorm1d_noisy = nn.BatchNorm1d(  # used for feature normalization
            num_features=(1 + self.binaural) * self.num_channels * self.frequency_bins,
        )

        self.receptive_field = self.Phin_estimator.receptive_field

        self.ref_indices = (
            self.filter_length - 1,
            self.filter_length * (self.num_channels + 1) - 1,
        )

    def forward_(self, batch):
        (
            noisy,
            num_samples,
            noise,
            speech,
            noisy_mag,
            features_cat,
        ) = self.get_features(batch)

        batch_size = noisy.shape[0]

        # construct multi-frame vectors for noisy, noise and speech
        noise, speech, noisy_signalmodel = self.get_multiframe_vectors(
            noise, speech, noisy
        )

        # Phi estimation variants
        correlation_noise = self.estimate_matrix_quantities(features_cat, batch_size)

        gammax, speech_psd_estimate, interaural_rtf = self.estimate_vector_quantities(
            features_cat,
            batch_size,
            noisy_signalmodel,
        )

        if not self.noise_stcm_left_and_right:
            correlation_noise = correlation_noise.unsqueeze(1)

        noisy_signalmodel = noisy_signalmodel.unsqueeze(1).unsqueeze(-1)

        filters, correlation_noise_inv_gammax = self.get_mvdr(
            gammax,
            correlation_noise,
            regularize=False,
        )

        if self.interaural_rtf == "global" and self.binaural:
            filters = self.get_filters_global(interaural_rtf, filters)

        postfilter = self.get_postfilter(
            speech_psd_estimate,
            gammax,
            correlation_noise,
            correlation_noise_inv_gammax=correlation_noise_inv_gammax,
            regularize=False,
        )
        filters = filters * postfilter

        # reintroduce first and last frequency bin (filtering with identity)
        filters = self.modify_filters_omit_first_last_bin(filters)

        # filtering and retransforming
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

            if output[key + "_proc_stft"].ndim == 3:  # no channel dim
                output[key + "_proc_stft"] = output[key + "_proc_stft"].unsqueeze(1)

        tmp = {}
        for key, val in output.items():
            tmp[key.replace("_stft", "")] = self.stft.get_istft(val, length=num_samples)
        output = {**output, **tmp}

        return output

    def estimate_vector_quantities(
        self,
        features_cat,
        batch_size,
        noisy_signalmodel,
    ):
        gammax, speech_psd_estimate, interaural_rtf = None, None, None, None

        gammax, speech_psd_estimate, interaural_rtf = self.get_gammax_direct(
            features_cat, batch_size, noisy_signalmodel
        )

        return gammax, speech_psd_estimate, interaural_rtf

    def estimate_matrix_quantities(self, features_cat, batch_size):
        correlation_noise = self.get_cov_matrices_hermitian_psd(
            features_cat, batch_size
        )
        return correlation_noise

    def modify_filters_omit_first_last_bin(self, filters):
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
        return filters

    def get_filters_global(self, interaural_rtf, filters):
        return torch.cat(
            [
                filters,
                interaural_rtf[..., -self.num_channels][:, None, ..., None, None].conj()
                * filters,
            ],
            dim=1,
        )

    def get_gammax_direct(self, features_cat, batch_size, noisy_signalmodel):
        gammax = self.speech_ifc_estimator(features_cat)

        (
            gammax,
            speech_psd_estimate,
        ) = self.separate_gammax_into_gammax_and_speech_psds(
            batch_size, noisy_signalmodel, gammax
        )
        gammax = self.reshape_gammax_direct(batch_size, gammax)

        # normalize w.r.t. left / right current frame
        (
            gammax,
            speech_psd_estimate,
            interaural_rtf,
        ) = self.gammax_direct_simplifications_spatiotemporal(
            gammax, speech_psd_estimate
        )

        gammax = gammax.unsqueeze(-1)
        return gammax, speech_psd_estimate, interaural_rtf

    def gammax_direct_simplifications_spatiotemporal(self, gammax, speech_psd_estimate):
        if self.interaural_rtf == "global":
            return self.gammax_direct_spatiotemporal_global(gammax, speech_psd_estimate)
        elif self.interaural_rtf == "ipsilateral":
            return self.gammax_direct_spatiotemporal_ipsilateral(
                gammax, speech_psd_estimate
            )
        else:
            return self.gammax_direct_spatiotemporal_none(gammax, speech_psd_estimate)

    def gammax_direct_spatiotemporal_none(self, gammax, speech_psd_estimate):
        gammax = gammax.unfold(
            dimension=-1,
            size=self.filter_length - 1,
            step=self.filter_length - 1,
        )
        shape_ones = list(gammax.shape)
        shape_ones[-1] = 1
        gammax = torch.cat([gammax, gammax.new_ones(size=shape_ones)], dim=-1)
        gammax = gammax.reshape(gammax.shape[:-2] + (-1,))

        if self.binaural:
            gammax = torch.cat(
                [
                    gammax[
                        ...,
                        : (1 + self.binaural)
                        * (self.num_channels * self.filter_length),
                    ],
                    gammax[
                        ...,
                        (1 + self.binaural)
                        * (self.num_channels * self.filter_length) :,
                    ],
                ],
                dim=1,
            )

        return gammax, speech_psd_estimate, None

    def gammax_direct_spatiotemporal_ipsilateral(self, gammax, speech_psd_estimate):
        interaural_rtf, gammax = gammax[:, 0].tensor_split(
            (2 * (self.num_channels - 1),), dim=-1
        )

        gammax = gammax.unfold(
            dimension=-1,
            size=self.filter_length - 1,
            step=self.filter_length - 1,
        ).transpose(-2, -1)

        shape_ones = list(gammax.shape)
        shape_ones[-1] = 1
        gammax = torch.cat([gammax, gammax.new_ones(size=shape_ones)], dim=-1)
        gammax = torch.stack(
            [
                torch.cat(
                    [gammax[..., 0, :]]
                    + [
                        gammax[..., 0, :] * interaural_rtf[..., x : x + 1]
                        for x in torch.arange(self.num_channels - 1)
                    ]
                    + [gammax[..., 1, :]]
                    + [
                        gammax[..., 1, :] * interaural_rtf[..., x : x + 1]
                        for x in torch.arange(
                            self.num_channels - 1,
                            2 * (self.num_channels - 1),
                        )
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [gammax[..., 2, :]]
                    + [
                        gammax[..., 2, :] * interaural_rtf[..., x : x + 1]
                        for x in torch.arange(self.num_channels - 1)
                    ]
                    + [gammax[..., 3, :]]
                    + [
                        gammax[..., 3, :] * interaural_rtf[..., x : x + 1]
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

        return gammax, speech_psd_estimate, interaural_rtf

    def gammax_direct_spatiotemporal_global(self, gammax, speech_psd_estimate):
        interaural_rtf, gammax = gammax[:, 0].tensor_split(
            ((1 + self.binaural) * self.num_channels - 1,), dim=-1
        )

        speech_psd_estimate = torch.cat(
            [
                speech_psd_estimate,
                interaural_rtf[
                    ...,
                    None,
                    -(self.binaural * self.num_channels),
                ]
                .abs()
                .pow(2)
                * speech_psd_estimate,
            ],
            dim=-1,
        )

        shape_ones = list(gammax.shape)
        shape_ones[-1] = 1
        gammax = torch.cat([gammax, gammax.new_ones(size=shape_ones)], dim=-1)
        gammax = torch.cat(
            [gammax]
            + [
                gammax * interaural_rtf[..., None, x]
                for x in torch.arange(interaural_rtf.shape[-1])
            ],
            dim=-1,
        ).unsqueeze(1)

        return gammax, speech_psd_estimate, interaural_rtf

    def reshape_gammax_direct(self, batch_size, gammax):
        reshape_size = (
            batch_size,
            1,
            self.frequency_bins,
            int(
                (
                    self.output_size_gamma
                    - self.use_mwf * (1 + self.independent_speech_psd_estimates)
                )
                / (self.frequency_bins * 2)
            ),
            2,  # complex values
            -1,
        )
        gammax = gammax.reshape(reshape_size)
        gammax = torch.view_as_complex(gammax.transpose(-2, -1).contiguous()).transpose(
            -2, -1
        )

        return gammax  # (B, 1, F, T, N)

    def separate_gammax_into_gammax_and_speech_psds(
        self, batch_size, noisy_signalmodel, gammax
    ):
        speech_psd_estimate = (
            F.softplus(
                gammax[
                    ...,
                    -(1 + self.independent_speech_psd_estimates)
                    * self.frequency_bins :,
                    :,
                ]
            )
            .reshape(
                batch_size,
                self.frequency_bins,
                (1 + self.independent_speech_psd_estimates),
                -1,
            )
            .transpose(-2, -1)
        )

        if self.independent_speech_psd_estimates:
            ref_indices = self.ref_indices
        else:
            ref_indices = (self.ref_indices[0],)

        speech_psd_estimate = (
            speech_psd_estimate
            * noisy_signalmodel[
                :,
                1:-1,
                ...,
                ref_indices,
            ].abs()
        ).pow(2)

        gammax = gammax[
            ...,
            : -(1 + self.independent_speech_psd_estimates) * self.frequency_bins,
            :,
        ]

        return gammax, speech_psd_estimate

    def get_cov_matrices_hermitian_psd(self, features_cat, batch_size):
        view_shape = (
            batch_size,
            self.frequency_bins,
            int(
                self.output_size_phi
                / self.frequency_bins
                / (1 + self.noise_stcm_left_and_right)
            ),
            -1,
        )
        if self.noise_stcm_left_and_right:
            view_shape = (view_shape[0],) + (2,) + view_shape[1:]
        correlation_noise = self.Phin_estimator(features_cat)
        correlation_noise = correlation_noise.reshape(view_shape).transpose(-1, -2)
        if self.noise_stcm_left_and_right:
            correlation_noise = torch.stack(
                [
                    utils.vector_to_Hermitian(
                        correlation_noise[:, 0],
                    ),
                    utils.vector_to_Hermitian(
                        correlation_noise[:, 1],
                    ),
                ],
                dim=1,
            )
        else:
            correlation_noise = utils.vector_to_Hermitian(
                correlation_noise,
            )

        return correlation_noise

    def get_multiframe_vectors(self, noise, speech, noisy):
        noisy_signalmodel = F.pad(noisy, pad=[self.filter_length - 1, 0]).unfold(
            dimension=-1, size=self.filter_length, step=1
        )
        noisy_signalmodel = torch.cat(
            [noisy_signalmodel[:, x] for x in torch.arange(noisy_signalmodel.shape[1])],
            dim=-1,
        )

        return noise, speech, noisy_signalmodel

    def get_features(self, batch):
        # channels are ordered as L, L, ..., R, R, ... if binaural
        noisy = batch["input"][:, : (1 + self.binaural) * self.num_channels]
        num_samples = noisy.shape[-1]
        noisy = torch.stack([self.stft.get_stft(x) for x in noisy])

        # use (log) magnitude and phase spectra
        noisy_mag, features_cat = self.get_features_mag_phase(noisy)

        return noisy, num_samples, noisy_mag, features_cat

    def get_features_mag_phase(self, noisy):
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

        noisy_mag = self.batchnorm1d_noisy(noisy_mag)

        features_cat = torch.cat([noisy_mag, noisy_phase_cos, noisy_phase_sin], dim=1)

        return noisy_mag, features_cat

    def get_mvdr(self, gammax, Phi, regularize=False):
        if regularize:
            Phi = utils.tik_reg(Phi, self.reg)
        b = torch.linalg.solve(Phi, gammax) if self.compute_inverse else Phi @ gammax
        return b / (gammax.mH @ b + EPS).real, b

    def get_postfilter(
        self,
        speech_psd_estimate,
        gammax,
        correlation_noise,
        correlation_noise_inv_gammax,
        regularize=False,
    ):
        nom = speech_psd_estimate.permute(0, 3, 1, 2)[..., None, None]
        if regularize:
            correlation_noise = utils.tik_reg(correlation_noise, self.reg)
        denom = nom + 1.0 / ((gammax.mH @ correlation_noise_inv_gammax).abs() + EPS)
        return nom / denom
