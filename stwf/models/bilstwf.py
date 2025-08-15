import math
from typing import Union

import torch

from . import BinSTWF, BaseLitModel

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class BilSTWF(BaseLitModel):
    """
    Bilateral Spatio-Temporal Wiener Filter (BilSTWF) model.

    This model applies a spatio-temporal Wiener filter to each channel (left and
    right) of a binaural audio signal independently. It uses two instances of the
    `BinSTWF` model to process the two channels.
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
        """
        Initializes the BilSTWF model.

        Args:
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            batch_size (int, optional): Batch size for training. Defaults to 4.
            loss (str, optional): Loss function to use. Defaults to "MagnitudeAbsoluteError".
            metrics_test (Union[tuple, str], optional): Metrics for testing.
            metrics_val (Union[tuple, str], optional): Metrics for validation.
            frame_length (int, optional): Length of each audio frame. Defaults to 128.
            shift_length (int, optional): Hop length between frames. Defaults to 32.
            filter_length (int, optional): Length of the Wiener filter. Defaults to 5.
            layer (int, optional): Number of layers in the TCN. Defaults to 6.
            stack (int, optional): Number of stacks in the TCN. Defaults to 2.
            kernel (int, optional): Kernel size in the TCN. Defaults to 3.
            hidden_dim (int, optional): Hidden dimension of the TCN. Defaults to None.
            fs (int, optional): Sampling frequency. Defaults to 16000.
            num_channels (int, optional): Number of input channels. Defaults to 1.
            minimum_gain (float, optional): Minimum gain to apply. Defaults to -20.0.
            window_type (str, optional): Windowing function to use. Defaults to "hann".
            interaural_rtf (str, optional): Whether to use interaural RTF. Defaults to "False".
            noise_stcm_left_and_right (bool, optional): Whether to use separate noise STCMs for left and right. Defaults to False.
        """
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
        self.frequency_bins -= 2

        self.model_left = BinSTWF(
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
        self.model_right = BinSTWF(
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
        """
        Forward pass of the BilSTWF model.

        It splits the input batch into left and right channels, processes them
        separately using the two `BinSTWF` models, and then concatenates the
        results.

        Args:
            batch (dict): A dictionary containing the input signals. The tensors
                          should have a channel dimension that can be split into
                          two halves (left and right).

        Returns:
            dict: A dictionary containing the processed output signals, with left
                  and right channels concatenated.
        """
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
