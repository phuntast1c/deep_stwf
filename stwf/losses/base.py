"""
This module defines the base class for all loss functions used in this project.
"""

import torch

from .. import utils

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class BaseSELoss(torch.nn.Module):
    """
    Base class for speech enhancement loss functions.

    This class provides a common interface for all loss functions. It handles
    optional STFT transformation and multichannel audio processing.
    """

    def __init__(
        self,
        use_stft: bool = False,
        multichannel_handling: str = "average",
        **kwargs,
    ) -> None:
        """
        Initializes the BaseSELoss.

        Args:
            use_stft (bool, optional): Whether to apply STFT to the signals
                                       before computing the loss. Defaults to False.
            multichannel_handling (str, optional): How to handle multichannel
                                                   audio. Can be "average", "cat",
                                                   or "pass". Defaults to "average".
            **kwargs: Additional arguments for the STFT.
        """
        super().__init__()
        self.use_stft = use_stft
        self.multichannel_handling = multichannel_handling
        self.kwargs = kwargs

        if self.use_stft:
            self.stft = utils.STFTTorch(
                frame_length=self.kwargs["frame_length"],
                overlap_length=self.kwargs["overlap_length"],
                window=self.kwargs["window_fn"],
                sqrt=self.kwargs["sqrt"],
            )

    def forward(self, outputs: dict, batch: dict, meta: dict = None):
        """
        Computes the loss.

        Args:
            outputs (dict): The model's output, containing the enhanced signal.
            batch (dict): The input batch, containing the target signal.
            meta (dict, optional): Metadata. Defaults to None.

        Returns:
            dict: A dictionary containing the computed loss.
        """
        target = batch["target"]
        estimate = outputs["input_proc"]

        assert target.ndim <= 3
        multichannel = target.ndim == 3  # (B x M x T)

        if self.use_stft:
            if multichannel:
                target = torch.stack([self.stft.get_stft(x) for x in target], dim=0)
                estimate = torch.stack([self.stft.get_stft(x) for x in estimate], dim=0)
            else:
                target = self.stft.get_stft(target)
                estimate = self.stft.get_stft(estimate)

        if multichannel:
            if self.multichannel_handling == "cat":
                # concatenate channels temporally
                target = torch.cat(
                    [target[:, idx] for idx in torch.arange(target.shape[1])], dim=-1
                )
                estimate = torch.cat(
                    [estimate[:, idx] for idx in torch.arange(estimate.shape[1])],
                    dim=-1,
                )
            elif self.multichannel_handling == "pass":
                pass
            elif self.multichannel_handling != "average":
                raise ValueError(
                    f"unknown multichannel handling type {self.multichannel_handling}!"
                )

        assert target.shape == estimate.shape

        return {"loss": self.get_loss(target, estimate)}

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for computing the loss.

        This method must be implemented by subclasses.

        Args:
            target (torch.Tensor): The target signal.
            estimate (torch.Tensor): The estimated (enhanced) signal.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
