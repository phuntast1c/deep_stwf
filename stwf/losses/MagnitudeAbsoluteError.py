"""
This module defines the Magnitude Absolute Error loss function.
"""

import torch

from . import BaseSELoss

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class MagnitudeAbsoluteError(BaseSELoss):
    """
    Magnitude Absolute Error loss function.

    This loss can be a combination of the magnitude absolute error and the
    complex absolute error in the STFT domain.
    """

    def __init__(
        self,
        frame_length=512,
        overlap_length=None,
        window_fn=torch.hann_window,
        sqrt=True,
        use_mask=False,
        beta: float = 0.4,
        kind: str = "combined",
        **kwargs,
    ):
        """
        Initializes the MagnitudeAbsoluteError loss.

        Args:
            frame_length (int, optional): Frame length for STFT. Defaults to 512.
            overlap_length (int, optional): Overlap length for STFT. Defaults to None.
            window_fn (function, optional): Window function for STFT. Defaults to torch.hann_window.
            sqrt (bool, optional): Whether to use the square root of the window. Defaults to True.
            use_mask (bool, optional): Whether to use a mask. Defaults to False.
            beta (float, optional): Weight for the complex loss component. Defaults to 0.4.
            kind (str, optional): Type of loss to compute. Can be "combined",
                                  "complex", or "magnitude". Defaults to "combined".
        """
        self.kind = kind
        self.beta = beta
        self.overlap_length = (
            int(0.5 * frame_length) if overlap_length is None else overlap_length
        )

        super().__init__(
            use_stft=True,
            frame_length=frame_length,
            overlap_length=self.overlap_length,
            use_mask=use_mask,
            beta=beta,
            kind=kind,
            window_fn=window_fn,
            sqrt=sqrt,
        )

    def get_loss(self, target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """
        Computes the Magnitude Absolute Error loss.

        Args:
            target (torch.Tensor): The target STFT.
            estimate (torch.Tensor): The estimated STFT.

        Returns:
            torch.Tensor: The computed loss.
        """
        loss_magnitude = (estimate.abs() - target.abs()).abs().mean()
        loss_complex = (estimate - target).abs().mean()

        if self.kind == "combined":
            loss = self.beta * loss_complex + (1.0 - self.beta) * loss_magnitude
        elif self.kind == "complex":
            loss = loss_complex
        elif self.kind == "magnitude":
            loss = loss_magnitude
        else:
            raise ValueError(f"unknown loss kind {self.kind}!")
        return loss
