"""
This module defines the PESQ (Perceptual Evaluation of Speech Quality) metric
and its variants.
"""

import torch
from pesq import pesq as pesq_
from pypesq import pesq as pesq_nb_raw

from stwf import utils

from . import BaseMetric


class BasePESQ(BaseMetric):
    """
    Base class for PESQ metrics.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *args,
        mode: str = "wb",
        fs: int = 16000,
        **kwargs,
    ):
        """
        Initializes the BasePESQ metric.

        Args:
            mode (str, optional): PESQ mode, "wb" (wide-band) or "nb" (narrow-band).
                                  Defaults to "wb".
            fs (int, optional): Sampling frequency. Defaults to 16000.
        """
        super().__init__(
            *args,
            requires_reference=True,
            requires_numpy=True,
            **kwargs,
        )
        assert fs in {8000, 16000}
        assert mode in {"wb", "nb"}

        self.fs = fs
        self.mode = mode

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the PESQ metric for a batch of signals.
        """
        return utils.compute_batch_metrics(preds, target, self._pesq_metric)

    def _pesq_metric(self, ref, deg):
        """
        Wrapper for the pesq library function.
        """
        return pesq_(fs=self.fs, ref=ref, deg=deg, mode=self.mode)


class PESQWB(BasePESQ):
    """
    Wide-band PESQ metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="wb", **kwargs)


class PESQNB(BasePESQ):
    """
    Narrow-band PESQ metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="nb", **kwargs)


class PESQNBRAW(BasePESQ):
    """
    Narrow-band PESQ metric using the pypesq library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="nb", **kwargs)

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the raw narrow-band PESQ metric for a batch of signals.
        """
        return utils.compute_batch_metrics(preds, target, self._pesq_nb_raw_metric)

    def _pesq_nb_raw_metric(self, ref, deg):
        """
        Wrapper for the pypesq library function.
        """
        return pesq_nb_raw(fs=self.fs, ref=ref, deg=deg)
