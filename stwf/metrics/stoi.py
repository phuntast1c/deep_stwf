"""
This module defines the STOI (Short-Time Objective Intelligibility) metric
and its extended version, ESTOI.
"""

import torch
from pystoi import stoi as stoi_

from stwf import utils

from . import BaseMetric


class BaseSTOI(BaseMetric):
    """
    Base class for STOI metrics.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        *args,
        extended: bool = False,
        fs: int = 16000,
        **kwargs,
    ):
        """
        Initializes the BaseSTOI metric.

        Args:
            extended (bool, optional): Whether to use the extended STOI (ESTOI).
                                       Defaults to False.
            fs (int, optional): Sampling frequency. Defaults to 16000.
        """
        super().__init__(
            *args,
            requires_reference=True,
            requires_numpy=True,
            **kwargs,
        )
        self.extended = extended
        self.fs = fs

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the STOI metric for a batch of signals.
        """
        return utils.compute_batch_metrics(preds, target, self._stoi_metric)

    def _stoi_metric(self, ref, deg):
        """
        Wrapper for the pystoi library function.
        """
        return stoi_(x=ref, y=deg, fs_sig=self.fs, extended=self.extended)


class STOI(BaseSTOI):
    """
    Short-Time Objective Intelligibility (STOI) metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, extended=False, **kwargs)


class ESTOI(BaseSTOI):
    """
    Extended Short-Time Objective Intelligibility (ESTOI) metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, extended=True, **kwargs)
