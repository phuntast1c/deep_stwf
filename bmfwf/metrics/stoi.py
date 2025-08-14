import torch
from pystoi import stoi as stoi_

from bmfwf import utils

from . import BaseMetric


class BaseSTOI(BaseMetric):
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
        super().__init__(
            *args,
            requires_reference=True,
            requires_numpy=True,
            **kwargs,
        )
        self.extended = extended
        self.fs = fs

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return utils.compute_batch_metrics(preds, target, self._stoi_metric)

    def _stoi_metric(self, ref, deg):
        return stoi_(x=ref, y=deg, fs_sig=self.fs, extended=self.extended)


class STOI(BaseSTOI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, extended=False, **kwargs)


class ESTOI(BaseSTOI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, extended=True, **kwargs)
