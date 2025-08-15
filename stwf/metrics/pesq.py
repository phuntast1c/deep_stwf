import torch
from pesq import pesq as pesq_
from pypesq import pesq as pesq_nb_raw

from stwf import utils

from . import BaseMetric


class BasePESQ(BaseMetric):
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
        return utils.compute_batch_metrics(preds, target, self._pesq_metric)

    def _pesq_metric(self, ref, deg):
        return pesq_(fs=self.fs, ref=ref, deg=deg, mode=self.mode)


class PESQWB(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="wb", **kwargs)


class PESQNB(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="nb", **kwargs)


class PESQNBRAW(BasePESQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode="nb", **kwargs)

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return utils.compute_batch_metrics(preds, target, self._pesq_nb_raw_metric)

    def _pesq_nb_raw_metric(self, ref, deg):
        return pesq_nb_raw(fs=self.fs, ref=ref, deg=deg)
