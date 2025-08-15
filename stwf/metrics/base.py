"""
This module defines the base class for all metrics used in this project.
"""

import pandas as pd
import torch
import torchmetrics
from torch import tensor

from ..utils import dcn


class BaseMetric(torchmetrics.Metric):
    """
    Base class for speech enhancement metrics.

    This class provides a common interface for all metrics, inheriting from
    `torchmetrics.Metric`. It handles state management, dataframe logging, and
    common requirements for metrics, such as whether they need a reference signal
    or numpy arrays.
    """

    def __init__(
        self,
        *args,
        requires_reference: bool = True,
        requires_numpy: bool = True,
        **kwargs,
    ):
        """
        Initializes the BaseMetric.

        Args:
            requires_reference (bool, optional): Whether the metric requires a
                                                 reference (clean) signal.
                                                 Defaults to True.
            requires_numpy (bool, optional): Whether the metric requires numpy
                                             arrays as input. Defaults to True.
        """
        super().__init__(
            *args,
            compute_on_cpu=True,
            # **kwargs,
        )
        self.requires_reference = requires_reference
        self.requires_numpy = requires_numpy
        self.add_state("enhanced_total", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", default=tensor(0), dist_reduce_fx="sum")
        self.dataframe = pd.DataFrame()

    def _get_values(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for computing the metric values.

        This method must be implemented by subclasses.

        Args:
            preds (torch.Tensor): The predicted (enhanced) signal.
            target (torch.Tensor): The target (clean) signal.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def update(
        self,
        inputs: torch.Tensor,
        preds: torch.Tensor,
        target: torch.Tensor,
        meta: dict,
        dataloader_idx: int,
    ) -> None:
        """
        Updates the metric state with new predictions.

        Args:
            inputs (torch.Tensor): The input (noisy) signal.
            preds (torch.Tensor): The predicted (enhanced) signal.
            target (torch.Tensor): The target (clean) signal.
            meta (dict): Metadata for the current batch.
            dataloader_idx (int): Index of the current dataloader.
        """
        if self.requires_reference and target is None:
            enhanced = torch.as_tensor(torch.nan)
        elif self.requires_numpy:
            enhanced = self._get_values(dcn(preds), dcn(target))
        else:
            enhanced = self._get_values(preds, target)

        self.enhanced_total = (self.enhanced_total + enhanced.sum()).to(
            device=inputs.device
        )
        self.numel = (self.numel + enhanced.numel()).to(device=inputs.device)

        self.update_dataframe(meta, dataloader_idx, enhanced)

    def update_dataframe(self, meta, dataloader_idx, enhanced):
        """
        Updates the internal dataframe with the results for the current batch.
        """
        # put in meta data that is no tensor
        meta_no_tensor = [
            {k: v for k, v in m.items() if not torch.is_tensor(v)} for m in meta
        ]
        dataframe_current = pd.concat(
            [
                pd.DataFrame(
                    {
                        "dataloader_idx": dataloader_idx,
                        self.__class__.__name__: dcn(enhanced[idx]),
                        # **meta_no_tensor[idx],
                    },
                    index=[meta_no_tensor[idx]["filename"]],
                )
                for idx in range(len(meta))
            ]
        )
        # combine all dataframes
        self.dataframe = pd.concat([self.dataframe, dataframe_current])

    def compute(self):
        """
        Computes the final metric value from the state.
        """
        # move to cuda required due to lightning and torchmetrics quirk...
        return {
            self.__class__.__name__: (self.enhanced_total.float() / self.numel).to(
                "cuda"
            ),
        }
