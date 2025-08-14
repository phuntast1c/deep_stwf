from logging import getLogger

import lightning as pl
import torch

logger = getLogger(__name__)

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class DatasetDNSBinaural(torch.utils.data.Dataset):
    """
    please implement yourself as I cannot share the data
    """

    def __init__(self, *args, **kwargs):
        super().__init__()


class DataModuleDNSBinaural(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
