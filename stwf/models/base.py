"""
This module defines the base class for all PyTorch Lightning models in this project.
"""

import contextlib
import os
from abc import abstractmethod
from typing import Any, Sequence, Union

import lightning as pl
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks.callback import Callback
from torchmetrics import MetricCollection

from .. import losses, utils
from .. import metrics as metrics


class BaseLitModel(pl.LightningModule):
    """
    Base class for PyTorch Lightning models.
    It handles model initialization, optimizer configuration, training, validation,
    and testing steps. It also includes methods for logging metrics and audio
    samples to Weights & Biases (wandb).
    """

    def __init__(
        self,
        lr: float,
        batch_size: int,
        loss: str,
        metrics_test: Union[tuple, str],
        metrics_val: Union[tuple, str],
        model_name: str,
        my_optimizer: str = "AdamW",
        my_lr_scheduler: str = "ReduceLROnPlateau",
        compute_complexity_metrics: bool = True,
        save_target: bool = False,
        **kwargs,
    ):
        """
        Initializes the BaseLitModel.

        Args:
            lr (float): Learning rate.
            batch_size (int): Batch size.
            loss (str): Name of the loss function to use.
            metrics_test (Union[tuple, str]): Metrics to use for testing.
            metrics_val (Union[tuple, str]): Metrics to use for validation.
            model_name (str): Name of the model.
            my_optimizer (str, optional): Optimizer to use. Defaults to "AdamW".
            my_lr_scheduler (str, optional): Learning rate scheduler. Defaults to "ReduceLROnPlateau".
            compute_complexity_metrics (bool, optional): Whether to compute complexity metrics. Defaults to True.
            save_target (bool, optional): Whether to save target audio during testing. Defaults to False.
        """
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.loss = loss
        self.model_name = model_name
        self.my_optimizer = my_optimizer
        self.my_lr_scheduler = my_lr_scheduler
        self.compute_complexity_metrics = compute_complexity_metrics
        self.save_target = save_target

        self.nan_batch_counter = 0.0

        self.criterion = getattr(losses, self.loss)(**kwargs)
        if isinstance(metrics_test, str):
            metrics_test = metrics_test.split(",")
        if isinstance(metrics_val, str):
            metrics_val = metrics_val.split(",")
        self.metrics_test = metrics_test
        self.metrics_val = metrics_val
        self.metric_collections = {
            "test": {},
            "val": {},
        }

        self.test_outputs = []

    @abstractmethod
    def forward_(self, x):
        """
        Abstract forward method to be implemented by subclasses.
        This method should contain the core model logic.
        """
        raise NotImplementedError

    def forward(self, x: dict) -> dict:
        """
        Main forward pass for the model.
        """
        return self.forward_(x)

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.
        """
        if self.my_optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.my_optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError

        if self.my_lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
        elif self.my_lr_scheduler == "OneCycleLR":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                    verbose=False,
                ),
                "interval": "step",
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    # skip batches including NaN gradients
    def on_after_backward(self) -> None:
        """
        Handles NaN gradients after the backward pass.
        It logs the number of batches with NaN gradients.
        """
        increase_nan_batch_counter = False
        for param in self.parameters():
            if param.grad is not None:
                nan_grads = torch.isnan(param.grad)
                if torch.any(nan_grads):
                    param.grad[nan_grads] = 0.0
                    increase_nan_batch_counter = True
        if increase_nan_batch_counter:
            self.nan_batch_counter += 1

        self.log(
            "ptl/nan_batch_counter",
            self.nan_batch_counter,
            batch_size=self.batch_size,
        )
        return super().on_after_backward()

    def training_step(self, batch, idx):
        """
        Performs a single training step.
        """
        output = self(batch.signals)
        loss = self.criterion(output, batch.signals, batch.meta)
        self.log_dict(
            {f"train/{x}": y for x, y in loss.items()},
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
        )
        if (
            idx == 0
            and self.trainer.current_epoch % self.trainer.log_every_n_steps == 0
        ):
            with contextlib.suppress(KeyError):
                self.log_wave(batch, idx, output, "train")
        return {"loss": loss["loss"]}

    def load_state_dict(self, state_dict: os.Mapping[str, Any], strict: bool = True):
        """
        Loads the model's state dict.
        """
        return super().load_state_dict(state_dict, strict)

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """
        Configures callbacks for the model.
        """
        return super().configure_callbacks()

    def _get_metric_collection(self, stage, dataloader_idx):
        """
        Gets the metric collection for a given stage and dataloader.
        """
        if dataloader_idx not in self.metric_collections[stage]:
            metrics_ = self.metrics_test if stage == "test" else self.metrics_val
            self.metric_collections[stage][dataloader_idx] = MetricCollection(
                [getattr(metrics, met)() for met in metrics_ if met != ""]
            )
        return self.metric_collections[stage][dataloader_idx]

    def validation_step(self, batch, idx, dataloader_idx: int = 0):
        """
        Performs a single validation step.
        """
        output = self(batch.signals)
        loss = self.criterion(output, batch.signals, batch.meta)

        self.log_dict(
            {f"val/{x}": y for x, y in loss.items()},
            reduce_fx="mean",
            batch_size=self.batch_size,
            prog_bar=False,
            sync_dist=True,
        )

        metrics_dict = {}
        metric_collection = self._get_metric_collection("val", dataloader_idx)
        for metric in metric_collection:
            (
                metrics_dict[f"metrics/val/enh_{metric.__class__.__name__.upper()}"],
                metrics_dict[f"metrics/val/{metric.__class__.__name__.upper()}"],
            ) = utils.get_measure_enhanced_noisy(output, batch.signals, metric)

        self.log_dict(metrics_dict)

        if idx == 0:
            with contextlib.suppress(KeyError):
                self.log_wave(batch, idx, output, "val")
        return {"loss_val": loss["loss"], "metrics": metrics_dict}

    def log_wave(self, batch, idx: int, output: dict, stage: str):
        """
        Logs audio samples to Weights & Biases.
        """
        noisy = batch.signals["input_eval"][0].T.detach().cpu().numpy()
        enhanced = output["input_proc"][0].T.detach().cpu().numpy()

        wandb.log(
            {
                f"audio/{stage}/noisy": wandb.Audio(
                    noisy, sample_rate=self.fs, caption=str(idx)
                )
            }
        )
        wandb.log(
            {
                f"audio/{stage}/enhanced": wandb.Audio(
                    enhanced, sample_rate=self.fs, caption=str(idx)
                )
            }
        )

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """
        Performs a single test step, calculates metrics, and saves enhanced audio.
        """

        signals, meta = batch.signals, batch.meta
        meta_dicts = meta.copy()  # [0]
        for d in meta_dicts:
            d["dataloader_idx"] = dataloader_idx

        output = self(signals)

        # save outputs
        filenames = [x["filename"].replace(".wav", "_enh.wav") for x in meta]
        for idx, filename in enumerate(filenames):
            self.save_individual_wave(
                dataloader_idx, output["input_proc"][idx], filename
            )
            if self.save_target:
                self.save_individual_wave(
                    dataloader_idx,
                    signals["target"][idx],
                    filename.replace("_enh", "_tgt"),
                )

        inputs = signals["input_eval"]
        preds = output["input_proc"]

        target = signals["target"]

        metric_collection = self._get_metric_collection("test", dataloader_idx)
        current_metrics = metric_collection(
            inputs, preds, target, meta_dicts, dataloader_idx
        )
        for key, val in current_metrics.items():
            self.log(
                f"metrics/test/{key}",
                val,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
                batch_size=self.batch_size,
            )

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.
        It collects all test results, combines them into a single DataFrame,
        adds complexity metrics, and saves the results to a CSV file.
        """
        # collect all results and save to dataframe
        dataframes = []
        for metric_collection in self.metric_collections["test"].values():
            for metric in metric_collection.values():
                if hasattr(metric, "dataframe"):
                    dataframes.append(metric.dataframe)

        combined_dataframe = pd.concat(dataframes, axis=0)
        combined_dataframe = combined_dataframe.loc[
            :, ~combined_dataframe.columns.duplicated()
        ].copy()
        combined_dataframe["filename"] = combined_dataframe.index

        # add complexity metrics as individual columns
        for key, value in self.complexity_metrics.items():
            combined_dataframe[key] = value

        save_dir = os.path.join(self.logger.experiment.notes, "test")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        combined_dataframe.to_csv(os.path.join(save_dir, "results.csv"), index=False)

    def train_dataloader(self):
        """
        Returns the training dataloader.
        """
        return self.datamodule.train_dataloader()

    def save_individual_wave(self, dataloader_idx, wave_tensor, filename):
        """
        Saves an individual audio waveform to a file.
        """
        save_dir = os.path.join(
            self.logger.experiment.notes,
            "test",
            str(dataloader_idx),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        utils.save_wave(
            data=wave_tensor.clone(),
            filename=os.path.join(save_dir, filename),
            fs=self.fs,
            normalize=True,
        )

    def on_fit_start(self):
        """
        Called at the beginning of the training fit.
        Computes complexity metrics if enabled.
        """
        if self.compute_complexity_metrics:
            self.compute_complexity_metrics_fn()

    def on_test_start(self):
        """
        Called at the beginning of the test.
        Computes complexity metrics if enabled.
        """
        if self.compute_complexity_metrics:
            self.compute_complexity_metrics_fn()

    def compute_complexity_metrics_fn(self):
        """
        Computes and logs complexity metrics for the model, such as the number
        of parameters, inference time, memory usage, FLOPs, and MACs.
        It performs profiling on both CPU and GPU.
        """
        print("computing complexity metrics...")
        # Number of trainable weights
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_channels = getattr(self, "num_channels", 1)
        binaural = getattr(self, "binaural", False)
        if "Bilat" in self.model_name or self.model_name == "BiConvTasNet":
            binaural = True
        if binaural:
            num_channels *= 2
        fs = getattr(self, "fs", 16000)

        # use torch.profiler with warmup steps and 3 repeats
        active = 10
        schedule = torch.profiler.schedule(
            wait=1,  # warmup steps
            warmup=1,
            active=active,  # number of repeats
            repeat=1,
        )
        self.eval()

        # CPU profiling
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.to("cpu")
        example_wave = torch.randn(1, num_channels, fs, device="cpu")
        example_input = {"input": example_wave, "input_eval": example_wave}
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=schedule,
                with_flops=True,
                profile_memory=True,
                record_shapes=False,
                with_stack=False,
            ) as prof:
                for _ in range(active + 2):  # total steps = wait + warmup + active
                    _ = self.forward_(example_input)
                    prof.step()
        prof_result_cpu = prof.key_averages()
        torch.set_num_threads(original_num_threads)

        # GPU profiling
        self.to("cuda")
        example_input = {k: v.to("cuda") for k, v in example_input.items()}
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=schedule,
                with_flops=True,
                profile_memory=True,
                record_shapes=False,
                with_stack=False,
            ) as prof:
                for _ in range(active + 2):  # total steps = wait + warmup + active
                    _ = self.forward_(example_input)
                    prof.step()

        prof_result_gpu = prof.key_averages()

        total_cpu_time = (
            sum([item.self_cpu_time_total for item in prof_result_cpu]) / active
        )
        total_cuda_time = (
            sum([item.self_cpu_time_total for item in prof_result_gpu]) / active
        )
        inference_time_cpu = total_cpu_time / 1000.0  # Convert μs to ms
        inference_time_gpu = total_cuda_time / 1000.0  # Convert μs to ms
        memory_used = sum([item.cpu_memory_usage for item in prof_result_cpu])
        total_flops_cpu = sum([item.flops for item in prof_result_cpu]) // active
        total_macs_cpu = total_flops_cpu // 2

        table = f"""
        Computational Complexity Metrics
        +--------------------------------+-------------------------+
        | Metric                         | Value                   |
        +--------------------------------+-------------------------+
        | Number of trainable parameters | {utils.format_parameters(total_params):<23} |
        | Inference time (CPU, ms)       | {inference_time_cpu:<23.2f} |
        | Inference time (GPU, ms)       | {inference_time_gpu:<23.2f} |
        | Memory usage                   | {utils.format_memory(memory_used):<23} |
        | FLOPS                          | {utils.format_flops(total_flops_cpu):<23} |
        | MACs                           | {utils.format_flops(total_macs_cpu):<23} |
        +--------------------------------+-------------------------+
        """
        print(table)
        wandb.run.summary["complexity/total_params"] = total_params
        wandb.run.summary["complexity/inference_time_cpu_ms"] = inference_time_cpu
        wandb.run.summary["complexity/inference_time_gpu_ms"] = inference_time_gpu
        wandb.run.summary["complexity/memory_used_bytes"] = memory_used
        wandb.run.summary["complexity/flops_cpu"] = total_flops_cpu
        wandb.run.summary["complexity/macs_cpu"] = total_macs_cpu

        # save as attributes
        self.complexity_metrics = {
            "total_params": total_params,
            "inference_time_cpu_ms": inference_time_cpu,
            "inference_time_gpu_ms": inference_time_gpu,
            "memory_used_bytes": memory_used,
            "flops_cpu": total_flops_cpu,
            "macs_cpu": total_macs_cpu,
        }
