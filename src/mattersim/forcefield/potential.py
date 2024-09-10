# -*- coding: utf-8 -*-
"""
Potential
"""
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from torchmetrics import MeanMetric

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.m3gnet.m3gnet_multi_head import M3Gnet_multi_head
from mattersim.jit_compile_tools.jit import compile_mode


@compile_mode("script")
class Potential(nn.Module):
    """
    A wrapper class for the force field model
    """

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler: str = "StepLR",
        ema=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allow_tf32=False,
        **kwargs,
    ):
        """
        Args:
            potential : a force field model
            lr : learning rate
            scheduler : a torch scheduler
            normalizer : an energy normalization module
        """
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = Adam(
                self.model.parameters(), lr=kwargs.get("lr", 1e-3), eps=1e-7
            )
        else:
            self.optimizer = optimizer
        if not isinstance(scheduler, str):
            self.scheduler = scheduler
        elif scheduler == "StepLR":
            step_size = kwargs.get("step_size", 10)
            gamma = kwargs.get("gamma", 0.95)
            self.scheduler = StepLR(
                self.optimizer, step_size=step_size, gamma=gamma  # noqa: E501
            )
        elif scheduler == "ReduceLROnPlateau":
            factor = kwargs.get("factor", 0.8)
            patience = kwargs.get("patience", 50)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=factor,
                patience=patience,
                verbose=False,
            )
        else:
            raise NotImplementedError
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        self.device = device
        self.to(device)

        if ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=kwargs.get("ema_decay", 0.99)
            )
        else:
            self.ema = ema
        self.model_name = kwargs.get("model_name", "m3gnet")
        self.validation_metrics = kwargs.get(
            "validation_metrics", {"loss": 10000.0}  # noqa: E501
        )
        self.last_epoch = kwargs.get("last_epoch", -1)
        self.description = kwargs.get("description", "")
        self.saved_name = ["loss", "MAE_energy", "MAE_force", "MAE_stress"]
        self.best_metric = 10
        self.rank = None

        self.use_finetune_label_loss = kwargs.get("use_finetune_label_loss", False)

    def freeze_reset_model(
        self,
        finetune_layers: int = -1,
        reset_head_for_finetune: bool = False,
    ):
        """
        Freeze the model in the fine-tuning process
        """
        if finetune_layers == -1:
            print("fine-tuning all layers")
        elif finetune_layers >= 0 and finetune_layers < len(
            self.model.node_head.unified_encoder_layers
        ):
            print(f"fine-tuning the last {finetune_layers} layers")
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            # for energy head
            if finetune_layers > 0:
                for name, param in self.model.node_head.unified_encoder_layers[
                    -finetune_layers:
                ].named_parameters():
                    param.requires_grad = True
                for (
                    name,
                    param,
                ) in self.model.node_head.unified_final_invariant_ln.named_parameters():
                    param.requires_grad = True
                for (
                    name,
                    param,
                ) in self.model.node_head.unified_output_layer.named_parameters():
                    param.requires_grad = True
                for name, param in self.model.layer_norm.named_parameters():
                    param.requires_grad = True
            for name, param in self.model.lm_head_transform_weight.named_parameters():
                param.requires_grad = True
            for name, param in self.model.energy_out.named_parameters():
                param.requires_grad = True
            if reset_head_for_finetune:
                self.model.lm_head_transform_weight.reset_parameters()
                self.model.energy_out.reset_parameters()
        else:
            raise ValueError(
                "finetune_layers should be -1 or a positive integer,and less than the number of layers"  # noqa: E501
            )

    def finetune_mode(
        self,
        finetune_layers: int = -1,
        finetune_head: nn.Module = None,
        reset_head_for_finetune: bool = False,
        finetune_task_mean: float = 0.0,
        finetune_task_std: float = 1.0,
        use_finetune_label_loss: bool = False,
    ):
        """
        Set the model to fine-tuning mode
        finetune_layers: the layer to finetune, former layers will be frozen
                        if -1, all layers will be finetuned
        finetune_head: the head to finetune
        reset_head_for_finetune: whether to reset the original head
        """
        if self.model_name not in ["graphormer", "geomformer"]:
            print("Only graphormer and geomformer support freezing layers")
            return
        self.model.finetune_mode = True
        if finetune_head is None:
            print("No finetune head is provided, using the original energy head")
        self.model.finetune_head = finetune_head
        self.model.finetune_task_mean = finetune_task_mean
        self.model.finetune_task_std = finetune_task_std
        self.freeze_reset_model(finetune_layers, reset_head_for_finetune)
        self.use_finetune_label_loss = use_finetune_label_loss

    def train_model(
        self,
        dataloader: Optional[list],
        val_dataloader,
        loss: torch.nn.modules.loss = torch.nn.MSELoss(),
        include_energy: bool = True,
        include_forces: bool = False,
        include_stresses: bool = False,
        force_loss_ratio: float = 1.0,
        stress_loss_ratio: float = 0.1,
        epochs: int = 100,
        early_stop_patience: int = 100,
        metric_name: str = "val_loss",
        wandb=None,
        save_checkpoint: bool = False,
        save_path: str = "./results/",
        ckpt_interval: int = 10,
        multi_head: bool = False,
        dataset_name_list: List[str] = None,
        sampler=None,
        is_distributed: bool = False,
        need_to_load_data: bool = False,
        **kwargs,
    ):
        """
        Train model
        Args:
            dataloader: training data loader
            val_dataloader: validation data loader
            loss (torch.nn.modules.loss): loss object
            include_energy (bool) : whether to use energy as
                                    optimization targets
            include_forces (bool) : whether to use forces as
                                    optimization targets
            include_stresses (bool) : whether to use stresses as
                                      optimization targets
            force_loss_ratio (float): the ratio of forces in loss
            stress_loss_ratio (float): the ratio of stress in loss
            ckpt_interval (int): the interval to save checkpoints
            early_stop_patience (int): the patience for early stopping
            metric_name (str): the metric used for saving `best` checkpoints
                               and early stopping supported metrics:
                               `val_loss`, `val_mae_e`,
                               `val_mae_f`, `val_mae_s`
            sampler: used in distributed training
            is_distributed: whether to use DistributedDataParallel
            need_to_load_data: whether to load data from disk

        """
        self.idx = ["val_loss", "val_mae_e", "val_mae_f", "val_mae_s"].index(
            metric_name
        )
        if is_distributed:
            self.rank = torch.distributed.get_rank()
        print(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"  # noqa: E501
        )
        for epoch in range(self.last_epoch + 1, epochs):
            print(f"Epoch: {epoch} / {epochs}")
            if not multi_head:
                if need_to_load_data:
                    assert isinstance(dataloader, list)
                    random.Random(kwargs.get("seed", 42) + epoch).shuffle(  # noqa: E501
                        dataloader
                    )
                    for idx, data_path in enumerate(dataloader):
                        with open(data_path, "rb") as f:
                            start = time.time()
                            train_data = pickle.load(f)
                        print(
                            f"TRAIN: loading {data_path.split('/')[-2]}"
                            f"/{data_path.split('/')[-1]} dataset with "
                            f"{len(train_data)} data points, "
                            f"{len(train_data)} data points in total, "
                            f"time: {time.time() - start}"  # noqa: E501
                        )
                        # Distributed Sampling
                        atoms_train_sampler = (
                            torch.utils.data.distributed.DistributedSampler(
                                train_data,
                                seed=kwargs.get("seed", 42)
                                + idx * 131
                                + epoch,  # noqa: E501
                            )
                        )
                        train_dataloader = DataLoader(
                            train_data,
                            batch_size=kwargs.get("batch_size", 32),
                            shuffle=(atoms_train_sampler is None),
                            num_workers=0,
                            sampler=atoms_train_sampler,
                        )
                        self.train_one_epoch(
                            train_dataloader,
                            epoch,
                            loss,
                            include_energy,
                            include_forces,
                            include_stresses,
                            force_loss_ratio,
                            stress_loss_ratio,
                            wandb,
                            is_distributed,
                            mode="train",
                            **kwargs,
                        )
                        del train_dataloader
                        del train_data
                        torch.cuda.empty_cache()
                else:
                    self.train_one_epoch(
                        dataloader,
                        epoch,
                        loss,
                        include_energy,
                        include_forces,
                        include_stresses,
                        force_loss_ratio,
                        stress_loss_ratio,
                        wandb,
                        is_distributed,
                        mode="train",
                        **kwargs,
                    )
                metric = self.train_one_epoch(
                    val_dataloader,
                    epoch,
                    loss,
                    include_energy,
                    include_forces,
                    include_stresses,
                    force_loss_ratio,
                    stress_loss_ratio,
                    wandb,
                    is_distributed,
                    mode="val",
                    **kwargs,
                )
            else:
                assert dataset_name_list is not None
                assert (
                    need_to_load_data is False
                ), "load_training_data is not supported for multi-head training"  # noqa: E501
                self.train_one_epoch_multi_head(
                    dataloader,
                    dataset_name_list,
                    epoch,
                    loss,
                    include_energy,
                    include_forces,
                    include_stresses,
                    force_loss_ratio,
                    stress_loss_ratio,
                    wandb,
                    mode="train",
                    **kwargs,
                )
                metric = self.train_one_epoch_multi_head(
                    val_dataloader,
                    dataset_name_list,
                    epoch,
                    loss,
                    include_energy,
                    include_forces,
                    include_stresses,
                    force_loss_ratio,
                    stress_loss_ratio,
                    wandb,
                    mode="val",
                    **kwargs,
                )

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metric)
            else:
                self.scheduler.step()

            self.last_epoch = epoch

            self.validation_metrics = {
                "loss": metric[0],
                "MAE_energy": metric[1],
                "MAE_force": metric[2],
                "MAE_stress": metric[3],
            }
            if is_distributed:
                # TODO 添加distributed训练早停
                if self.save_model_ddp(
                    epoch,
                    early_stop_patience,
                    save_path,
                    metric_name,
                    save_checkpoint,
                    metric,
                    ckpt_interval,
                ):
                    break
            else:
                # return True时为早停
                if self.save_model(
                    epoch,
                    early_stop_patience,
                    save_path,
                    metric_name,
                    save_checkpoint,
                    metric,
                    ckpt_interval,
                ):
                    break

    def save_model(
        self,
        epoch,
        early_stop_patience,
        save_path,
        metric_name,
        save_checkpoint,
        metric,
        ckpt_interval,
    ):
        with self.ema.average_parameters():
            try:
                best_model = torch.load(
                    os.path.join(save_path, "best_model.pth")  # noqa: E501
                )
                assert metric_name in [
                    "val_loss",
                    "val_mae_e",
                    "val_mae_f",
                    "val_mae_s",
                ], (
                    f"`{metric_name}` metric name not supported. "
                    "supported metrics: `val_loss`, `val_mae_e`, "
                    "`val_mae_f`, `val_mae_s`"
                )

                if (
                    save_checkpoint is True
                    and metric[self.idx]
                    < best_model["validation_metrics"][
                        self.saved_name[self.idx]
                    ]  # noqa: E501
                ):
                    self.save(os.path.join(save_path, "best_model.pth"))
                if epoch > best_model["last_epoch"] + early_stop_patience:
                    print("Early stopping")
                    return True
                del best_model
            except BaseException:
                if save_checkpoint is True:
                    self.save(os.path.join(save_path, "best_model.pth"))

            if save_checkpoint is True and epoch % ckpt_interval == 0:
                self.save(os.path.join(save_path, f"ckpt_{epoch}.pth"))
            if save_checkpoint is True:
                self.save(os.path.join(save_path, "last_model.pth"))
            return False

    def save_model_ddp(
        self,
        epoch,
        early_stop_patience,
        save_path,
        metric_name,
        save_checkpoint,
        metric,
        ckpt_interval,
    ):
        with self.ema.average_parameters():
            assert metric_name in [
                "val_loss",
                "val_mae_e",
                "val_mae_f",
                "val_mae_s",
            ], (  # noqa: E501
                f"`{metric_name}` metric name not supported. "
                "supported metrics: `val_loss`, `val_mae_e`, "
                "`val_mae_f`, `val_mae_s`"
            )
            # Loading on multiple GPUs is too time consuming,
            # so this operation should not be performed.
            # Only save the model on GPU 0,
            # the model on each GPU should be exactly the same.

            if metric[self.idx] < self.best_metric:
                self.best_metric = metric[self.idx]
                if save_checkpoint and self.rank == 0:
                    self.save(os.path.join(save_path, "best_model.pth"))
            if self.rank == 0 and save_checkpoint:
                if epoch % ckpt_interval == 0:
                    self.save(os.path.join(save_path, f"ckpt_{epoch}.pth"))
                self.save(os.path.join(save_path, "last_model.pth"))
            # torch.distributed.barrier()
            return False

    def test_model(
        self,
        val_dataloader,
        loss: torch.nn.modules.loss = torch.nn.MSELoss(),
        include_energy: bool = True,
        include_forces: bool = False,
        include_stresses: bool = False,
        wandb=None,
        multi_head: bool = False,
        **kwargs,
    ):
        """
        Test model performance on a given dataset
        """
        if not multi_head:
            return self.train_one_epoch(
                val_dataloader,
                1,
                loss,
                include_energy,
                include_forces,
                include_stresses,
                1.0,
                0.1,
                wandb=wandb,
                mode="val",
            )
        else:
            return self.train_one_epoch_multi_head(
                val_dataloader,
                kwargs["dataset_name_list"],
                1,
                loss,
                include_energy,
                include_forces,
                include_stresses,
                1.0,
                0.1,
                wandb=wandb,
                mode="val",
            )

    def predict_properties(
        self,
        dataloader,
        include_forces: bool = False,
        include_stresses: bool = False,
        **kwargs,
    ):
        """
        Predict properties (e.g., energies, forces) given a well-trained model
        Return: results tuple
            - results[0] (list[float]): a list of energies
            - results[1] (list[np.ndarray]): a list of atomic forces
            - results[2] (list[np.ndarray]): a list of stresses
        """
        self.model.eval()
        energies = []
        forces = []
        stresses = []
        for batch_idx, graph_batch in enumerate(dataloader):
            if self.model_name == "graphormer" or self.model_name == "geomformer":
                raise NotImplementedError
            else:
                graph_batch.to(self.device)
                input = batch_to_dict(graph_batch)
            result = self.forward(
                input,
                include_forces=include_forces,
                include_stresses=include_stresses,  # noqa: E501
            )
            if self.model_name == "graphormer" or self.model_name == "geomformer":
                raise NotImplementedError
            else:
                energies.extend(result["total_energy"].cpu().tolist())
                if include_forces:
                    forces_tuple = torch.split(
                        result["forces"].cpu().detach(),
                        graph_batch.num_atoms.cpu().tolist(),
                        dim=0,
                    )
                    for atomic_force in forces_tuple:
                        forces.append(np.array(atomic_force))
                if include_stresses:
                    stresses.extend(list(result["stresses"].cpu().detach().numpy()))

        return (energies, forces, stresses)

    # ============================

    def train_one_epoch(
        self,
        dataloader,
        epoch,
        loss,
        include_energy,
        include_forces,
        include_stresses,
        loss_f,
        loss_s,
        wandb,
        is_distributed=False,
        mode="train",
        log=True,
        **kwargs,
    ):
        start_time = time.time()
        loss_avg = MeanMetric().to(self.device)
        train_e_mae = MeanMetric().to(self.device)
        train_f_mae = MeanMetric().to(self.device)
        train_s_mae = MeanMetric().to(self.device)

        # scaler = torch.cuda.amp.GradScaler()

        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()

        for batch_idx, graph_batch in enumerate(dataloader):
            if self.model_name == "graphormer" or self.model_name == "geomformer":
                raise NotImplementedError
            else:
                graph_batch.to(self.device)
                input = batch_to_dict(graph_batch)
            if mode == "train":
                result = self.forward(
                    input,
                    include_forces=include_forces,
                    include_stresses=include_stresses,
                )
            elif mode == "val":
                with self.ema.average_parameters():
                    result = self.forward(
                        input,
                        include_forces=include_forces,
                        include_stresses=include_stresses,
                    )

            loss_, e_mae, f_mae, s_mae = self.loss_calc(
                graph_batch,
                result,
                loss,
                include_energy,
                include_forces,
                include_stresses,
                loss_f,
                loss_s,
            )

            # loss backward
            if mode == "train":
                self.optimizer.zero_grad()
                loss_.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, norm_type=2  # noqa: E501
                )
                self.optimizer.step()
                # scaler.scale(loss_).backward()
                # scaler.step(self.optimizer)
                # scaler.update()
                self.ema.update()

            loss_avg.update(loss_.detach())
            if include_energy:
                train_e_mae.update(e_mae.detach())
            if include_forces:
                train_f_mae.update(f_mae.detach())
            if include_stresses:
                train_s_mae.update(s_mae.detach())

        loss_avg_ = loss_avg.compute().item()
        if include_energy:
            e_mae = train_e_mae.compute().item()
        else:
            e_mae = 0
        if include_forces:
            f_mae = train_f_mae.compute().item()
        else:
            f_mae = 0
        if include_stresses:
            s_mae = train_s_mae.compute().item()
        else:
            s_mae = 0

        if log:
            print(
                "%s: Loss: %.4f, MAE(e): %.4f, MAE(f): %.4f, MAE(s): %.4f, Time: %.2fs, lr: %.8f\n"  # noqa: E501
                % (
                    mode,
                    loss_avg.compute().item(),
                    e_mae,
                    f_mae,
                    s_mae,
                    time.time() - start_time,
                    self.scheduler.get_last_lr()[0],
                ),
                end="",
            )

        if wandb and ((not is_distributed) or self.rank == 0):
            wandb.log(
                {
                    f"{mode}/loss": loss_avg_,
                    f"{mode}/mae_e": e_mae,
                    f"{mode}/mae_f": f_mae,
                    f"{mode}/mae_s": s_mae,
                    f"{mode}/lr": self.scheduler.get_last_lr()[0],
                    f"{mode}/mae_tot": e_mae + f_mae + s_mae,
                },
                step=epoch,
            )

        if mode == "val":
            return (loss_avg_, e_mae, f_mae, s_mae)

    def train_one_epoch_multi_head(
        self,
        dataloader_list,
        dataset_name_list,
        epoch,
        loss,
        include_energy=True,
        include_forces=False,
        include_stresses=False,
        loss_f=1.0,
        loss_s=0.1,
        wandb=None,
        mode="train",
        **kwargs,
    ):
        start_time = time.time()

        metrics = {}
        for dataset_name in dataset_name_list:
            metrics_ = {}
            metrics_["loss_avg"] = MeanMetric().to(self.device)
            metrics_["train_e_mae"] = MeanMetric().to(self.device)
            metrics_["train_f_mae"] = MeanMetric().to(self.device)
            metrics_["train_s_mae"] = MeanMetric().to(self.device)
            metrics[dataset_name] = metrics_

        dataloader_iter = [
            dataloader.__iter__() for dataloader in dataloader_list  # noqa: E501
        ]
        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()

        dataloader_len = [len(dataloader) for dataloader in dataloader_list]
        for i in range(1, len(dataloader_len)):
            dataloader_len[i] += dataloader_len[i - 1]
        idx_list = list(range(dataloader_len[-1]))
        random.shuffle(idx_list)

        for idx in idx_list:
            for dataset_idx, bound in enumerate(dataloader_len):
                if idx < bound:
                    break

            graph_batch = dataloader_iter[dataset_idx].__next__()
            graph_batch.to(self.device)
            input = batch_to_dict(graph_batch)
            dataset_name = dataset_name_list[dataset_idx]

            if mode == "train":
                result = self.forward(
                    input,
                    include_forces=include_forces,
                    include_stresses=include_stresses,
                    dataset_idx=dataset_idx,
                )
            elif mode == "val":
                with self.ema.average_parameters():
                    result = self.forward(
                        input,
                        include_forces=include_forces,
                        include_stresses=include_stresses,
                        dataset_idx=dataset_idx,
                    )

            loss_, e_mae, f_mae, s_mae = self.loss_calc(
                graph_batch,
                result,
                loss,
                include_energy,
                include_forces,
                include_stresses,
                loss_f,
                loss_s,
            )

            # loss backward
            if mode == "train":
                self.optimizer.zero_grad()
                loss_.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0, norm_type=2  # noqa: E501
                )
                self.optimizer.step()
                self.ema.update()

            metrics[dataset_name]["loss_avg"].update(loss_.detach())
            if include_energy:
                metrics[dataset_name]["train_e_mae"].update(e_mae.detach())
            if include_forces:
                metrics[dataset_name]["train_f_mae"].update(f_mae.detach())
            if include_stresses:
                metrics[dataset_name]["train_s_mae"].update(s_mae.detach())

        loss_all = 0
        e_mae = 0
        f_mae = 0
        s_mae = 0
        for dataset_name in dataset_name_list:
            train_f_mae = train_s_mae = 0
            loss_avg = metrics[dataset_name]["loss_avg"].compute().item()
            loss_all += loss_avg
            if include_energy:
                train_e_mae = metrics[dataset_name]["train_e_mae"].compute().item()
                e_mae += train_e_mae
            if include_forces and (dataset_name != "QM9"):
                train_f_mae = (
                    metrics[dataset_name]["train_f_mae"].compute().item()
                )  # noqa: E501
                f_mae += train_f_mae
            if include_stresses:
                train_s_mae = (
                    metrics[dataset_name]["train_s_mae"].compute().item()
                )  # noqa: E501
                s_mae += train_s_mae

            print(
                "%s %s: Loss: %.4f, MAE(e): %.4f, MAE(f): %.4f, MAE(s): %.4f, Time: %.2fs"  # noqa: E501
                % (
                    dataset_name,
                    mode,
                    loss_avg,
                    train_e_mae,
                    train_f_mae,
                    train_s_mae,
                    time.time() - start_time,
                )
            )

            if wandb:
                wandb.log(
                    {
                        f"{dataset_name}/{mode}_loss": loss_avg,
                        f"{dataset_name}/{mode}_mae_e": train_e_mae,
                        f"{dataset_name}/{mode}_mae_f": train_f_mae,
                        f"{dataset_name}/{mode}_mae_s": train_s_mae,
                    },
                    step=epoch,
                )

        if wandb:
            wandb.log({"lr": self.scheduler.get_last_lr()[0]}, step=epoch)

        if mode == "val":
            return (loss_all, e_mae, f_mae, s_mae)

    def loss_calc(
        self,
        graph_batch,
        result,
        loss,
        include_energy,
        include_forces,
        include_stresses,
        loss_f=1.0,
        loss_s=0.1,
    ):
        e_mae = 0.0
        f_mae = 0.0
        s_mae = 0.0
        loss_ = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.model_name == "graphormer" or self.model_name == "geomformer":
            raise NotImplementedError
        else:
            if include_energy:
                e_gt = graph_batch.energy / graph_batch.num_atoms
                e_pred = result["total_energy"] / graph_batch.num_atoms
                loss_ = loss_ + loss(e_pred, e_gt)
                e_mae = torch.nn.L1Loss()(e_pred, e_gt)
            if include_forces:
                f_gt = graph_batch.forces
                f_pred = result["forces"]
                loss_ = loss_ + loss(f_pred, f_gt) * loss_f
                f_mae = torch.nn.L1Loss()(f_pred, f_gt)
                # f_mae = torch.mean(torch.abs(f_pred - f_gt)).item()
            if include_stresses:
                s_gt = graph_batch.stress
                s_pred = result["stresses"]
                loss_ = loss_ + loss(s_pred, s_gt) * loss_s
                s_mae = torch.nn.L1Loss()(s_pred, s_gt)
                # s_mae = torch.mean(torch.abs((s_pred - s_gt))).item()
        return loss_, e_mae, f_mae, s_mae

    def get_properties(
        self,
        graph_batch,
        include_forces: bool = True,
        include_stresses: bool = True,
        **kwargs,
    ):
        """
        get energy, force and stress from a list of graph
        Args:
            graph_batch:
            include_forces (bool): whether to include force
            include_stresses (bool): whether to include stress
        Returns:
            results: a tuple, which consists of energies, forces and stress
        """
        warnings.warn(
            "This interface (get_properties) has been deprecated. "
            "Please use Potential.forward(input, include_forces, "
            "include_stresses) instead.",
            DeprecationWarning,
        )
        if self.model_name == "graphormer" or self.model_name == "geomformer":
            raise NotImplementedError
        else:
            graph_batch.to(self.device)
            input = batch_to_dict(graph_batch)
        result = self.forward(
            input,
            include_forces=include_forces,
            include_stresses=include_stresses,
            **kwargs,
        )
        # Warning: tuple
        if not include_forces and not include_stresses:
            return (result["total_energy"],)
        elif include_forces and not include_stresses:
            return (result["total_energy"], result["forces"])
        elif include_forces and include_stresses:
            return (result["total_energy"], result["forces"], result["stresses"])

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        include_forces: bool = True,
        include_stresses: bool = True,
        dataset_idx: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        get energy, force and stress from a list of graph
        Args:
            input: a dictionary contains all necessary info.
                   The `batch_to_dict` method could convert a graph_batch from
                   pyg dataloader to the input dictionary.
            include_forces (bool): whether to include force
            include_stresses (bool): whether to include stress
            dataset_idx (int): used for multi-head model, set to -1 by default
        Returns:
            results: a dictionary, which consists of energies,
                     forces and stresses
        """
        output = {}
        if self.model_name == "graphormer" or self.model_name == "geomformer":
            raise NotImplementedError
        else:
            strain = torch.zeros_like(input["cell"], device=self.device)
            volume = torch.linalg.det(input["cell"])
            if include_forces is True:
                input["atom_pos"].requires_grad_(True)
            if include_stresses is True:
                strain.requires_grad_(True)
                input["cell"] = torch.matmul(
                    input["cell"],
                    (torch.eye(3, device=self.device)[None, ...] + strain),
                )
                strain_augment = torch.repeat_interleave(
                    strain, input["num_atoms"], dim=0
                )
                input["atom_pos"] = torch.einsum(
                    "bi, bij -> bj",
                    input["atom_pos"],
                    (torch.eye(3, device=self.device)[None, ...] + strain_augment),
                )
                volume = torch.linalg.det(input["cell"])

            energies = self.model.forward(input, dataset_idx)
            output["total_energy"] = energies

            # Only take first derivative if only force is required
            if include_forces is True and include_stresses is False:
                grad_outputs: List[Optional[torch.Tensor]] = [
                    torch.ones_like(
                        energies,
                    )
                ]
                grad = torch.autograd.grad(
                    outputs=[
                        energies,
                    ],
                    inputs=[input["atom_pos"]],
                    grad_outputs=grad_outputs,
                    create_graph=self.model.training,
                )

                # Dump out gradient for forces
                force_grad = grad[0]
                if force_grad is not None:
                    forces = torch.neg(force_grad)
                    output["forces"] = forces

            # Take derivatives up to second order
            # if both forces and stresses are required
            if include_forces is True and include_stresses is True:
                grad_outputs: List[Optional[torch.Tensor]] = [
                    torch.ones_like(
                        energies,
                    )
                ]
                grad = torch.autograd.grad(
                    outputs=[
                        energies,
                    ],
                    inputs=[input["atom_pos"], strain],
                    grad_outputs=grad_outputs,
                    create_graph=self.model.training,
                )

                # Dump out gradient for forces and stresses
                force_grad = grad[0]
                stress_grad = grad[1]

                if force_grad is not None:
                    forces = torch.neg(force_grad)
                    output["forces"] = forces

                if stress_grad is not None:
                    stresses = 1 / volume[:, None, None] * stress_grad * 160.21766208
                    output["stresses"] = stresses

        return output

    def save(self, save_path):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # 保存为单卡可加载的模型，多卡加载时需要先加载后放入DDP中
        checkpoint = {
            "model_name": self.model_name,
            "model": self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict(),
            "model_args": self.model.module.get_model_args()
            if hasattr(self.model, "module")
            else self.model.get_model_args(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "last_epoch": self.last_epoch,
            "validation_metrics": self.validation_metrics,
            "description": self.description,
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load(
        model_name: str = "m3gnet",
        load_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        args: Dict = None,
        load_training_state: bool = True,
        **kwargs,
    ):
        if load_path is None:
            if model_name == "m3gnet":
                print("Loading the pre-trained M3GNet model")
                current_dir = os.path.dirname(__file__)
                load_path = os.path.join(
                    current_dir, "m3gnet/pretrained/mpf/best_model.pth"
                )
            elif model_name == "graphormer" or model_name == "geomformer":
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            print("Loading the model from %s" % load_path)

        checkpoint = torch.load(load_path, map_location=device)

        assert checkpoint["model_name"] == model_name
        if model_name == "m3gnet":
            model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
        elif model_name == "m3gnet_multi_head":
            model = M3Gnet_multi_head(device=device, **checkpoint["model_args"]).to(
                device
            )
        elif model_name == "graphormer" or model_name == "geomformer":
            raise NotImplementedError
        else:
            raise NotImplementedError
        model.load_state_dict(checkpoint["model"], strict=False)

        if load_training_state:
            optimizer = Adam(model.parameters())
            scheduler = StepLR(optimizer, step_size=10, gamma=0.95)
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except BaseException:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"].state_dict())
                except BaseException:
                    optimizer = None
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
            except BaseException:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"].state_dict())
                except BaseException:
                    scheduler = "StepLR"
            try:
                last_epoch = checkpoint["last_epoch"]
                validation_metrics = checkpoint["validation_metrics"]
                description = checkpoint["description"]
            except BaseException:
                last_epoch = -1
                validation_metrics = {"loss": 0.0}
                description = ""
            try:
                ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
                ema.load_state_dict(checkpoint["ema"])
            except BaseException:
                ema = None
        else:
            optimizer = None
            scheduler = "StepLR"
            last_epoch = -1
            validation_metrics = {"loss": 0.0}
            description = ""
            ema = None

        model.eval()

        del checkpoint

        return Potential(
            model,
            optimizer=optimizer,
            ema=ema,
            scheduler=scheduler,
            device=device,
            model_name=model_name,
            last_epoch=last_epoch,
            validation_metrics=validation_metrics,
            description=description,
            **kwargs,
        )

    @staticmethod
    def load_from_multi_head_model(
        model_name: str = "m3gnet",
        head_index: int = -1,
        load_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Load one head of the multi-head model.
        Args:
            head_index:
                -1: reset the head (final layer and
                energy normalization module)
        """
        if load_path is None:
            if model_name == "m3gnet":
                print("Loading the pre-trained multi-head M3GNet model")
                current_dir = os.path.dirname(__file__)
                load_path = os.path.join(
                    current_dir,
                    "m3gnet/pretrained/Transition1x-MD17-MPF21-QM9-HME21-OC20/"
                    "best_model.pth",
                )
            else:
                raise NotImplementedError
        else:
            print("Loading the model from %s" % load_path)
        if head_index == -1:
            print("Reset the final layer and normalization module")
        checkpoint = torch.load(load_path, map_location=device)
        if model_name == "m3gnet":
            model = M3Gnet(device=device, **checkpoint["model_args"]).to(
                device
            )  # noqa: E501
            ori_ckpt = checkpoint["model"].copy()
            for key in ori_ckpt:
                if "final_layer_list" in key:
                    if "final_layer_list.%d" % head_index in key:
                        checkpoint["model"][
                            key.replace("_layer_list.%d" % head_index, "")
                        ] = ori_ckpt[key]
                    del checkpoint["model"][key]
                if "normalizer_list" in key:
                    if "normalizer_list.%d" % head_index in key:
                        checkpoint["model"][
                            key.replace("_list.%d" % head_index, "")
                        ] = ori_ckpt[key]
                    del checkpoint["model"][key]
                if "sph_2" in key:
                    del checkpoint["model"][key]
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            raise NotImplementedError
        description = checkpoint["description"]
        model.eval()

        del checkpoint

        return Potential(
            model,
            device=device,
            model_name=model_name,
            description=description,
            **kwargs,
        )

    def load_model(self, **kwargs):
        warnings.warn(
            "The interface of loading M3GNet model has been deprecated. "
            "Please use Potential.load() instead.",
            DeprecationWarning,
        )
        warnings.warn(
            "It only supports loading the pre-trained M3GNet model. "
            "For other models, please use Potential.load() instead."
        )
        current_dir = os.path.dirname(__file__)
        load_path = os.path.join(
            current_dir, "m3gnet/pretrained/mpf/best_model.pth"  # noqa: E501
        )
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint["model"])

    def set_description(self, description):
        self.description = description

    def get_description(self):
        return self.description


def batch_to_dict(graph_batch, model_type="m3gnet", device="cuda"):
    if model_type == "m3gnet":
        # TODO: key_list
        atom_pos = graph_batch.atom_pos
        cell = graph_batch.cell
        pbc_offsets = graph_batch.pbc_offsets
        atom_attr = graph_batch.atom_attr
        edge_index = graph_batch.edge_index
        three_body_indices = graph_batch.three_body_indices
        num_three_body = graph_batch.num_three_body
        num_bonds = graph_batch.num_bonds
        num_triple_ij = graph_batch.num_triple_ij
        num_atoms = graph_batch.num_atoms
        num_graphs = graph_batch.num_graphs
        num_graphs = torch.tensor(num_graphs)
        batch = graph_batch.batch

        # Resemble input dictionary
        input = {}
        input["atom_pos"] = atom_pos
        input["cell"] = cell
        input["pbc_offsets"] = pbc_offsets
        input["atom_attr"] = atom_attr
        input["edge_index"] = edge_index
        input["three_body_indices"] = three_body_indices
        input["num_three_body"] = num_three_body
        input["num_bonds"] = num_bonds
        input["num_triple_ij"] = num_triple_ij
        input["num_atoms"] = num_atoms
        input["num_graphs"] = num_graphs
        input["batch"] = batch
    elif model_type == "graphormer" or model_type == "geomformer":
        raise NotImplementedError
    else:
        raise NotImplementedError

    return input


class DeepCalculator(Calculator):
    """
    Deep calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        potential: Potential,
        args_dict: dict = {},
        compute_stress: bool = True,
        stress_weight: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """
        Args:
            potential (Potential): m3gnet.models.Potential
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.args_dict = args_dict
        self.device = device

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        self.args_dict["batch_size"] = 1
        self.args_dict["only_inference"] = 1
        dataloader = build_dataloader(
            [atoms], model_type=self.potential.model_name, **self.args_dict
        )
        for graph_batch in dataloader:
            # Resemble input dictionary
            if (
                self.potential.model_name == "graphormer"
                or self.potential.model_name == "geomformer"
            ):
                raise NotImplementedError
            else:
                graph_batch = graph_batch.to(self.device)
                input = batch_to_dict(graph_batch)

            result = self.potential.forward(
                input, include_forces=True, include_stresses=self.compute_stress
            )
            if (
                self.potential.model_name == "graphormer"
                or self.potential.model_name == "geomformer"
            ):
                raise NotImplementedError
            else:
                self.results.update(
                    energy=result["total_energy"].detach().cpu().numpy()[0],
                    free_energy=result["total_energy"].detach().cpu().numpy()[0],
                    forces=result["forces"].detach().cpu().numpy(),
                )
            if self.compute_stress:
                self.results.update(
                    stress=self.stress_weight
                    * full_3x3_to_voigt_6_stress(
                        result["stresses"].detach().cpu().numpy()[0]
                    )
                )
