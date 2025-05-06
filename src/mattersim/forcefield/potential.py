# -*- coding: utf-8 -*-
"""
Potential
"""
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.units import GPa
from deprecated import deprecated
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from torchmetrics import MeanMetric

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.jit_compile_tools.jit import compile_mode
from mattersim.utils.download_utils import download_checkpoint
from mattersim.utils.logger_utils import get_logger

rank = int(os.getenv("RANK", 0))
logger = get_logger()


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
            "validation_metrics", {"loss": 10000000.0}  # noqa: E501
        )
        self.last_epoch = kwargs.get("last_epoch", -1)
        self.description = kwargs.get("description", "")
        self.saved_name = ["loss", "MAE_energy", "MAE_force", "MAE_stress"]
        self.best_metric = 10000
        self.best_metric_epoch = 0
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
            logger.info("fine-tuning all layers")
        elif finetune_layers >= 0 and finetune_layers < len(
            self.model.node_head.unified_encoder_layers
        ):
            logger.info(f"fine-tuning the last {finetune_layers} layers")
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
            logger.warning("Only graphormer and geomformer support freezing layers")
            return
        self.model.finetune_mode = True
        if finetune_head is None:
            logger.info("No finetune head is provided, using the original energy head")
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
        logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"  # noqa: E501
        )
        for epoch in range(self.last_epoch + 1, epochs):
            logger.info(f"Epoch: {epoch} / {epochs}")
            if need_to_load_data:
                assert isinstance(dataloader, list)
                random.Random(kwargs.get("seed", 42) + epoch).shuffle(  # noqa: E501
                    dataloader
                )
                for idx, data_path in enumerate(dataloader):
                    with open(data_path, "rb") as f:
                        start = time.time()
                        train_data = pickle.load(f)
                    logger.info(
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
                    metric = self.train_one_epoch(
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
                metric = self.train_one_epoch(
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
            if val_dataloader is not None:
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
                    logger.info("Early stopping")
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
            if epoch > self.best_metric_epoch + early_stop_patience:
                logger.info("Early stopping")
                return True

            if metric[self.idx] < self.best_metric:
                self.best_metric = metric[self.idx]
                self.best_metric_epoch = epoch
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
        **kwargs,
    ):
        """
        Test model performance on a given dataset
        """
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
            **kwargs,
        )

    def predict_properties(
        self,
        dataloader,
        include_forces: bool = False,
        include_stresses: bool = False,
        **kwargs,
    ) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
        """
        Predict properties (e.g., energies, forces) given a well-trained model
        Return: results tuple
            - results[0] (list[float]): a list of energies
            - results[1] (list[np.ndarray]): a list of atomic forces
            - results[2] (list[np.ndarray]): a list of stresses (in GPa)
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
            logger.info(
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

        return (loss_avg_, e_mae, f_mae, s_mae)

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
                    stresses = (
                        1 / volume[:, None, None] * stress_grad / GPa
                    )  # 1/GPa = 160.21766208
                    output["stresses"] = stresses

        return output

    def save(self, save_path):
        dir_name = os.path.dirname(save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
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

    @classmethod
    def from_checkpoint(
        cls,
        load_path: str = None,
        *,
        model_name: str = "m3gnet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_training_state: bool = True,
        **kwargs,
    ):
        if model_name.lower() != "m3gnet":
            raise NotImplementedError

        checkpoint_folder = os.path.expanduser("~/.local/mattersim/pretrained_models")
        os.makedirs(checkpoint_folder, exist_ok=True)
        if (
            load_path is None
            or load_path.lower() == "mattersim-v1.0.0-1m.pth"
            or load_path.lower() == "mattersim-v1.0.0-1m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-1M.pth")
            if not os.path.exists(load_path):
                logger.info(
                    "The pre-trained model is not found locally, "
                    "attempting to download it from the server."
                )
                download_checkpoint(
                    "mattersim-v1.0.0-1M.pth", save_folder=checkpoint_folder
                )
            logger.info(f"Loading the pre-trained {os.path.basename(load_path)} model")
        elif (
            load_path.lower() == "mattersim-v1.0.0-5m.pth"
            or load_path.lower() == "mattersim-v1.0.0-5m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-5M.pth")
            if not os.path.exists(load_path):
                logger.info(
                    "The pre-trained model is not found locally, "
                    "attempting to download it from the server."
                )
                download_checkpoint(
                    "mattersim-v1.0.0-5M.pth", save_folder=checkpoint_folder
                )
            logger.info(f"Loading the pre-trained {os.path.basename(load_path)} model")
        else:
            logger.info("Loading the model from %s" % load_path)
        assert os.path.exists(load_path), f"Model file {load_path} not found"

        checkpoint = torch.load(load_path, map_location=device)

        assert checkpoint["model_name"] == model_name
        checkpoint["model_args"].update(kwargs)
        model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
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

        return cls(
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

    @deprecated(version="1.0.0", reason="Please use from_checkpoint instead.")
    def load(
        load_path: str = None,
        *,
        model_name: str = "m3gnet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        args: Dict = None,
        load_training_state: bool = True,
        **kwargs,
    ):
        if model_name.lower() != "m3gnet":
            raise NotImplementedError

        checkpoint_folder = os.path.expanduser("~/.local/mattersim/pretrained_models")
        os.makedirs(checkpoint_folder, exist_ok=True)
        if (
            load_path is None
            or load_path.lower() == "mattersim-v1.0.0-1m.pth"
            or load_path.lower() == "mattersim-v1.0.0-1m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-1M.pth")
            if not os.path.exists(load_path):
                logger.info(
                    "The pre-trained model is not found locally, "
                    "attempting to download it from the server."
                )
                download_checkpoint(
                    "mattersim-v1.0.0-1M.pth", save_folder=checkpoint_folder
                )
            logger.info(f"Loading the pre-trained {os.path.basename(load_path)} model")
        elif (
            load_path.lower() == "mattersim-v1.0.0-5m.pth"
            or load_path.lower() == "mattersim-v1.0.0-5m"
        ):
            load_path = os.path.join(checkpoint_folder, "mattersim-v1.0.0-5M.pth")
            if not os.path.exists(load_path):
                logger.info(
                    "The pre-trained model is not found locally, "
                    "attempting to download it from the server."
                )
                download_checkpoint(
                    "mattersim-v1.0.0-5M.pth", save_folder=checkpoint_folder
                )
            logger.info(f"Loading the pre-trained {os.path.basename(load_path)} model")
        else:
            logger.info("Loading the model from %s" % load_path)

        assert os.path.exists(load_path), f"Model file {load_path} not found"

        checkpoint = torch.load(load_path, map_location=device)

        assert checkpoint["model_name"] == model_name
        checkpoint["model_args"].update(kwargs)
        model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
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


@deprecated(version="1.0.0", reason="Please use MatterSimCalculator instead.")
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

    @classmethod
    def from_checkpoint(cls, load_path: str, **kwargs):
        potential = Potential.from_checkpoint(load_path, **kwargs)
        return cls(potential, **kwargs)

    @classmethod
    def from_potential(cls, potential: Potential, **kwargs):
        return cls(potential, **kwargs)

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
        cutoff = (
            self.potential.model.model_args["cutoff"]
            if self.potential.model_name == "m3gnet"
            else 5.0
        )
        threebody_cutoff = (
            self.potential.model.model_args["threebody_cutoff"]
            if self.potential.model_name == "m3gnet"
            else 4.0
        )

        dataloader = build_dataloader(
            [atoms],
            model_type=self.potential.model_name,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            **self.args_dict,
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


class MatterSimCalculator(Calculator):
    """
    Deep calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        potential: Potential = None,
        args_dict: dict = {},
        compute_stress: bool = True,
        stress_weight: float = GPa,
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
        if potential is None:
            self.potential = Potential.from_checkpoint(device=device, **kwargs)
        else:
            self.potential = potential
        self.compute_stress = compute_stress
        self.stress_weight = stress_weight
        self.args_dict = args_dict
        self.device = device

    @classmethod
    def from_checkpoint(cls, load_path: str, **kwargs):
        potential = Potential.from_checkpoint(load_path, **kwargs)
        return cls(potential=potential, **kwargs)

    @classmethod
    def from_potential(cls, potential: Potential, **kwargs):
        return cls(potential=potential, **kwargs)

    @deprecated(
        version="1.0.0", reason="Please use from_checkpoint or from_potential instead."
    )
    def load(
        load_path: str = None,
        *,
        model_name: str = "m3gnet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        args: Dict = None,
        load_training_state: bool = True,
        args_dict: dict = {},
        compute_stress: bool = True,
        stress_weight: float = GPa,
        **kwargs,
    ):
        potential = Potential.load(
            load_path=load_path,
            model_name=model_name,
            device=device,
            args=args,
            load_training_state=load_training_state,
        )
        return MatterSimCalculator(
            potential=potential,
            args_dict=args_dict,
            compute_stress=compute_stress,
            stress_weight=stress_weight,
            device=device,
            **kwargs,
        )

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
        cutoff = (
            self.potential.model.model_args["cutoff"]
            if self.potential.model_name == "m3gnet"
            else 5.0
        )
        threebody_cutoff = (
            self.potential.model.model_args["threebody_cutoff"]
            if self.potential.model_name == "m3gnet"
            else 4.0
        )

        dataloader = build_dataloader(
            [atoms],
            model_type=self.potential.model_name,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            **self.args_dict,
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
