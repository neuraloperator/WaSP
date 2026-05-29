"""
Training orchestration for WaSP.

Exposes :class:`Trainer`, which wires together the model, equation, replay buffer,
normalizer, and transport scheduler and runs the WaSP training loop.
"""

from __future__ import annotations

import warnings

import torch
import torch.distributed as dist
from neuralop import get_model
from neuralop.training import setup
from neuralop.utils import count_model_params
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .. import equations
from ..equations import EqCondition, EqSolution, EquationBase
from .checkpointing import CheckpointManager
from .loggers import WandBLogger
from .normalizer import Normalizer
from .replay_buffer import ReplayBuffer
from .scheduler import TransportScheduler
from .utils import filter_kwargs, optimizer_registry, scheduler_registry, update_config


class Trainer:
    """
    Orchestrates the WaSP training loop.

    Manages the model, equation, optimizer, learning-rate scheduler, replay buffer,
    normalizer, and adaptive transport scheduler. The main entry point is
    :meth:`__call__`, which calls :meth:`setup` and then :meth:`train`. Checkpoints
    can be resumed via the :meth:`load_checkpoint` class method.

    Attributes:
        config: Config dict built from the zencfg classes, set by :meth:`setup`;
            ``None`` before setup.
        model: The neural operator being trained.
        equation: PDE instance providing forward/backward operators and data sampling.
        optimizer: Parameter optimizer.
        scheduler: Learning-rate scheduler.
        normalizer: Online EMA normalizer for solution and input fields.
        buffer: Circular replay buffer supplying targets for the solution loss.
        transport_scheduler: Adaptive scheduler advancing the interpolation
            parameter ``s`` along the transport path.
        logger: W&B logger (batched; no-op when disabled).
        checkpoint_manager: Handles checkpoint serialization.
        device: Torch device resolved by the neuralop distributed setup.
        dtype: Floating-point dtype for model and data tensors.
        step: Current training step (1-indexed).

    """

    def __init__(self) -> None:
        """
        Initialize the trainer.
        """
        self.config = None

    def setup(self, config: dict, from_checkpoint: bool) -> None:
        """
        Setup the trainer with the given configuration.

        Args:
            config: Configuration dictionary.
            from_checkpoint: Whether to load from a checkpoint.

        """
        self.device, self.is_main_proc = setup(config)
        self.dtype: torch.dtype = getattr(torch, config.trainer.dtype)
        self.verbose = config.verbose and self.is_main_proc

        self.model = self.load_model(config)
        self.logger = WandBLogger(
            config,
            count_model_params(self.model),
            enabled=(
                self.is_main_proc
                and not config.get("debug", False)
                and not from_checkpoint
            ),
            log_every=config.trainer.get("log_every", 50),
        )

        self.equation = self.load_equation(config)
        self.optimizer, self.scheduler = self.load_optim(config)
        self.normalizer = self.load_normalizer(config)
        self.buffer = self.load_buffer(config, fill_buffer=not from_checkpoint)
        self.transport_scheduler = self.load_transport_scheduler(config)

        self.model = self.model.to(device=self.device)
        self.normalizer = self.normalizer.to(device=self.device)

        self.config = config
        self.step = 1

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.trainer.checkpoint_dir,
            save_checkpoints=config.trainer.get("save_checkpoints", False)
            and not config.get("debug", False),
        )

    def __call__(self, config: dict) -> None:
        """
        Set up the trainer from scratch and run the full training loop.

        Args:
            config: Config dict built from the zencfg classes.

        """
        self.setup(config, False)
        self.train()

        if self.is_main_proc:
            self.logger.finish()

    def load_model(self, config: dict) -> nn.Module:
        """
        Load the model based on the configuration.

        Args:
            config: Configuration dictionary
        Returns:
            model: Initialized model

        """
        model = get_model(config)
        if config.distributed.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            model = DDP(model, device_ids=[device_id], output_device=device_id)

        for param in model.parameters():
            if param.is_complex():
                param.data = param.data.to(self.dtype.to_complex())
            else:
                param.data = param.data.to(self.dtype.to_real())

        for buf in model.buffers():
            if buf.is_complex():
                buf.data = buf.data.to(self.dtype.to_complex())
            else:
                buf.data = buf.data.to(self.dtype.to_real())
        return model

    def load_equation(self, config: dict) -> EquationBase:
        """
        Load the equation based on the configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            equation: Initialized equation.

        """
        equation_cls = getattr(equations, config.equation.name)
        equation = equation_cls(**filter_kwargs(equation_cls, config.equation))
        equation.setup(self.device)
        return equation

    def load_optim(
        self, config: dict
    ) -> tuple[
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
    ]:
        """
        Load the optimizer and scheduler based on the configuration.

        Args:
            config: Configuration dictionary
        Returns:
            optimizer: optimizer for model learning
            scheduler: learning rate scheduler for model learning

        """
        optimizer_cls = optimizer_registry.get(config.trainer.optimizer)
        optimizer = optimizer_cls(
            self.model.parameters(),
            **filter_kwargs(optimizer_cls, config.trainer, exclude=["params"]),
        )

        scheduler_cls = scheduler_registry.get(config.trainer.scheduler)
        scheduler = scheduler_cls(
            optimizer,
            **filter_kwargs(scheduler_cls, config.trainer, exclude=["optimizer"]),
        )

        return optimizer, scheduler

    def load_normalizer(self, config: dict) -> Normalizer:
        """
        Build the online EMA normalizer from config and equation metadata.

        Args:
            config: Configuration dictionary.

        Returns:
            Normalizer initialized with the equation's solution and condition keys.

        """
        normalizer = Normalizer(
            solution_keys=self.equation.solution_normalizer_key,
            condition_keys=self.equation.condition_normalizer_key,
            momentum=config.trainer.normalizer_momentum,
        ).to(device=self.device)
        return normalizer

    def load_buffer(self, config: dict, fill_buffer: bool = True) -> ReplayBuffer:
        """
        Load the replay buffer based on the configuration.

        Args:
            config: Configuration dictionary
            fill_buffer: Whether to fill the buffer with initial data
        Returns:
            ReplayBuffer: Replay buffer

        """
        buffer = ReplayBuffer(
            max_size=config.trainer.buffer_size,
            solution_shapes=self.equation.solution_shapes,
            domain=self.equation.domain,
            device=torch.device("cpu"),
            dtype=self.dtype,
        )

        if not fill_buffer:
            return buffer

        if "buffer_gen_batch_size" in config.trainer:
            buffer_gen_batch_size = config.trainer.buffer_gen_batch_size
            for _ in range(config.trainer.init_buffer_size // buffer_gen_batch_size):
                solution = self.equation.sample_solution(
                    buffer_gen_batch_size,
                    device=self.device,
                )
                buffer.insert(solution)
                self.normalizer.update(solution, self.equation.forward(solution))
        else:
            solution = self.equation.sample_solution(
                config.trainer.init_buffer_size,
                device=self.device,
            )
            buffer.insert(solution)
            self.normalizer.update(solution, self.equation.forward(solution))

        return buffer

    def load_transport_scheduler(self, config: dict) -> TransportScheduler:
        """
        Build the adaptive transport scheduler from config.

        Args:
            config: Configuration dictionary.

        Returns:
            TransportScheduler controlling how the interpolation parameter ``s``
            advances along the transport path.

        """
        scheduler = TransportScheduler(
            enabled=config.trainer.transport_enabled,
            start_s=config.trainer.transport_start_s,
            threshold=config.trainer.transport_threshold,
            increment=config.trainer.transport_increment,
            min_sample=config.trainer.transport_min_sample,
            max_steps=config.trainer.n_steps,
            adaptive_range=config.trainer.transport_adaptive_range,
            pre_training_ratio=config.trainer.transport_pre_training_ratio,
            base_schedule=config.trainer.transport_base_schedule,
        )
        return scheduler

    def train_step(self, s: float, bs_buffer: int, bs_gen: int) -> None:
        """
        Take one optimizer step on the WaSP objective.

        Inputs are drawn from the manufactured input distribution nu_0, pushed to
        the interpolated distribution nu_s, and passed through the operator. The
        forward pass supplies the PDE residual loss on these fresh inputs, and its
        predictions enter the replay buffer as synthesized solutions. The solution
        loss regresses on buffered solutions u paired with their exact inputs N(u).

        Args:
            s: Interpolation parameter of the transport path, in [0, 1].
            bs_buffer: Batch size drawn from the replay buffer (solution loss).
            bs_gen: Batch size of freshly generated inputs (PDE residual loss).

        """
        if bs_buffer > 0:
            sol_buffer = self.buffer.sample(bs_buffer).to(device=self.device)
            cond_buffer = self.equation.forward(sol_buffer).to(device=self.device)

        sol_gen = self.equation.sample_solution(
            bs_gen or self.config.trainer.buffer_nsamples,
            device=self.device,
        )
        cond_gen = self.equation.forward(sol_gen)
        sol_gen = sol_gen.to(dtype=self.dtype)
        cond_gen = cond_gen.to(dtype=self.dtype)
        cond_gen = self.equation.map_condition(cond_gen, s=s)

        # sol_true / cond_true are regression targets (buffered or manufactured u,
        # sampled v), not solver labels
        if bs_buffer > 0 and bs_gen > 0:
            sol_true = EqSolution.from_list([sol_buffer, sol_gen])
            cond_true = EqCondition.from_list([cond_buffer, cond_gen])
        elif bs_buffer > 0:
            sol_true = sol_buffer
            cond_true = cond_buffer
        else:
            sol_true = sol_gen
            cond_true = cond_gen

        self.optimizer.zero_grad()

        norm_cond = self.normalizer.normalize_cond(cond_true)
        pred_u = self.model(self.equation.prepare_input(norm_cond))
        pred_u = self.equation.postprocess(pred_u, **self.config.trainer.postprocess)
        sol_pred = sol_true.new_solution(pred_u)
        sol_pred = self.normalizer.denormalize_sol(sol_pred)

        sol_buffer_pred, sol_gen_pred = sol_pred[:bs_buffer], sol_pred[bs_buffer:]

        if bs_gen > 0:
            cond_gen_pred = self.equation.forward(sol_gen_pred)

        if bs_buffer > 0:
            buf_loss, buf_loss_dict = self.equation.solution_loss(
                sol_buffer_pred,
                sol_buffer,
                normalizers=self.normalizer.stats,
                stage="train",
            )
        else:
            buf_loss = 0
            buf_loss_dict = {}

        if bs_gen > 0:
            gen_loss, gen_loss_dict = self.equation.condition_loss(
                cond_gen_pred,
                cond_gen,
                normalizers=self.normalizer.stats,
                stage="train",
            )
        else:
            gen_loss = 0
            gen_loss_dict = {}

        self.transport_scheduler.step(
            loss=gen_loss.item() if bs_gen > 0 else 0,
            numel=bs_gen,
            step=self.step,
        )

        loss = buf_loss + gen_loss

        loss.backward()
        if "max_grad_norm" in self.config.trainer:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.trainer.max_grad_norm
            )
        else:
            grad_norm = torch.nn.utils.get_total_norm(self.model.parameters())
        self.optimizer.step()
        self.scheduler.step()

        # insert the synthesized solutions once s > 0; during warm-up (s = 0)
        # insert the manufactured samples instead
        self.buffer.insert(
            eq_solution=sol_gen_pred if s > 0 else sol_gen,
            n=self.config.trainer.buffer_nsamples,
        )

        normalizer_dict = self.normalizer.update(
            sol_gen_pred, cond_gen, enabled=self.transport_scheduler.adaptive
        )

        self.logger.log(
            {
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/total_loss": loss.item() / (bs_buffer + bs_gen),
                "train/transport_s": self.transport_scheduler.s,
                "train/grad_norm": grad_norm.item(),
                **buf_loss_dict,
                **gen_loss_dict,
                **normalizer_dict,
            },
            step=self.step,
        )

    def train(self) -> None:
        """Run the full training loop until ``n_steps`` is reached."""
        bs_buffer = self.config.trainer.batch_size_buffer
        bs_gen = self.config.trainer.batch_size_gen
        assert bs_buffer > 0 or bs_gen > 0, "Batch size must be greater than 0"

        self.pbar = tqdm(total=self.config.trainer.n_steps, desc="Training")
        self.pbar.update(self.step)
        self.model.train()

        while self.step <= self.config.trainer.n_steps:
            s = self.transport_scheduler.s
            self.train_step(s, bs_buffer, bs_gen)

            if self.step % self.config.trainer.eval_freq == 0:
                self.model.eval()
                self.eval()
                self.model.train()
                self.visualize_buffer()

            if (
                self.is_main_proc
                and (self.step % self.config.trainer.save_freq == 0)
                and self.checkpoint_manager.save_checkpoints
            ):
                self.save_checkpoint()

            self.logger.commit(step=self.step)
            self.step += 1
            self.pbar.update(1)

    @torch.no_grad()
    def eval(self) -> None:
        """Evaluate the model on the test set and log losses and visualizations."""
        if "solution" in self.equation.test_data:
            solution = self.equation.test_data["solution"]
            if "condition" in self.equation.test_data:
                condition = self.equation.test_data["condition"]
            else:
                condition = self.equation.map_to_test_condition(solution)
        elif "condition" in self.equation.test_data:
            condition = self.equation.test_data["condition"]
            solution = self.equation.backward(condition)
        else:
            warnings.warn("No solution or condition found in test data, skipping eval")
            return

        eval_log_dict = {}
        batch_size = self.config.trainer.batch_size_eval
        eval_log_dict["eval/transport_s"] = self.transport_scheduler.s

        for start_idx in range(0, solution.n_samples, batch_size):
            end_idx = min(start_idx + batch_size, solution.n_samples)
            sol_true = solution[start_idx:end_idx].to(
                device=self.device, dtype=self.dtype
            )
            cond_true = condition[start_idx:end_idx].to(
                device=self.device, dtype=self.dtype
            )

            norm_cond = self.normalizer.normalize_cond(cond_true)
            pred_u = self.model(self.equation.prepare_input(norm_cond))
            pred_u = self.equation.postprocess(
                pred_u, **self.config.trainer.postprocess
            )
            sol_pred = sol_true.new_solution(pred_u)
            sol_pred = self.normalizer.denormalize_sol(sol_pred)

            cond_pred = self.equation.forward(sol_pred)

            _, sol_loss_dict = self.equation.solution_loss(
                sol_pred,
                sol_true,
                normalizers=self.normalizer.stats,
                stage="eval",
            )
            _, cond_loss_dict = self.equation.condition_loss(
                cond_pred,
                cond_true,
                normalizers=self.normalizer.stats,
                stage="eval",
            )

            for k, v in {**sol_loss_dict, **cond_loss_dict}.items():
                if k not in eval_log_dict:
                    eval_log_dict[k] = 0.0
                eval_log_dict[k] += v * (end_idx - start_idx) / solution.n_samples

            if start_idx == 0:
                eval_num_vis = min(self.config.trainer.eval_num_vis, solution.n_samples)

                img_pred_sol, img_pred_cond = self.equation.visualize(
                    sol_pred,
                    cond_pred,
                    self.normalizer.stats,
                )
                imgs_to_log = [img_pred_sol, img_pred_cond]
                names_to_log = ["solution", "condition"]

                img_gt_sol, img_gt_cond = self.equation.visualize(
                    sol_true,
                    cond_true,
                    self.normalizer.stats,
                )
                imgs_to_log.extend([img_gt_sol, img_gt_cond])
                names_to_log.extend(["solution_gt", "condition_gt"])

                for imgs, img_type in zip(imgs_to_log, names_to_log):
                    self.logger.log_images(
                        key=f"eval/images_{img_type}",
                        images=imgs[:eval_num_vis],
                        step=self.step,
                        caption_prefix=f"Eval {img_type.capitalize()} Sample",
                    )

        self.logger.log(eval_log_dict, step=self.step)

    def visualize_buffer(self) -> None:
        """Sample recent buffer entries and log solution/input images to W&B."""
        num_vis = min(self.config.trainer.eval_num_vis, self.buffer.size)
        buf_sol = self.buffer.sample(num_vis, last=True)
        buf_cond = self.equation.forward(buf_sol)
        imgs_buf_sol, imgs_buf_cond = self.equation.visualize(
            buf_sol.to(self.device),
            buf_cond.to(self.device),
            self.normalizer.stats,
        )
        for imgs, img_type in zip(
            [imgs_buf_sol, imgs_buf_cond],
            ["Solution", "Condition"],
        ):
            self.logger.log_images(
                key=f"train/images_{img_type}",
                images=imgs[:num_vis],
                step=self.step,
                caption_prefix=f"Buffer {img_type} Sample",
            )

    def save_checkpoint(self) -> None:
        """Write model weights, normalizer, and transport scheduler state to disk."""
        if dist.is_initialized():
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        dict_to_save = {
            "model": model_state,
            "step": self.step,
            "config": self.config,
            "normalizer": self.normalizer.state_dict(),
            "scheduler": self.transport_scheduler.state_dict(),
        }

        wandb_name = self.config.wandb.name if self.config.wandb.name else "run"
        self.checkpoint_manager.save(dict_to_save, self.step, name=wandb_name)

    @classmethod
    def load_checkpoint(cls, path: str, override_config: dict | None = None) -> Trainer:
        """
        Restore a Trainer from a checkpoint written by :meth:`save_checkpoint`.

        Args:
            path: Path to the ``.pt`` checkpoint file.
            override_config: Optional dict of config keys to override after loading,
                useful for changing eval-time settings without retraining.

        Returns:
            A fully set-up Trainer with model, normalizer, and scheduler state
            restored and ``step`` set to the saved value.

        """
        ckpt = CheckpointManager.load(path)

        if override_config is not None:
            update_config(ckpt["config"], override_config)

        trainer = cls()
        trainer.setup(ckpt["config"], from_checkpoint=True)
        trainer.step = ckpt["step"]

        CheckpointManager.load_model_state(trainer.model, ckpt["model"])

        normalizer_ckpt = ckpt["normalizer"]
        trainer.normalizer.load_state_dict(normalizer_ckpt)
        scheduler_ckpt = ckpt["scheduler"]
        trainer.transport_scheduler.load_state_dict(scheduler_ckpt)

        return trainer
