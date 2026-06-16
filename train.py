# Standard library
import argparse
import json
import logging
import os
import psutil
import random
import time
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

# Third party
import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)

# Local
import tplr


class Timer:
    """Context manager for timing code blocks."""

    _timings = defaultdict(list)
    _active_timers = {}
    _disable = False

    def __init__(self, name, logger=None, disabled=False, enabled=False):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.disabled = disabled or (Timer._disable and not enabled)

    def __enter__(self):
        if self.disabled:
            return self

        self.start_time = time.perf_counter()
        Timer._active_timers[self.name] = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled or self.start_time is None:
            return

        end_time = time.perf_counter()
        duration = end_time - self.start_time
        Timer._timings[self.name].append(duration)

        if self.logger and self.name in Timer._active_timers:
            self.logger.debug(f"{self.name}: {duration:.6f}s")

        if self.name in Timer._active_timers:
            del Timer._active_timers[self.name]

    @classmethod
    def get_stats(cls, name=None):
        """Get timing statistics for a specific timer or all timers."""
        if name is not None:
            times = cls._timings.get(name, [])
            if not times:
                return {}
            return {
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "last": times[-1],
            }
        else:
            return {name: cls.get_stats(name) for name in cls._timings.keys()}

    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls._timings = defaultdict(list)
        cls._active_timers = {}

    @classmethod
    def disable(cls, disabled=True):
        """Disable all timers."""
        cls._disable = disabled

    @classmethod
    def summarize(cls, logger=None):
        """Summarize all timings."""
        result = {}
        for name, times in cls._timings.items():
            if not times:
                continue

            stats = cls.get_stats(name)
            msg = (
                f"{name} - total: {stats['total']:.3f}s, "
                f"mean: {stats['mean']:.3f}s, "
                f"max: {stats['max']:.3f}s"
            )

            result[name] = stats

            if logger:
                logger.info(msg)

        return result


def dict_parser_type(value):
    """Helper function to parse a JSON string into a dict for argparse."""
    try:
        value = value.replace("'", '"')
        loaded_dict = json.loads(value)
        return loaded_dict
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for dictionary: {value}")


class DistributedLLMTrainer:
    """Distributed LLM Trainer."""

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description="Distributed LoCo Trainer")
        parser.add_argument("--project", type=str, default="boom", help="Wandb project.")
        parser.add_argument("--run_name", type=str, default="", help="Wandb run name.")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--trace", action="store_true", help="Enable trace logging")
        parser.add_argument("--hparams_file", type=str, default="hparams.json", help="hparams file.")
        parser.add_argument("--use_compile", action="store_true", help="Use torch.compile to optimize model execution")

        # Virtual workers per GPU
        parser.add_argument("--virtual_workers", type=int, default=1, help="Number of virtual workers per physical GPU (>=1).")

        # Optimizer args
        parser.add_argument("--micro_batch_size", type=int, default=-1, help="Micro batches for data loader")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size for grad accum")
        parser.add_argument("--sequence_length", type=int, default=2048, help="sequence length for training")
        parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for regularization")
        parser.add_argument("--warmup_steps", type=float, default=0.07, help="Number of warmup steps for learning rate scheduler")
        parser.add_argument("--warmup_start_factor", type=float, default=0.1, help="Start factor for warmup learning rate scheduler")

        # Strategy args
        parser.add_argument("--strategy", type=str, default="diloco", help="Training strategy to use")
        parser.add_argument("--inner_optimizer", type=str, default=None, choices=["adamw"], help="inner optimizer to use. None means simple gradient accumulation")
        parser.add_argument("--outer_optimizer", type=str, default="adamw", choices=["adamw", "mergeloco", "tsvloco", "isocloco", "isoctsloco", "nesterov", "sgd"], help="Outer optimizer to use for training")

        # Inner optimizer
        parser.add_argument("--inner_steps", type=int, default=15, help="Local steps before communication (H)")
        parser.add_argument("--inner_learning_rate", type=float, default=6e-4, help="Learning rate for inner optimizer")
        parser.add_argument("--inner_beta1", type=float, default=0.9, help="Beta1 parameter for inner AdamW optimizer")
        parser.add_argument("--inner_beta2", type=float, default=0.95, help="Beta2 parameter for inner AdamW optimizer")

        # Outer optimizer
        parser.add_argument("--outer_learning_rate", type=float, default=0.7, help="Learning rate for outer optimizer")
        parser.add_argument("--outer_momentum", type=float, default=0.9, help="Momentum for outer optimizer")
        parser.add_argument("--outer_nesterov", action="store_true", help="Nesterov for outer optimizer")
        # MergeLoCo specific args
        parser.add_argument("--sparsity", type=float, default=0.0, help="Proportion of parameters to remove (0.0 = no sparsification)")
        parser.add_argument("--prune_method", type=str, default="magnitude", choices=["magnitude", "random"], help="Pruning method")
        parser.add_argument("--prune_dimension", type=str, default="all", choices=["row", "column", "all"], help="Dimension for pruning")
        parser.add_argument("--rescale_pruned", type=str, default="false", choices=["true", "false"], help="Rescale pruned gradients")
        parser.add_argument("--ties_disjoint", type=str, default="false", choices=["true", "false"], help="TIES-disjoint merge")

        # TSVLoCo specific args
        parser.add_argument("--tsv_singular_ratio", type=float, default=None, help="Proportion of singular components to keep.")
        parser.add_argument("--tsv_decorrelate", type=str, default="true", choices=["true", "false"], help="TSV decorrelate matrices")

        # ISOCLoCo specific args
        parser.add_argument("--orthogonalize_object", type=str, default="gradient", choices=["gradient", "update"],
            help="For IsoLoCo: apply ISO-C to 'gradient' (before momentum) or 'update' (after momentum)")

        # ISOCTSLoCo specific args
        parser.add_argument("--common_space_fraction", type=float, default=0.8,
            help="Fraction of SVD dimensions allocated to common space in ISO-CTS (default: 0.8)")

        # Dataset args
        parser.add_argument("--token_budget", type=int, default=15728640, help="Token budget for training. If negative, is set from hparams file.")
        parser.add_argument("--shards_path", type=str, default="~/datasets/dclm_tokenized-train_valid_test", help="Path to the dataset shards.")
        parser.add_argument("--shard_token_size", type=int, default=1024**3, help="Number of tokens in each shard (must match shard creation)")
        parser.add_argument("--eval_tokens", type=int, default=100 * 1024**2, help="Number of tokens to evaluate on (default: ~100M)")
        parser.add_argument("--eval_split", type=str, default="valid", choices=["valid", "test"], help="Which split to evaluate on (default: valid)")
        parser.add_argument("--eval_throughout", action="store_true", help="Evaluate throughout training (default: only at the end)")
        parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (None for unlimited)")
        parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic page selection")
        parser.add_argument("--data_in_gpu", action="store_true", help="Keep whole dataset in GPU.")

        # Checkpoint args
        parser.add_argument("--save_path", type=str, default="~/checkpoints", help="Path to save model checkpoints")
        parser.add_argument("--save_interval", type=int, default=1000000000, help="Save checkpoint every N windows")
        parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")

        # Timing args
        parser.add_argument("--timing_log", type=str, default="timings.log", help="File to write timing information to")

        # Expand paths
        config = parser.parse_args()
        config.save_path = os.path.expandvars(os.path.expanduser(config.save_path))
        config.shards_path = os.path.expandvars(os.path.expanduser(config.shards_path))
        config.hparams_file = os.path.expandvars(os.path.expanduser(config.hparams_file))

        # Setup the predefined strategy
        tplr.logger.info(f"Strategy: {config.strategy}:")
        if config.strategy == "adam_baseline":
            tplr.logger.info(
                f"[Strat] Hardcoding inner optimizer to None (simple grad accumulation), outer optimizer to 'adamw', and inner_steps to 1."
            )
            config.inner_optimizer = None
            config.outer_optimizer = "adamw"
            config.inner_steps = 1
        elif config.strategy == "diloco_baseline":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'SGD+Nesterov'."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "nesterov"
        elif config.strategy == "diloco_sgd_noMom":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'SGD' (no momentum)."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "sgd"
        elif config.strategy == "mergeloco":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'mergeloco'."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "mergeloco"
        elif config.strategy == "tsvloco":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'tsvloco'."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "tsvloco"
        elif config.strategy == "isocloco":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'isocloco' (no momentum)."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "isocloco"
            config.outer_momentum = 0.0  # Hardcode no momentum
            config.outer_nesterov = False
            config.orthogonalize_object = "gradient"
        elif config.strategy == "isoloco":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'isocloco' with momentum."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "isocloco"
            # Momentum controlled by --outer_momentum and --outer_nesterov
        elif config.strategy == "isoctsloco":
            tplr.logger.info(
                f"[Strat] Inner optimizer set to '{config.inner_optimizer or 'adamw'}' and outer optimizer to 'isoctsloco' (no momentum)."
            )
            config.inner_optimizer = config.inner_optimizer or "adamw"
            config.outer_optimizer = "isoctsloco"
            config.outer_momentum = 0.0
            config.outer_nesterov = False
        else:
            if config.strategy != "custom":
                tplr.logger.warning(
                    f"Unknown strategy '{config.strategy}', defaulting to 'custom'"
                )
            tplr.logger.info(
                f"[Strat] Using custom strategy with inner optimizer '{config.inner_optimizer}' and outer optimizer '{config.outer_optimizer}'."
            )

        if config.micro_batch_size < 0:
            config.micro_batch_size = config.batch_size
            
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()

        tplr.logger.info(f"Config: {config}")
        return config

    def __init__(self):
        tplr.logger.debug("Starting AdamW baseline initialization...")
        self.config = DistributedLLMTrainer.config()
        self.is_diloco = self.config.inner_optimizer is not None
        self._set_seed_and_backend(self.config.seed)
        self._setup_distributed()
        self._calculate_steps()
        self._initialize_model_and_tokenizer()
        self._setup_optimizers_and_schedulers()
        self._initialize_state_and_metrics()
        self._initialize_dataloader()
        self._initialize_eval_dataloader()
        self._setup_wandb_and_logging()
        self._initialize_strategy()

        # summary info
        if self.global_rank == 0:
            # Calculate expected training time and tokens
            tokens_per_step = (
                self.config.batch_size
                * self.world_size
                * self.virtual_workers  
                * self.config.sequence_length
                * self.config.inner_steps
            )
            total_tokens = tokens_per_step * self.config.max_steps

            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB

            tplr.logger.info("\n" + "=" * 80)
            tplr.logger.info(f"TRAINING CONFIGURATION SUMMARY:")
            tplr.logger.info(f"→ Hardware: {self.world_size} GPU(s)  | virtual_workers/GPU: {self.virtual_workers}  | effective workers: {self.effective_world_size}")
            tplr.logger.info(f"→ Effective world size: {self.effective_world_size} (physical {self.world_size} × virtual {self.virtual_workers})")
            tplr.logger.info(f"→ Model memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved (excluding batches)")
            tplr.logger.info(f"→ Training strategy: {self.config.strategy.upper()} with {self.config.inner_steps} inner steps")

            if self.is_diloco:
                tplr.logger.info(
                    f"→ Inner optimizer: {self.config.inner_optimizer} (lr={self.config.inner_learning_rate}, weight_decay={self.config.weight_decay}, inner_steps={self.config.inner_steps})"
                )

            tplr.logger.info(f"→ Outer optimizer: {self.config.outer_optimizer} (lr={self.config.outer_learning_rate}, weight_decay={self.outer_weight_decay})")
            tplr.logger.info(f"→ Batch hierarchy: {self.config.micro_batch_size} (micro) → {self.config.batch_size} (accum)")
            tplr.logger.info(f"→ Sequence length: {self.config.sequence_length} tokens per sample")

            # Add token computation information
            inner_effective_tokens = (
                self.config.batch_size
                * self.world_size
                * self.virtual_workers
                * self.config.inner_steps
                * self.config.sequence_length
            )

            tplr.logger.info(f"→ Inner cycle: {inner_effective_tokens:,} tokens processed per full inner cycle across all (virtual) workers")
            tplr.logger.info(f"→ Training plan: {self.config.max_steps:,} steps, targeting {total_tokens:,} tokens total (given target: {self.config.token_budget:,})")
            tplr.logger.info(f"→ Scheduler plan: {self.warmup_steps:,} warmup steps, {self.cosine_steps:,} cosine steps, {self.total_scheduler_steps:,} total scheduler steps")
            tplr.logger.info(
                f"→ Data: per-virtual samples: {[len(ds) for ds in self._datasets]} "
                f"(sum={sum(len(ds) for ds in self._datasets)}) with seq_len={self.config.sequence_length:,}"
            )

            if self.config.use_compile:
                tplr.logger.info(f"→ Optimization: Using torch.compile for model execution")

            tplr.logger.info("=" * 80 + "\n")

    def _set_seed_and_backend(self, seed):
        """Sets the seed and torch.backend."""
        tplr.logger.info(f"Setting global seed to {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _setup_distributed(self):
        """Set up the distributed training environment."""
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.world_size > 1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
        else:
            self.local_rank = 0
            self.global_rank = 0

        self.virtual_workers = max(1, int(self.config.virtual_workers))
        self.effective_world_size = self.world_size * self.virtual_workers
        self.virtual_rank_base = self.global_rank * self.virtual_workers

        Timer.disable(not (self.config.debug and self.global_rank == 0))

        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            init_process_group(
                backend="nccl", rank=self.global_rank, world_size=self.world_size
            )
            tplr.logger.info(f"Initialized DDP: rank {self.global_rank}/{self.world_size - 1} on device {self.local_rank}")

        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")

    def _calculate_steps(self):
        """Calculate training steps."""
        if not self.is_diloco:
            self.config.inner_steps = 1

        # Max outer steps to hit token_budget (if -1)
        if self.config.max_steps == -1:
            self.config.max_steps = self.config.token_budget // (
                self.config.batch_size * self.config.sequence_length * self.config.inner_steps * self.effective_world_size
            )

        # Total scheduler "iteration count" (used by warmup/cosine)
        # This matches how many (inner or outer) optimizer .step() calls we expect over the budget.
        # For Diloco, we step inner each inner step; with virtual workers, that multiplies too.
        self.total_scheduler_steps = self.config.token_budget // (
            self.config.batch_size * self.config.sequence_length * self.effective_world_size
        )

    def _initialize_model_and_tokenizer(self):
        """Initialize the model(s) and tokenizer."""
        with open(self.config.hparams_file, "r") as fp:
            hparams = json.load(fp)

        tokenizer_name = hparams.pop("tokenizer_name")
        model_config = LlamaConfig(**hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        master = LlamaForCausalLM(model_config).to(self.device)

        self.hparams = SimpleNamespace(model_config=model_config, tokenizer=self.tokenizer)

        if self.global_rank == 0:
            total = sum(p.numel() for p in master.parameters())
            train = sum(p.numel() for p in master.parameters() if p.requires_grad)
            tplr.logger.info(f"using model config: {model_config}")
            tplr.logger.info(f"→ Total params:     {total:,}")
            tplr.logger.info(f"→ Trainable params: {train:,}")

        # Broadcast initial weights if distributed (do this before cloning VMs)
        if self.world_size > 1:
            for p in master.parameters():
                dist.broadcast(p.data, src=0)
            if self.global_rank == 0:
                tplr.logger.info("Synchronized model parameters across all processes")

        # Set as master (uncompiled for now so state_dict is clean)
        self.model = master
        self.virtual_models = None

        # Helper: always pull state_dict from the real, unwrapped module
        def _unwrap(m):
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            return m

        # TODO: Check if virtual_workers > 1 works without diloco
        self.virtual_models = [self.model]
        base_sd = _unwrap(self.model).state_dict()
        for _ in range(self.virtual_workers-1):
            vm = LlamaForCausalLM(model_config).to(self.device)
            vm.load_state_dict(base_sd, strict=True)
            self.virtual_models.append(vm)

        # Compile after broadcasting weights
        if self.config.use_compile:
            self.virtual_models = [torch.compile(vm, dynamic=True) for vm in self.virtual_models]


    def _initialize_dataloader(self):
        """Initialize one dataloader per (virtual) rank."""
        if self.global_rank == 0:
            ram_before = psutil.virtual_memory()
            tplr.logger.info(
                f"RAM before dataset creation: {ram_before.used / 1024**3:.2f}GB used, "
                f"{ram_before.available / 1024**3:.2f}GB available"
            )

        # Helper to create one dataset/dataloader for a specific effective rank
        def _make_loader(effective_rank: int):
            ds = tplr.ShardedGPUDataset(
                shards_path=self.config.shards_path,
                token_budget=self.config.token_budget,
                sequence_length=self.config.sequence_length,
                rank=effective_rank,
                world_size=self.effective_world_size,   # key change: partition using VIRTUAL world size
                device=self.device,
                split="train",
                pin_to_gpu=self.config.data_in_gpu,
                shard_token_size=self.config.shard_token_size,
            )
            return tplr.get_dataloader(
                ds, batch_size=self.config.micro_batch_size, shuffle=True
            ), ds

        # Build loaders
        self.train_loaders = []
        self._datasets = []   # for logging only
        for v in range(self.virtual_workers):
            eff_rank = self.virtual_rank_base + v
            loader, ds = _make_loader(eff_rank)
            self.train_loaders.append(loader)
            self._datasets.append(ds)

        if self.global_rank == 0:
            ram_after = psutil.virtual_memory()
            tplr.logger.info(
                f"RAM after dataset creation: {ram_after.used / 1024**3:.2f}GB used, "
                f"{ram_after.available / 1024**3:.2f}GB available"
            )

    def _initialize_eval_dataloader(self):
        """Initialize evaluation dataloader. All ranks load their portion for distributed eval."""
        self.eval_loader = None

        # Check if eval shard exists (only rank 0 checks to avoid log spam)
        eval_path = os.path.join(self.config.shards_path, f"{self.config.eval_split}_000000.npy")
        if self.global_rank == 0 and not os.path.exists(eval_path):
            tplr.logger.warning(f"Eval file not found at {eval_path}. Skipping {self.config.eval_split} evaluation.")

        if not os.path.exists(eval_path):
            return

        tplr.logger.info(f"[Rank {self.global_rank}] Loading {self.config.eval_split} eval data from {self.config.shards_path}")
        eval_dataset = tplr.ShardedGPUDataset(
            shards_path=self.config.shards_path,
            token_budget=self.config.eval_tokens,
            sequence_length=self.config.sequence_length,
            rank=self.global_rank,
            world_size=self.world_size,
            device=self.device,
            split=self.config.eval_split,
            pin_to_gpu=self.config.data_in_gpu,
            shard_token_size=self.config.eval_tokens,  # Single shard
        )

        self.eval_loader = tplr.get_dataloader(
            eval_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=False,
        )

    def _evaluate(self):
        """Evaluate model on eval set. All ranks participate, results aggregated."""
        if self.eval_loader is None:
            return None

        self.model.eval()
        sum_nll = 0.0
        sum_tokens = 0

        with torch.no_grad():
            for batch in self.eval_loader:
                if not self.config.data_in_gpu:
                    batch = batch.to(self.device)

                # Create labels with padding masked to -100 (matching training behavior)
                labels = batch.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                # Count valid (non-padding) tokens
                valid_tokens = (labels != -100).sum().item()

                # Skip batch if all tokens are padding
                if valid_tokens == 0:
                    continue

                # Use autocast to match training precision (BF16)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch, labels=labels)

                # Accumulate token-weighted loss
                # outputs.loss is mean loss per valid token, multiply by valid_tokens for total NLL
                sum_nll += outputs.loss.item() * valid_tokens
                sum_tokens += valid_tokens

        self.model.train()

        # Aggregate across all ranks
        loss_tensor = torch.tensor([sum_nll, float(sum_tokens)], device=self.device)
        if self.world_size > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        global_sum_nll = loss_tensor[0].item()
        global_sum_tokens = loss_tensor[1].item()

        # Handle edge case where all tokens are padding
        if global_sum_tokens == 0:
            if self.global_rank == 0:
                tplr.logger.warning(f"No valid tokens found in {self.config.eval_split} eval set")
            return None

        avg_loss = global_sum_nll / global_sum_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Only rank 0 returns metrics for logging
        if self.global_rank == 0:
            return {
                "eval_loss": avg_loss,
                "eval_perplexity": perplexity,
                "eval_tokens": int(global_sum_tokens),
            }
        return None

    def _create_scheduler(self, optimizer, lr):
        """Create a standard scheduler with warmup and cosine annealing."""
        warmup_steps = self.config.warmup_steps
        # If warmup_steps is given as a fraction of total steps:
        if warmup_steps < 1:
            warmup_steps = self.total_scheduler_steps * warmup_steps

        warmup_steps = int(warmup_steps)
        cosine_steps = max(1, self.total_scheduler_steps - warmup_steps)
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

        if warmup_steps >= self.total_scheduler_steps:
            raise ValueError(
                f"Warmup steps ({self.config.warmup_steps:,}) must be less than total scheduler steps "
                f"({self.total_scheduler_steps:,})."
            )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.config.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=lr * 0.1,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_optimizers_and_schedulers(self):
        """Set up optimizers and schedulers for training."""
        # ---- INNER OPTIMIZER(S) ----
        self.inner_optimizers = [None] * self.virtual_workers
        self.inner_schedulers = [None] * self.virtual_workers

        if self.is_diloco:
            # One inner optimizer/scheduler per virtual model
            self.inner_optimizers = []
            self.inner_schedulers = []
            if self.config.inner_optimizer.lower() == "adamw":
                for vm in self.virtual_models:
                    opt = AdamW(
                        vm.parameters(),
                        lr=self.config.inner_learning_rate,
                        weight_decay=self.config.weight_decay,
                        betas=(self.config.inner_beta1, self.config.inner_beta2),
                    )
                    self.inner_optimizers.append(opt)
            else:
                raise NotImplementedError(f"Unknown inner optimizer: {self.config.inner_optimizer}")

            # One scheduler object per inner optimizer
            for opt in self.inner_optimizers:
                self.inner_schedulers.append(self._create_scheduler(opt, self.config.inner_learning_rate))
            if self.global_rank == 0:
                tplr.logger.info(
                    f"Using {self.config.inner_optimizer} inner optimizers (x{self.virtual_workers}) with lr={self.config.inner_learning_rate}, "
                    f"weight_decay={self.config.weight_decay}"
                    )

        # ---- OUTER OPTIMIZER ----
        self.outer_weight_decay = (0.0 if self.is_diloco else self.config.weight_decay)

        # one outer optimizer per GPU. If Diloco, it wraps all virtual models' params
        per_vm_outer = self.config.outer_optimizer.lower() in ["mergeloco", "tsvloco", "isocloco", "isoctsloco"]
        # Optimizers that handle their own communication need per-virtual-model optimizers
        # for local momentums; otherwise we optimize the first virtual model and broadcast.
        outer_params = [vm.parameters() for vm in self.virtual_models] if per_vm_outer else self.virtual_models[0].parameters()

        if self.config.outer_optimizer.lower() == "adamw":
            self.outer_optimizer = AdamW(
                outer_params,
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                betas=(self.config.inner_beta1, self.config.inner_beta2),
                eps=0.1 if self.is_diloco else 1e-8,
            )
        elif self.config.outer_optimizer.lower() == "nesterov":
            self.outer_optimizer = SGD(
                outer_params,
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                momentum=self.config.outer_momentum,
                nesterov=True,
            )
        elif self.config.outer_optimizer.lower() == "sgd":
            self.outer_optimizer = SGD(
                outer_params,
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                momentum=0.0,
                nesterov=False,
            )
        elif self.config.outer_optimizer.lower() == "mergeloco":
            rescale_pruned = self.config.rescale_pruned.lower() == "true"
            ties_disjoint = self.config.ties_disjoint.lower() == "true"
            
            self.outer_optimizer = tplr.MergeLoCo(
                outer_params,
                lr=self.config.outer_learning_rate,
                momentum=self.config.outer_momentum,
                nesterov=self.config.outer_nesterov,
                weight_decay=self.outer_weight_decay,
                sparsity=self.config.sparsity,
                prune_method=self.config.prune_method,
                prune_dimension=self.config.prune_dimension,
                rescale_pruned=rescale_pruned,
                ties_disjoint=ties_disjoint,
                virtual_workers=self.virtual_workers,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
            )
        elif self.config.outer_optimizer.lower() == "tsvloco":
            decorrelate = self.config.tsv_decorrelate.lower() == "true"
         
            self.outer_optimizer = tplr.TSVLoCo(
                outer_params,
                lr=self.config.outer_learning_rate,
                singular_ratio=self.config.tsv_singular_ratio,
                decorrelate=decorrelate,
                virtual_workers=self.virtual_workers,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
            )
        elif self.config.outer_optimizer.lower() == "isocloco":
            self.outer_optimizer = tplr.ISOCLoCo(
                outer_params,
                lr=self.config.outer_learning_rate,
                momentum=self.config.outer_momentum,
                nesterov=self.config.outer_nesterov,
                orthogonalize_object=self.config.orthogonalize_object,
                virtual_workers=self.virtual_workers,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
            )
        elif self.config.outer_optimizer.lower() == "isoctsloco":
            self.outer_optimizer = tplr.ISOCTSLoCo(
                outer_params,
                lr=self.config.outer_learning_rate,
                momentum=self.config.outer_momentum,
                nesterov=self.config.outer_nesterov,
                common_space_fraction=self.config.common_space_fraction,
                virtual_workers=self.virtual_workers,
                process_group=dist.group.WORLD if self.world_size > 1 else None,
            )
        else:
            raise NotImplementedError(f"Unknown outer optimizer: {self.config.outer_optimizer}")

        if self.global_rank == 0:
            tplr.logger.info(
                f"Using {self.config.outer_optimizer} outer optimizer with LR={self.config.outer_learning_rate} "
                f"and weight_decay={self.outer_weight_decay}"
            )

        # ---- SCHEDULERS ----
        # Scheduler runs on inner if Diloco, otherwise on outer
        if self.is_diloco:
            self.scheduler = None
            self.outer_scheduler = None
            # already created inner_schedulers/inner_schedulers above
        else:
            self.scheduler = self._create_scheduler(self.outer_optimizer, self.config.outer_learning_rate)
            self.inner_schedulers = [None] * self.virtual_workers
            self.outer_scheduler = self.scheduler

    def _initialize_state_and_metrics(self):
        """Initialize state variables and metrics tracking."""
        if self.global_rank == 0:
            os.makedirs(self.config.save_path, exist_ok=True)

        self.step_counter = 0
        self.global_step = 0

        self.total_tokens_processed = 0
        self.batch_times = []

        if self.config.load_checkpoint is not None and False: # TODO: No checkpoint loading for now
            self._load_checkpoint(self.config.load_checkpoint)

    def _setup_wandb_and_logging(self):
        """Set up WandB and timing loggers."""
        if self.global_rank == 0:
            self.wandb = wandb.init(
                project=self.config.project,
                name=f"{self.config.run_name}" if self.config.run_name else "runner",
                config=vars(self.config),
                group="loco",
                job_type="loco_training",
            )
        else:
            self.wandb = None

        self.timing_logger = None
        if self.config.debug:
            self.setup_timing_logger()

    def _initialize_strategy(self):
        """Initialize the training strategy."""
        if self.is_diloco:
            self.strategy = tplr.Diloco(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )
        else:
            self.strategy = tplr.SimpleAccum(
                self.device,
                self.world_size,
                self.global_rank,
                self.tokenizer,
                self.config,
            )

    def setup_timing_logger(self):
        """Set up a separate logger for performance timing information."""
        log_dir = os.path.dirname(self.config.timing_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        self.timing_logger = logging.getLogger("timing")
        self.timing_logger.setLevel(logging.DEBUG)
        self.timing_logger.propagate = False  # Don't propagate to root logger

        if self.timing_logger.handlers:
            self.timing_logger.handlers.clear()
        file_handler = logging.FileHandler(self.config.timing_log, mode="w")

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)

        self.timing_logger.addHandler(file_handler)

        self.timing_logger.info(
            f"Starting new training run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.timing_logger.info(
            f"Configuration: optimizer={self.config.outer_optimizer}, lr={self.config.outer_learning_rate}, "
            f"world_size={self.world_size}, batch_size={self.config.batch_size}"
        )
        self.timing_logger.info("-" * 80)

    def log_timing(self, message):
        """Helper to log timing information to the timing log file."""
        if self.global_rank == 0 and self.timing_logger is not None:
            self.timing_logger.info(message)

    def run(self):
        """Main training loop."""
        for window in range(self.global_step, self.config.max_steps):
            if self.global_rank == 0:
                tplr.logger.info(
                    f"\n{'-' * 40} Window: {window}/{self.config.max_steps} {'-' * 40}"
                )
                if self.config.debug:
                    self.log_timing(f"Window {window} - Starting gradient accumulation")
                Timer.reset()

            with Timer("window_total", enabled=True):
                if self.global_rank == 0:
                    tplr.logger.info("Start accumulating gradients...")

                for vm in self.virtual_models:
                    vm.zero_grad(set_to_none=True)

                # ---- INNER STEP ----
                with Timer("inner_step"):
                    per_gpu_metrics = {
                            "total_loss": 0,
                            "loss_after_gather": 0,
                            "batch_count": 0,
                            "batch_tokens": 0,
                        }
                    for vmodel, voptim, vsched, vloader in zip(
                        self.virtual_models, 
                        self.inner_optimizers, 
                        self.inner_schedulers, 
                        self.train_loaders
                        ):

                        virtual_metrics = self.strategy.inner_step(vmodel, vloader, voptim, vsched)

                        per_gpu_metrics["total_loss"] += virtual_metrics["total_loss"] / self.virtual_workers
                        per_gpu_metrics["loss_after_gather"] += virtual_metrics["loss_after_gather"] / self.virtual_workers
                        per_gpu_metrics["batch_count"] += virtual_metrics["batch_count"]
                        per_gpu_metrics["batch_tokens"] += virtual_metrics["batch_tokens"]


                # ---- REDUCE METRICS ACROSS GPUs ----
                with Timer("reduce_metrics"):
                    metrics_to_reduce = torch.tensor(
                        [
                            per_gpu_metrics["total_loss"],
                            per_gpu_metrics["batch_count"],
                            per_gpu_metrics["batch_tokens"],
                            per_gpu_metrics["loss_after_gather"],
                        ],
                        device=self.device,
                    )
                    if self.world_size > 1:
                        torch.distributed.all_reduce(metrics_to_reduce, op=torch.distributed.ReduceOp.SUM)

                    loss_after_inner = metrics_to_reduce[0].item() / self.world_size
                    batch_count = int(metrics_to_reduce[1].item())
                    batch_tokens = int(metrics_to_reduce[2].item())
                    loss_after_gather = metrics_to_reduce[3].item() / self.world_size

                # ---- OUTER STEP ----
                with Timer("outer_step"):
                    # We have only one outer optimizer per GPU
                    self.strategy.outer_step(
                        self.virtual_models,
                        self.outer_optimizer,
                        self.outer_scheduler,
                    )

                overlap_metrics = {}

            # ---- EVALUATION (all ranks participate) ----
            # Evaluate throughout if flag is set, otherwise only at the end
            is_last_step = (window == self.config.max_steps - 1)
            should_eval = self.config.eval_throughout or is_last_step

            eval_metrics = None
            if should_eval:
                with Timer(f"Evaluation", enabled=True):
                    eval_metrics = self._evaluate()

            if self.global_rank == 0:
                stats = Timer.get_stats(f"Evaluation")
                if stats and eval_metrics is not None:
                    tplr.logger.info(
                        f"[Eval:{self.config.eval_split}] "
                        f"loss={eval_metrics['eval_loss']:.4f}, "
                        f"ppl={eval_metrics['eval_perplexity']:.2f}, "
                        f"tokens={eval_metrics['eval_tokens']:,} | "
                        f"took {stats['last']:.2f}s "
                        f"(mean {stats['mean']:.2f}s)"
                    )

            # ---- LOGGING ----
            if self.global_rank == 0:
                all_stats = Timer.summarize(logger=self.timing_logger if self.config.debug else None)
                window_duration = all_stats.get("window_total", {}).get("total", 0.0)
                tokens_per_second = (batch_tokens / window_duration) if window_duration > 0 else 0.0

                tplr.logger.info(f"Window {window}: Processing rate: {tokens_per_second:.2f} tokens/sec")

                # Gradient stats from MASTER model
                grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                weight_norms = [p.norm().item() for p in self.model.parameters()]

                # Effective batch size includes virtual workers *and* inner steps (for Diloco)
                eff_bs = self.config.batch_size * self.world_size * (self.config.inner_steps) * self.virtual_workers
                
                learning_rate = self.inner_schedulers[0].get_last_lr()[0] if self.is_diloco else self.outer_scheduler.get_last_lr()[0]
                metrics_dict = {
                    "baseline/loss_after_inner": loss_after_inner,
                    "baseline/loss_after_gather": loss_after_gather,
                    "baseline/total_tokens": self.total_tokens_processed + batch_tokens,
                    "baseline/batch_tokens": batch_tokens,
                    "baseline/global_step": self.global_step,
                    "baseline/perplexity_after_inner": torch.exp(torch.tensor(loss_after_inner)).item(),
                    "baseline/perplexity_after_gather": torch.exp(torch.tensor(loss_after_gather)).item(),
                    "baseline/tokens_per_sec": tokens_per_second,
                    "misc/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
                    "misc/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,
                    "setting/num_gpus": self.world_size,
                    "setting/virtual_workers": self.virtual_workers,
                    "setting/effective_batch_size": eff_bs,
                    "setting/learning_rate": learning_rate,
                    "gradient/mean_grad_norm": (sum(grad_norms) / len(grad_norms)) if grad_norms else 0,
                    "gradient/max_grad_norm": max(grad_norms) if grad_norms else 0,
                    "gradient/min_grad_norm": min(grad_norms) if grad_norms else 0,
                    "gradient/grad_norm_std": (torch.tensor(grad_norms).std().item() if grad_norms else 0),
                    "gradient/mean_weight_norm": (sum(weight_norms) / len(weight_norms)) if weight_norms else 0,
                    "gradient/grad_to_weight_ratio": (
                        (sum(grad_norms) / len(grad_norms)) / (sum(weight_norms) / len(weight_norms))
                        if grad_norms and weight_norms else 0
                    ),
                }

                # Timer metrics (unchanged)
                if self.config.debug:
                    self.log_timing(f"Window {window} - Timing summary:")
                    self.log_timing(f"  Total tokens: {batch_tokens}, Tokens/sec: {tokens_per_second:.2f}")
                    for timer_name, stats in all_stats.items():
                        metrics_dict[f"timing/{timer_name}/total"] = stats.get("total", 0)
                        metrics_dict[f"timing/{timer_name}/mean"] = stats.get("mean", 0)
                        metrics_dict[f"timing/{timer_name}/max"] = stats.get("max", 0)
                    self.log_timing("-" * 40)

                if overlap_metrics:
                    metrics_dict.update(overlap_metrics)

                if eval_metrics:
                    metrics_dict[f"eval/{self.config.eval_split}_loss"] = eval_metrics["eval_loss"]
                    metrics_dict[f"eval/{self.config.eval_split}_perplexity"] = eval_metrics["eval_perplexity"]
                    metrics_dict[f"eval/{self.config.eval_split}_tokens"] = eval_metrics["eval_tokens"]

                self.wandb.log(metrics_dict, step=self.global_step)

                self.total_tokens_processed += batch_tokens

                if ((window + 1) % self.config.save_interval == 0) or (window == self.config.max_steps - 1):
                    self._save_checkpoint(window)
            else:
                self.total_tokens_processed += batch_tokens

            self.global_step += 1

        if self.global_rank == 0:
            tplr.logger.info(f"Training completed with {self.total_tokens_processed} tokens processed.")

    def _save_checkpoint(self, window):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.save_path, f"checkpoint_window_{window}_{timestamp}.pt")
        tplr.logger.info(f"Saving checkpoint for window {window} to {path}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            # "optimizer_state_dict": self.outer_optimizer.state_dict(),
            # "scheduler_state_dict": self.scheduler.state_dict()
            # if self.scheduler
            # else None,
        }

        # Add training state
        checkpoint.update(
            {
                "window": window,
                "global_step": self.global_step,
            }
        )

        torch.save(checkpoint, path)
        tplr.logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            tplr.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer and scheduler states
        self.outer_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
            and self.scheduler
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


        # Load training state
        self.global_step = checkpoint.get("global_step", 0)

        tplr.logger.info(f"Resumed training at global step {self.global_step}")

    def cleanup(self):
        """Clean up resources."""
        if self.world_size > 1:
            destroy_process_group()

        if self.wandb is not None:
            self.wandb.finish()

        # Close timing logger
        if self.global_rank == 0 and self.timing_logger is not None:
            for handler in self.timing_logger.handlers:
                handler.close()
                self.timing_logger.removeHandler(handler)


def main():
    """Entry point."""
    trainer = DistributedLLMTrainer()

    try:
        trainer.run()
    except KeyboardInterrupt:
        tplr.logger.info("Training interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Error during training: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
