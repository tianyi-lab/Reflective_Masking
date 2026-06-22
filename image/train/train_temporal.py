
from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path

TEMPORAL_ROOT = Path(__file__).resolve().parents[1]
if str(TEMPORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPORAL_ROOT))

import numpy as np
import torch
from fairscale.nn.model_parallel import initialize as fs_init
from torch.utils.data import Dataset
from transformers import AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from config import PROMPT_TEMPLATES, SPECIAL_TOKENS
from model.modeling_temporal import TemporalTrainingModel
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.data.sampler import FinetuneDistSampler
from xllmx.solvers.finetune.finetune import FinetuneSolverBase
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc


REQUIRED_CACHE_FILES = {"before_ids.npy", "after_ids.npy", "instruction.txt"}
HISTORY_FILE = "history_tokens.npz"
MASK_TOKEN = int(SPECIAL_TOKENS["mask_token"])
NEWLINE_TOKEN = int(SPECIAL_TOKENS["newline_token"])
DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATES["image_editing"]


# ---------------------------------------------------------------------------
# Per-frame supervision targets
# ---------------------------------------------------------------------------


def _build_labels_from_current_frame(
    *,
    current_tokens: np.ndarray,
    final_tokens: np.ndarray,
    prefix_len: int,
    image_hw: tuple[int, int],
    keep_loss_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_h, image_w = image_hw
    img_begin = int(prefix_len) + 2
    img_len = int(image_h) * (int(image_w) + 1)
    img_end = img_begin + img_len

    current_img = np.asarray(current_tokens[img_begin:img_end], dtype=np.int64)
    final_img = np.asarray(final_tokens[img_begin:img_end], dtype=np.int64)
    img_labels = np.asarray(final_img, dtype=np.int64).copy()
    img_labels[img_labels == NEWLINE_TOKEN] = -100

    transition_mask = (
        (current_img != final_img)
        & (current_img != MASK_TOKEN)
        & (final_img != NEWLINE_TOKEN)
    )
    reveal_mask = (
        (current_img == MASK_TOKEN)
        & (final_img != NEWLINE_TOKEN)
    )
    img_labels[transition_mask] = MASK_TOKEN

    all_labels = np.full((int(len(current_tokens)),), -100, dtype=np.int64)
    all_labels[img_begin:img_end] = img_labels

    img_weights = np.zeros_like(img_labels, dtype=np.float32)
    valid_mask = img_labels != -100
    edit_mask = transition_mask | reveal_mask
    keep_mask = valid_mask & ~edit_mask
    img_weights[edit_mask] = 1.0
    img_weights[keep_mask] = float(keep_loss_weight)

    all_loss_weights = np.zeros((int(len(current_tokens)),), dtype=np.float32)
    all_loss_weights[img_begin:img_end] = img_weights
    return all_labels, all_loss_weights


# ---------------------------------------------------------------------------
# Item processor: samples a current step from the offline history trajectory
# ---------------------------------------------------------------------------


class TemporalItemProcessor(ItemProcessorBase):
    def __init__(
        self,
        tokenizer,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        keep_loss_weight: float = 1.0,
        overfit_seed: int | None = None,
    ):
        del tokenizer
        self.system_prompt = str(system_prompt)
        self.keep_loss_weight = float(keep_loss_weight)
        self.overfit_seed = None if overfit_seed is None else int(overfit_seed)

    def set_epoch(self, epoch, start_sample_offset=0):
        del epoch, start_sample_offset

    def _resolve_full_history(self, data_item: dict) -> np.ndarray:
        if "history_token_ids" not in data_item:
            raise ValueError("history_token_ids is required for temporal training")

        history_np = np.asarray(data_item["history_token_ids"], dtype=np.int64)
        if history_np.ndim != 2:
            raise ValueError(f"history_token_ids must be 2D, got shape {history_np.shape}")

        if "max_temporal_steps" not in data_item:
            raise ValueError("history npz is missing max_temporal_steps")
        max_temporal_steps = int(np.asarray(data_item["max_temporal_steps"], dtype=np.int64).item())
        expected_len = max_temporal_steps + 1
        if history_np.shape[0] != expected_len:
            raise ValueError(
                "history_token_ids must store the full trajectory: "
                f"got {history_np.shape[0]} frames, expected {expected_len} "
                "(max_temporal_steps + 1)."
            )

        if "step_t" in data_item:
            saved_step_t = int(np.asarray(data_item["step_t"], dtype=np.int64).item())
            if saved_step_t != max_temporal_steps:
                raise ValueError(
                    "history npz stores a single sampled step "
                    f"(step_t={saved_step_t}); the full trajectory is required "
                    f"(expected step_t == max_temporal_steps == {max_temporal_steps})."
                )
        return np.ascontiguousarray(history_np)

    def process_item(self, data_item: dict, training_mode=False):
        del training_mode

        if self.overfit_seed is not None:
            np.random.seed(int(self.overfit_seed))

        full_history = self._resolve_full_history(data_item)
        total_steps = int(full_history.shape[0] - 1)
        current_step = int(np.random.randint(0, total_steps + 1))

        current_tokens = np.asarray(full_history[current_step], dtype=np.int64)
        final_tokens = np.asarray(full_history[-1], dtype=np.int64)
        prefix_len = int(np.asarray(data_item["prefix_len"], dtype=np.int64).item())
        if "image_hw" in data_item:
            image_hw = tuple(int(v) for v in np.asarray(data_item["image_hw"], dtype=np.int64).tolist())
        else:
            before_ids = np.asarray(data_item["before_ids"], dtype=np.int64)
            image_hw = tuple(int(v) for v in before_ids.shape)

        all_labels, all_loss_weights = _build_labels_from_current_frame(
            current_tokens=current_tokens,
            final_tokens=final_tokens,
            prefix_len=prefix_len,
            image_hw=image_hw,
            keep_loss_weight=self.keep_loss_weight,
        )
        img_begin = int(prefix_len) + 2
        img_len = int(image_hw[0]) * (int(image_hw[1]) + 1)
        img_end = img_begin + img_len
        history_token_mask = np.zeros((int(len(current_tokens)),), dtype=bool)
        history_token_mask[img_begin:img_end] = current_tokens[img_begin:img_end] != NEWLINE_TOKEN
        history_list = [
            np.asarray(full_history[idx], dtype=np.int64).tolist()
            for idx in range(current_step - 1, -1, -1)
        ]
        history_distances = list(range(1, current_step + 1))
        meta = {
            "mode": "offline_full_history",
            "current_step": current_step,
            "total_steps": total_steps,
            "history_len": len(history_list),
        }
        return (
            current_tokens.tolist(),
            all_labels.tolist(),
            all_loss_weights.tolist(),
            history_list,
            history_distances,
            history_token_mask.tolist(),
            current_step,
            meta,
        )

    def predict_item_token_length(self, data_item: dict) -> int:
        if "before_ids" in data_item:
            h, w = data_item["before_ids"].shape
            return 64 + 1 + int(h) * (int(w) + 1) + 1
        if "history_token_ids" in data_item:
            return int(np.asarray(data_item["history_token_ids"]).shape[-1])
        return 1


# ---------------------------------------------------------------------------
# Dataset: one directory per sample under <cache_root>/<source>/<sample>/
# ---------------------------------------------------------------------------


class DirectoryCacheDataset(Dataset):

    def __init__(
        self,
        cache_root: str,
        item_processor,
        sources: list[str] | None = None,
        edit_indices_field: str = "edit_indices",
        history_root: str | None = None,
        max_samples_per_source: int = 0,
        require_history: bool = True,
    ):
        self.item_processor = item_processor
        self.edit_indices_field = str(edit_indices_field)
        root = Path(cache_root)
        if not root.is_dir():
            raise FileNotFoundError(f"cache_root does not exist: {cache_root!r}")
        self.cache_root = root
        self.history_root = None if history_root is None else Path(history_root)
        if self.history_root is not None and not self.history_root.is_dir():
            raise FileNotFoundError(f"history_root does not exist: {history_root!r}")

        if sources:
            source_names = list(sources)
        else:
            source_names = sorted(
                d.name for d in root.iterdir() if d.is_dir()
            )

        self.sample_dirs: list[Path] = []
        for src in source_names:
            src_dir = root / src
            if not src_dir.is_dir():
                continue
            count = 0
            for p in sorted(src_dir.iterdir()):
                if not p.is_dir():
                    continue
                if not self._has_required_files(p, require_history):
                    continue
                self.sample_dirs.append(p)
                count += 1
                if 0 < max_samples_per_source <= count:
                    break

        if not self.sample_dirs:
            raise FileNotFoundError(
                f"No valid samples found in {cache_root!r} "
                f"(sources={sources}, require_history={require_history})"
            )

        first = self._read_sample(self.sample_dirs[0])
        H, W = first["before_ids"].shape
        estimated_len = 64 + 1 + H * (W + 1) + 1

        self.meta_collection = [
            {
                "type": "editing",
                "len": len(self.sample_dirs),
                "item_len_list": [estimated_len] * len(self.sample_dirs),
            }
        ]

    def _has_required_files(self, sample_dir: Path, require_history: bool) -> bool:
        for name in REQUIRED_CACHE_FILES:
            if not (sample_dir / name).is_file():
                return False
        if not (sample_dir / f"{self.edit_indices_field}.npy").is_file():
            return False
        if require_history and not self._history_path_for_sample(sample_dir).is_file():
            return False
        return True

    def _history_path_for_sample(self, sample_dir: Path) -> Path:
        if self.history_root is None:
            return sample_dir / HISTORY_FILE
        rel_sample_dir = sample_dir.relative_to(self.cache_root)
        return self.history_root / rel_sample_dir / HISTORY_FILE

    def _read_sample(self, sample_dir: Path) -> dict:
        data = {
            "before_ids": np.load(sample_dir / "before_ids.npy", allow_pickle=False),
            "after_ids": np.load(sample_dir / "after_ids.npy", allow_pickle=False),
            "edit_indices": np.load(
                sample_dir / f"{self.edit_indices_field}.npy", allow_pickle=False
            ),
            "instruction": (sample_dir / "instruction.txt")
            .read_text(encoding="utf-8", errors="replace")
            .strip(),
        }
        hist_path = self._history_path_for_sample(sample_dir)
        if hist_path.is_file():
            npz = np.load(hist_path, allow_pickle=False)
            for key in npz.files:
                data[key] = npz[key]
        return data

    def __len__(self):
        return len(self.sample_dirs)

    def set_epoch(self, epoch, start_sample_offset=0):
        if hasattr(self.item_processor, "set_epoch"):
            self.item_processor.set_epoch(epoch, start_sample_offset)

    def __getitem__(self, idx):
        data_item = self._read_sample(self.sample_dirs[idx])
        return self.item_processor.process_item(data_item, training_mode=True)


class TemporalEditingSolver(FinetuneSolverBase):
    def __init__(self, args):
        if float(getattr(args, "save_epoch_percent", 0.0)) <= 0:
            args.save_intermediate_checkpoints = False
        if float(getattr(args, "temporal_history_decay", 0.0)) < 0.0:
            raise ValueError("--temporal-history-decay must be >= 0")
        if float(getattr(args, "temporal_history_scale", 1.0)) < 0.0:
            raise ValueError("--temporal-history-scale must be >= 0")
        valid_history_modes = {"post_rms", "gated_post_rms", "residual_norm", "gated_residual"}
        if str(getattr(args, "temporal_history_mode", "post_rms")) not in valid_history_modes:
            raise ValueError("--temporal-history-mode must be post_rms, gated_post_rms, residual_norm, or gated_residual")
        if float(getattr(args, "temporal_history_gate_min", 0.0)) < 0.0:
            raise ValueError("--temporal-history-gate-min must be >= 0")
        if float(getattr(args, "temporal_history_gate_max", 0.0)) < float(getattr(args, "temporal_history_gate_min", 0.0)):
            raise ValueError("--temporal-history-gate-max must be >= --temporal-history-gate-min")
        super().__init__(args)

    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--cache-root", type=str, default=None,
                            help="VQ cache directory with per-source/per-sample editing data")
        parser.add_argument("--history-root", type=str, default=None,
                            help="Mirrored root that stores preprocessed history files separately from cache-root")
        parser.add_argument("--sources", type=str, default=None,
                            help="Comma-separated source names to use (default: all in cache-root)")
        parser.add_argument("--max_seq_len", type=int, default=2048)
        parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
        parser.add_argument("--keep-loss-weight", type=float, default=1.0,
                            help="CE weight for non-edit and already-correct image tokens.")
        parser.add_argument("--temporal-history-decay", type=float, default=0.2,
                            help="Exponential decay for history residuals: exp(-decay * (distance - 1)).")
        parser.add_argument("--temporal-history-scale", type=float, default=1.0,
                            help="Scale applied after history residual normalization.")
        parser.add_argument("--temporal-history-mode", type=str, default="post_rms",
                            choices=["post_rms", "gated_post_rms", "residual_norm", "gated_residual"],
                            help="post_rms matches the old stable accumulated-input normalization; gated_post_rms adds a bounded gate before that normalization; residual_norm matches train_dllm; gated_residual gates the normalized residual path.")
        parser.add_argument("--temporal-history-gate-min", type=float, default=0.002,
                            help="Lower bound for gated modes' learnable history gate.")
        parser.add_argument("--temporal-history-gate-max", type=float, default=0.04,
                            help="Upper bound for gated modes' learnable history gate.")
        parser.add_argument("--temporal-distance-encoding", type=str, default="rope", choices=["rope"],
                            help="Temporal distance encoding for history residuals.")
        parser.add_argument("--debug-nonfinite", action="store_true",
                            help="Print tensor finite stats when forward/loss produces non-finite values.")
        parser.add_argument("--overfit-seed", type=int, default=None,
                            help="If set, deterministic overfit mode for current-step sampling.")
        parser.add_argument("--save-epoch-percent", type=float, default=10.0,
                            help="Save an intermediate checkpoint every N percent of each epoch; <=0 disables intermediate checkpoints.")
        parser.add_argument("--progress-bar", action="store_true", default=True,
                            help="Show a tqdm progress bar on global rank 0.")
        parser.add_argument("--no-progress-bar", action="store_false", dest="progress_bar",
                            help="Disable the tqdm progress bar.")
        parser.set_defaults(num_workers=0)
        return parser

    def _resolve_pretrained_sources(self, init_from: str) -> tuple[str, str]:
        resume_path = str(getattr(self.args, "resume_path", "") or "").strip()
        if not resume_path:
            source = str(init_from)
            return source, source

        tokenizer_source = str(getattr(self.args, "init_from", "") or init_from).strip()
        resume_args_path = Path(resume_path) / "args.json"
        if resume_args_path.is_file():
            try:
                resume_args = json.loads(resume_args_path.read_text(encoding="utf-8"))
                checkpoint_init_from = str(resume_args.get("init_from") or "").strip()
                if checkpoint_init_from:
                    if tokenizer_source and tokenizer_source != checkpoint_init_from:
                        self.logger.warning(
                            "[model] resume checkpoint init_from=%s differs from current init_from=%s; "
                            "use checkpoint init_from for tokenizer/code loading",
                            checkpoint_init_from,
                            tokenizer_source,
                        )
                    tokenizer_source = checkpoint_init_from
            except Exception as exc:
                self.logger.warning(
                    "[model] failed to read %s for resume source resolution: %s",
                    resume_args_path,
                    exc,
                )

        if not tokenizer_source:
            tokenizer_source = str(init_from)
        return tokenizer_source, resume_path

    def _model_func(self, init_from):
        tokenizer_source, model_source = self._resolve_pretrained_sources(init_from)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

        cfg = TemporalTrainingModel.config_class.from_pretrained(model_source)
        cfg.init_device = "cpu"
        model = TemporalTrainingModel.from_pretrained(
            model_source,
            config=cfg,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        model.model.set_activation_checkpointing("whole_layer")
        return model, tokenizer

    def _make_and_save_starting_point(self, save_path):
        tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)
        config = TemporalTrainingModel.config_class.from_pretrained("Alpha-VLLM/Lumina-DiMOO")
        model = TemporalTrainingModel(config)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    def _item_processor_func(self):
        return TemporalItemProcessor(
            tokenizer=self.tokenizer,
            system_prompt=self.args.system_prompt,
            keep_loss_weight=float(self.args.keep_loss_weight),
            overfit_seed=self.args.overfit_seed,
        )

    def build_data(self):
        eff_batch_size = self.args.batch_size * self.args.accum_iter * fs_init.get_data_parallel_world_size()
        self.logger.info("effective batch size: %d", eff_batch_size)

        processor = self._item_processor_func()

        if not self.args.cache_root:
            raise ValueError("--cache-root must be provided")

        sources = None
        if self.args.sources:
            sources = [s.strip() for s in self.args.sources.split(",") if s.strip()]
        dataset_train = DirectoryCacheDataset(
            self.args.cache_root,
            processor,
            sources=sources,
            history_root=self.args.history_root,
            require_history=True,
        )
        self.logger.info(
            "Using DirectoryCacheDataset: %d samples from %s (history_root=%s)",
            len(dataset_train),
            self.args.cache_root,
            self.args.history_root,
        )

        sampler_train = FinetuneDistSampler(
            dataset_train,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            batch_size=self.args.batch_size,
            acc_grad=self.args.accum_iter,
            seed=self.args.seed,
            length_clustering=self.args.length_clustering,
        )

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            sampler=sampler_train,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=True,
        )
        return dataset_train, sampler_train, dataloader_train

    def train_one_epoch(
        self,
        epoch: int,
        start_iter: int,
        log_writer=None,
        metric_logger=None,
    ):
        start_sample_offset = start_iter * self.args.batch_size
        self.dataset_train.set_epoch(epoch, start_sample_offset)

        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = f"Epoch: [{epoch}]"
        print_freq = 10
        accum_iter = self.args.accum_iter
        accum_counter = 0
        total_steps = len(self.dataloader_train)
        save_epoch_percent = float(getattr(self.args, "save_epoch_percent", 0.0))
        next_save_percent = save_epoch_percent
        if save_epoch_percent > 0 and start_iter > 0:
            start_percent = start_iter * 100.0 / total_steps
            while next_save_percent <= start_percent:
                next_save_percent += save_epoch_percent

        train_iter = metric_logger.log_every(
            self.dataloader_train,
            print_freq,
            header,
            start_iter,
            self.args.batch_size * fs_init.get_data_parallel_world_size(),
        )
        progress_bar = None
        if self.global_rank == 0 and self.args.progress_bar:
            if tqdm is None:
                self.logger.warning("tqdm is not installed; progress bar disabled")
            else:
                progress_bar = tqdm(
                    train_iter,
                    total=total_steps,
                    initial=start_iter,
                    desc=header,
                    dynamic_ncols=True,
                    leave=True,
                    ascii=True,
                    mininterval=1.0,
                )
                train_iter = progress_bar

        self.optimizer.zero_grad()
        for data_iter_step, batch_data in enumerate(train_iter, start=start_iter):
            accum_counter = (accum_counter + 1) % accum_iter
            is_gradient_accumulation_boundary = accum_counter == 0

            if len(batch_data) == 8:
                (
                    examples,
                    labels,
                    loss_weights,
                    history_lists,
                    history_distance_lists,
                    history_token_mask_lists,
                    current_step_list,
                    online_meta_list,
                ) = batch_data
            elif len(batch_data) == 7:
                (
                    examples,
                    labels,
                    loss_weights,
                    history_lists,
                    history_distance_lists,
                    current_step_list,
                    online_meta_list,
                ) = batch_data
                history_token_mask_lists = None
            else:
                raise ValueError(f"Unexpected processor batch tuple length: {len(batch_data)}")
            if is_gradient_accumulation_boundary or data_iter_step == start_iter:
                lr_sched.adjust_learning_rate_epoch(
                    self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
                )

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                total_loss = self.model(
                    examples,
                    labels=labels,
                    loss_weights=loss_weights,
                    history_token_ids_list=history_lists,
                    history_distance_lists=history_distance_lists,
                    history_token_mask_list=history_token_mask_lists,
                    temporal_history_decay=float(self.args.temporal_history_decay),
                    temporal_history_scale=float(self.args.temporal_history_scale),
                    temporal_history_mode=str(self.args.temporal_history_mode),
                    temporal_history_gate_min=float(self.args.temporal_history_gate_min),
                    temporal_history_gate_max=float(self.args.temporal_history_gate_max),
                    debug_nonfinite=bool(self.args.debug_nonfinite),
                )

            loss_value = total_loss.item()
            if not math.isfinite(loss_value):
                weight_sums = [float(sum(weights)) for weights in loss_weights]
                history_lengths = [len(history) for history in history_lists]
                self.logger.error(
                    "Loss is %s, stopping training. current_steps=%s history_lengths=%s loss_weight_sums=%s",
                    loss_value,
                    list(current_step_list),
                    history_lengths,
                    weight_sums,
                )
                sys.exit(1)

            effective_loss = total_loss / accum_iter
            with (
                self.model.no_sync()
                if self.args.data_parallel in ["sdp", "hsdp"] and not is_gradient_accumulation_boundary
                else contextlib.nullcontext()
            ):
                effective_loss.backward()

            grad_norm_value = None
            if is_gradient_accumulation_boundary:
                grad_norm = self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                metric_logger.update(grad_norm=grad_norm)
                grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            metric_logger.update(
                closs=loss_value,
            )
            history_gate_value = self._history_gate_value()
            if history_gate_value is not None:
                metric_logger.update(history_gate=history_gate_value)
            lr = self.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            if progress_bar is not None:
                postfix = {
                    "loss": f"{loss_value:.4f}",
                    "lr": f"{lr:.2e}",
                }
                if grad_norm_value is not None:
                    postfix["grad"] = f"{grad_norm_value:.2f}"
                if history_gate_value is not None:
                    postfix["gate"] = f"{history_gate_value:.4f}"
                progress_bar.set_postfix(postfix)

            for metric_name, metric in metric_logger.meters.items():
                metric_value = util.dist.all_reduce_mean(metric.value)
                if log_writer is not None:
                    log_writer.add_scalar(metric_name, metric_value, data_iter_step + len(self.dataloader_train) * epoch)

            if self.args.save_intermediate_checkpoints and save_epoch_percent > 0:
                progress_percent = (data_iter_step + 1) * 100.0 / total_steps
                should_save_progress = (
                    is_gradient_accumulation_boundary
                    and progress_percent >= next_save_percent
                    and next_save_percent < 100.0
                )
                if should_save_progress:
                    while next_save_percent <= progress_percent:
                        next_save_percent += save_epoch_percent
                    util.ckpt.save(
                        self.args.output_dir,
                        self.global_rank == 0,
                        self.model,
                        self.optimizer if self.args.save_optimizer else None,
                        None,
                        self.args,
                        epoch=epoch,
                        iteration=data_iter_step,
                        additional_rank_specific={"metric_logger": metric_logger},
                        max_keep=self.args.ckpt_max_keep,
                    )

        if progress_bar is not None:
            progress_bar.close()
        metric_logger.synchronize_between_processes()
        self.logger.info("Averaged stats:\n%s", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def _history_gate_value(self):
        if str(getattr(self.args, "temporal_history_mode", "")) not in {"gated_post_rms", "gated_residual"}:
            return None
        model_root = self.model.module if hasattr(self.model, "module") else self.model
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        local_value = torch.zeros((), device=device, dtype=torch.float32)
        local_count = torch.zeros((), device=device, dtype=torch.float32)

        last_forward_gate = getattr(model_root, "_last_temporal_history_gate", None)
        if last_forward_gate is not None:
            with torch.no_grad():
                local_value.copy_(last_forward_gate.detach().to(device=device, dtype=torch.float32))
                local_count.fill_(1.0)
        else:
            gate_logit = getattr(model_root, "history_gate_logit", None)
            if gate_logit is None:
                return None
            gate_min = float(self.args.temporal_history_gate_min)
            gate_max = float(self.args.temporal_history_gate_max)
            with torch.no_grad():
                local_gate_logit = gate_logit.detach().float()
                if local_gate_logit.numel() > 0:
                    value = gate_min + (gate_max - gate_min) * torch.sigmoid(local_gate_logit.reshape(-1)[0])
                    local_value.copy_(value.to(device=device, dtype=torch.float32))
                    local_count.fill_(1.0)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                local_value,
                op=torch.distributed.ReduceOp.SUM,
                group=fs_init.get_data_parallel_group(),
            )
            torch.distributed.all_reduce(
                local_count,
                op=torch.distributed.ReduceOp.SUM,
                group=fs_init.get_data_parallel_group(),
            )

        if float(local_count.item()) <= 0.0:
            return None
        return float((local_value / local_count.clamp_min(1.0)).item())


if __name__ == "__main__":
    args = TemporalEditingSolver.get_args_parser().parse_args()
    solver = TemporalEditingSolver(args)
    solver.run()
