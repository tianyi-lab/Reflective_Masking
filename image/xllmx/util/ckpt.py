import json
import logging
import os
import shutil
from typing import Dict, Optional, Sequence

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

logger = logging.getLogger(__name__)


def split_ckpt_str_into_epoch_iter(ckpt_str: str):
    parts = ckpt_str.split("-")
    epoch = int(parts[0].replace("epoch", ""))
    if len(parts) == 2:
        iter_part = int(parts[1].replace("iter", ""))
    else:
        iter_part = None
    return epoch, iter_part


def remove_early_ckpts(out_dir, max_keep=2, exclude_names: Optional[Sequence[str]] = None):
    if max_keep < 0:
        return
    if not os.path.isdir(out_dir):
        return

    def ckpt_sort_key(s):
        epoch, iteration = split_ckpt_str_into_epoch_iter(s)
        if iteration is None:
            iteration = float("inf")
        return epoch, iteration

    excluded = set(exclude_names or [])
    existing_checkpoints = [_ for _ in os.listdir(out_dir) if "epoch" in _ and _ not in excluded]
    existing_checkpoints = sorted(existing_checkpoints, key=ckpt_sort_key, reverse=True)

    for dir_to_remove in existing_checkpoints[max_keep:]:
        dir_to_remove = os.path.join(out_dir, dir_to_remove)
        shutil.rmtree(dir_to_remove)
        logger.info(f"Deleted {dir_to_remove}")


def save(
    output_dir,
    is_main_process,
    model: FSDP,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer=None,
    args=None,
    epoch=None,
    iteration=None,
    additional_rank_common: Optional[Dict] = None,
    additional_rank_specific: Optional[Dict] = None,
    max_keep=2,
    delete_old_before_save=False,
):
    save_name = f"epoch{epoch}"
    if iteration is not None:
        save_name += f"-iter{iteration}"
    save_dir = os.path.join(output_dir, save_name)

    if delete_old_before_save:
        if is_main_process:
            remove_early_ckpts(
                output_dir,
                max_keep=max_keep - 1,
                exclude_names=[save_name],
            )
        dist.barrier()

    os.makedirs(save_dir, exist_ok=True)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        def _save_model():
            save_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "tf32": torch.float,
            }[
                args.precision
            ]
            if getattr(args, "only_save_trainable", False):
                model_trainable_params = model.get_trainable_params()
                model_trainable_params = [
                    ".".join([_ for _ in key.split(".") if not _.startswith("_")])
                    for key in model_trainable_params.keys()
                ]
                consolidated_model_state_dict = {
                    key: val.to(save_dtype) for key, val in model.state_dict().items() if key in model_trainable_params
                }
            else:
                consolidated_model_state_dict = {key: val.to(save_dtype) for key, val in model.state_dict().items()}

            if is_main_process:
                model.save_pretrained(save_dir, state_dict=consolidated_model_state_dict)

        _save_model()
        logger.info("model saved")

    if optimizer is not None:
        with FSDP.state_dict_type(
            model,
            StateDictType.LOCAL_STATE_DICT,
        ):
            opt_path = os.path.join(
                save_dir,
                f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth",
            )
            torch.save(optimizer.state_dict(), opt_path)
            logger.info("optimizer saved")
    else:
        logger.info("optimizer is None, skip saving")

    if additional_rank_specific is not None:
        torch.save(
            additional_rank_specific,
            os.path.join(save_dir, f"additional.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"),
        )
        logger.info(f"additional_rank_specific {list(additional_rank_specific.keys())} saved")

    if not is_main_process:
        dist.barrier()
        return

    if tokenizer is not None:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_dir)
            logger.info("tokenizer saved via save_pretrained")
        elif hasattr(tokenizer, "save"):
            tokenizer.save(save_dir)
            logger.info("tokenizer saved via save")
        else:
            logger.warning("tokenizer does not provide save_pretrained/save, skip saving")
    else:
        logger.info("tokenizer is None, skip saving")

    if args is not None:
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("args saved")
    else:
        logger.info("args is None, skip saving")

    if additional_rank_common is not None:
        torch.save(additional_rank_common, os.path.join(save_dir, "additional_rank_common.pth"))
        logger.info(f"additional_resources {list(additional_rank_common.keys())} saved")

    if not delete_old_before_save:
        remove_early_ckpts(output_dir, max_keep=max_keep if max_keep > 0 else -1)

    dist.barrier()
    return
