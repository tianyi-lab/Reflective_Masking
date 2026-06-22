# Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models

Official code for **Reflective Masking (RM)**: post-training a mask diffusion model to
revise its own output instead of regenerating it.

[arXiv:2606.16700](https://arxiv.org/abs/2606.16700) |
[Project page](https://zhangyanming-cs.github.io/Multi-Turn_RM/) |
[Code](https://github.com/tianyi-lab/Reflective_Masking.git)

---

## Overview

Mask Diffusion Models (MDMs) make one-shot predictions and cannot revise what they already
produced. Reflective Masking (RM) post-trains an MDM into a multi-turn reviser: instead of
regenerating an answer, it revisits its previous output and picks one action per token —
**reveal** (commit), **reflectively mask** (re-mask a wrong token for a later turn), or
**keep**. **History Reference (HR)** is a parameter-free mechanism that feeds the model's
earlier denoising states back in during revision.

Two independent parts:

- `image/`: image editing on the Lumina-DiMOO backbone.
- `text/`: math reasoning on the LLaDA-8B-Instruct backbone.

Both build a synthetic revision *trajectory* offline and SFT the model to predict the
per-token action at a sampled step. The `text/` model additionally uses HR (earlier steps are
fed back in). The released `image/` model is trained RM-only — the HR code is present but was
disabled (history weight 0) for these results.

## Release status
- [√] **Training/Inference Code**
- [ ] **Finetuned weights** — `text/` (RM+HR) and `image/` (RM) checkpoints.
- [ ] **Data-preprocessing code** — scripts that build the datasets from raw Hendrycks MATH / ImgEdit.
- [ ] **Processed data** — the datasets used in the paper.

---

## `text/` — Text Reasoning

- `examples/llada/synthetic_revision_history_sft.py` — SFT on the revision-history dataset.
- `dllm/pipelines/llada/infer_history.py` — single-prompt inference.

### Setup & downloads

```bash
cd text
pip install -e .

# Base model (default: CKPT/LLaDA-8B-Instruct)
hf download GSAI-ML/LLaDA-8B-Instruct --local-dir CKPT/LLaDA-8B-Instruct
# Raw dataset
hf download --repo-type dataset EleutherAI/hendrycks_math --local-dir data/hendrycks_math
```

### Data

The SFT script loads a preprocessed `datasets` dataset (`dllm/data/synthetic_revision_history.py`).
The two key columns are `input_ids` (the gold prompt + answer) and `current_input_ids` (the
current revision state, where each answer token is the correct token, a *wrong* token, or
`[MASK]`); `history_input_ids` + `history_distances` hold the earlier revision states. The
trainer derives the action labels by comparing the two: wrong → reflectively mask, `[MASK]` →
reveal, correct → keep.

- **Wrong tokens:** corrupt a fraction of answer tokens, sampling each replacement from the
  frozen base model's top-`k` predictions (plausible, model-confusable errors).
- **History:** a short trajectory of successively-revised states; for a sample at step `t`,
  the earlier states are fed back as history.

### Usage

```bash
# from text/  — SFT (accelerate; configs in scripts/accelerate_configs/)
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml \
  examples/llada/synthetic_revision_history_sft.py [args...]

# single-prompt inference
python dllm/pipelines/llada/infer_history.py \
  --pretrained output/train/<run>/checkpoint-N --prompt "<math problem>"
```

---

## `image/` — Image Editing

Two-phase mask-then-unmask decoding that localizes edits to the instructed region.

- `train/train_temporal.py` — RM finetune of Lumina-DiMOO (HR disabled for the release).
- `inference/mask_then_unmask_temporal.py` — two-phase inference on a cached source image.

### Downloads

```bash
# from image/ — base model + VAE + tokenizer
hf download Alpha-VLLM/Lumina-DiMOO --local-dir CKPT/Lumina-DiMOO
```

### Data

Training reads a directory cache under `image/DATA/<source>/<sample>/`, each sample holding
`before_ids.npy` / `after_ids.npy` (source / target VQ token grids), `instruction.txt`,
`edit_indices.npy`, and `history_tokens.npz`. The `.npz` stores `history_token_ids`
(`[T+1, L]`, the full mask-then-unmask trajectory from most-corrupted frame `0` to the target
frame `T`) plus `prefix_len` and `image_hw`.

Each intermediate frame of that trajectory mixes correct tokens, wrong tokens (differ from the
target), and `[MASK]`. Training samples a step `t` and derives the per-token action labels
against the target frame. (The image model is trained RM-only; the HR history-injection code
ships but was disabled — `--temporal-history-scale 0` — for the released results.)

### Usage

Inference runs on a prebuilt sample directory (`--sample-dir`); `--prompt` only overrides the
instruction text.

```bash
# from image/  — RM finetune
torchrun --nproc_per_node=<N> train/train_temporal.py [args...]

# two-phase inference
python inference/mask_then_unmask_temporal.py \
  --checkpoint output/temporal_train/.../epochN \
  --sample-dir DATA/ImgEdit_test_cache/<source>/<sample> \
  --tokenizer-path CKPT/Lumina-DiMOO --vae-ckpt CKPT/Lumina-DiMOO
```

Pass `--help` to each entry point for the full argument list.

---

## Citation

```bibtex
@misc{zhang2026multiturn,
  title={Multi-Turn Reflective Masking Elicits Reasoning in Mask Diffusion Models},
  author={Zhang, Yanming and Bian, Yihan and Qi, Jingyuan and Yao, Yuguang and Huang, Lifu and Zhou, Tianyi},
  year={2026},
  eprint={2606.16700},
  archivePrefix={arXiv}
}
```

## Acknowledgements

- `image/` is adapted from [Lumina-DiMOO](https://github.com/Alpha-VLLM/Lumina-DiMOO).
- `text/` is adapted from [dllm](https://github.com/ZHZisZZ/dllm).
