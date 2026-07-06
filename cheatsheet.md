# GR00T N1.6 — SO-101 Bimanual Fine-tune Cheatsheet

End-to-end commands for post-training GR00T N1.6 on the SO-101 **bimanual** setup
with a custom Hugging Face dataset, on an NVIDIA Brev H100 instance.

| Thing | Value |
|---|---|
| Base model | `nvidia/GR00T-N1.6-3B` |
| Dataset (HF) | `ChihHanShen/bimanual-so101-pickvials` (public, LeRobot **v3.0**) |
| Dataset size | 16 episodes / 24,854 frames / 1 task / 30 fps / 12-dim bimanual |
| Local dataset path | `$HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials` |
| Modality config | `examples/SO101_bimanual/so101_bimanual_config.py` |
| modality.json | `examples/SO101_bimanual/modality.json` |
| Embodiment tag | `NEW_EMBODIMENT` |
| Repo/branch | `maxshen1212/Isaac-GR00T` @ `n1.6-graphen` |

---

## 0. One-time setup (Brev instance)

Run the setup script (installs deps, clones repo, downloads + converts dataset, verifies GPU):

```bash
bash NVIDIA_Brev_script.sh
```

Then activate the env for everything below:

```bash
cd $HOME/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
```

---

## 1. Dataset prep (only if doing it manually)

The setup script already does this. To redo by hand:

```bash
# download
uv run hf download ChihHanShen/bimanual-so101-pickvials \
  --repo-type dataset --local-dir $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials

# convert LeRobot v3.0 -> v2.1 (GR00T's loader only reads v2.1; original backed up as *_v30)
GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id bimanual-so101-pickvials \
  --root $HOME/Isaac-GR00T/datasets

# install the bimanual modality.json (required; not shipped in the dataset)
cp examples/SO101_bimanual/modality.json \
   $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/modality.json
```

---

## 2. Verify environment

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print('GR00T imports OK')"
python -c "import json; print(json.load(open('$HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/info.json'))['codebase_version'])"  # -> v2.1
```

---

## 3. Smoke test (validate the pipeline, ~2 min)

100 steps, no wandb — just proves data load -> forward/backward -> checkpoint works.

```bash
MAX_STEPS=100 SAVE_STEPS=100 USE_WANDB=0 GLOBAL_BATCH_SIZE=32 \
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so101_smoketest
```

Pass = runs to step 100, finite decreasing loss, `/tmp/so101_smoketest/checkpoint-100/` exists.

---

## 4. Real training run (~6.5 epochs)

```bash
wandb login    # once — key from https://wandb.ai/authorize

MAX_STEPS=5000 SAVE_STEPS=500 GLOBAL_BATCH_SIZE=32 \
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so101_bimanual_finetune \
  --wandb-project so101-bimanual \
  --experiment-name pickvials-run1
```

Saves checkpoints at steps 3000/3500/4000/4500/5000 (`save_total_limit=5` keeps the last 5).

---

## 5. Pick the best checkpoint (open-loop eval)

```bash
for ckpt in 3000 3500 4000 4500 5000; do
  echo "=== checkpoint-$ckpt ==="
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /tmp/so101_bimanual_finetune/checkpoint-$ckpt \
    --traj-ids 0 1 2 --action-horizon 16 --steps 400
done
```

Pick the checkpoint with lowest action MSE where the wandb loss had flattened (not collapsed).

---

## 6. Upload the trained model to Hugging Face

`/tmp` is wiped when the instance stops — **upload before shutting down.**

```bash
hf auth login    # once — WRITE token from https://huggingface.co/settings/tokens

ls /tmp/so101_bimanual_finetune/     # pick a checkpoint that exists
uv run hf upload \
  ChihHanShen/gr00t-n1.6-so101-bimanual-pickvials \
  /tmp/so101_bimanual_finetune/checkpoint-5000 \
  . \
  --repo-type model      # add --private to keep it unlisted
```

---

## Hyperparameters — how to tune

Master formula: **`epochs = max_steps × global_batch_size / num_frames`**.
Set `max_steps` from a target epoch count (**5–8** for a small single-task set), not a fixed number.
For 24,854 frames at batch 32: ~770 steps/epoch → `MAX_STEPS=5000` ≈ 6.5 epochs.
(N1.6's default `MAX_STEPS=10000` assumes a ~150k-frame dataset — too many epochs for this one.)

**Where each knob lives:**
- **Env vars** (prefix the command): `MAX_STEPS`, `SAVE_STEPS`, `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `USE_WANDB`, `DATALOADER_NUM_WORKERS`, `SHARD_SIZE`, `NUM_SHARDS_PER_EPOCH`, `EPISODE_SAMPLING_RATE`.
- **finetune.sh flags**: `--base-model-path`, `--dataset-path`, `--modality-config-path`, `--embodiment-tag`, `--output-dir`, `--experiment-name`, `--wandb-project`, `--state-dropout-prob`.
- **`--` passthrough** (to launch_finetune.py): `--tune_llm`, `--tune_visual`, `--tune_projector`, `--tune_diffusion_model`, `--gradient_accumulation_steps`.
- **Edit finetune.sh** (hardcoded): `learning_rate` (1e-4), `warmup_ratio` (0.05), `weight_decay` (1e-5), `save_total_limit` (5).

**N1.6 defaults:** LLM + vision backbone frozen (`tune_llm=False`, `tune_visual=False`); only projector + diffusion action head are trained. LR 1e-4 is correct — leave it.

**Adjust by what wandb shows:**
- Overfitting (train loss down, eval MSE up) → fewer `MAX_STEPS`, or `--state-dropout-prob 0.1`, or **collect more data** (16 episodes is tiny; aim 50+).
- Loss NaN early → lower LR to `5e-5` (edit finetune.sh).
- OOM → `GLOBAL_BATCH_SIZE=16` + `-- --gradient_accumulation_steps 2` (not needed on H100 80GB).

---

## Gotchas

- **CUDA error 803** ("unsupported display driver / cuda driver combination") = user/kernel driver mismatch. Cause here: forcing `cuda-compat` onto `LD_LIBRARY_PATH` when the instance driver (580, CUDA 12.8) already works. **Do not force compat.** If a shell still 803s: `sed -i '/cuda-12.8\/compat/d' ~/.bashrc && exec bash`, and `echo "$LD_LIBRARY_PATH"` must not contain `cuda-12.8/compat`. `torch` (cu128 wheel) bundles its own CUDA runtime and only needs a compatible driver.
- **Python must be 3.10 or 3.12, not 3.11** (no prebuilt flash-attn wheel for 3.11 → source build fails on missing nvcc).
- **Dataset must be v2.1**, not v3.0 — the GR00T loader can't read v3.0.
- **Use the bimanual modality.json / config** — NVIDIA's SO-100 tutorial templates are single-arm and won't match this 12-dim dataset.
- **First training run downloads `nvidia/GR00T-N1.6-3B`** (~6 GB, needs HF access; may be license-gated → `hf auth login` + accept terms).
- **`hf` CLI** (huggingface_hub ≥ 0.34) replaces the deprecated `huggingface-cli`. `hf download` / `hf upload` / `hf auth login`.
