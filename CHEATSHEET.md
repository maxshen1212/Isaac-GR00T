# GR00T N1.7 · SO-101 雙臂微調 Cheatsheet

4× RTX PRO 6000 96GB / 60 CPU / 768 GiB RAM @ $10.51/hr。參數推導見 [`TRAINING_ANALYSIS.md`](TRAINING_ANALYSIS.md)。

## ⚠️ 開機前

> 🚫 **This instance cannot be stopped or restarted. If your organization runs out of
> credits, Brev deletes the instance and all data. Deleted data cannot be recovered.**

- **訓練必須在 tmux 裡**,且必須帶 `UPLOAD_TO_HUB_REPO`(STEP 3)。

| 項目 | 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(`maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集 | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials` |
| Checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_finetune` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials` |

---

## STEP 0 — 環境(Brev Setup Script 自動跑,不用手動)

把 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script**。它只裝環境(依賴/repo/venv/GPU),
**不碰資料集也不需要 token** —— 資料集在 STEP 1 登入後才下載。SSH 進去後驗證:

```bash
ls ~/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
nvidia-smi -L | wc -l    # 4
nproc                    # ~60
which uv || ls ~/.local/bin/uv
```

> ⚠️ `uv` 找不到 → 檢查 `sudo ls /root/Isaac-GR00T`。setup 若以 root 跑,東西都在 `/root`,
> 後續改用 `sudo -i`。

---

## STEP 1 — 登入 + 資料集(一次,~30–45 分鐘)

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

# 1) 先在網頁接受 gated repo（沒授權會在「載入模型」時才 403，資料集下載成功也沒用）：
#      https://huggingface.co/nvidia/Cosmos-Reason2-2B    VLM backbone ★ 必要
#      https://huggingface.co/nvidia/GR00T-N1.7-3B        base model
hf auth login                 # 要 WRITE token（checkpoint 要上傳）
wandb login <API_KEY>         # 不用就在後續指令加 USE_WANDB=0 並移除 --wandb-project

# 2) 資料集：下載 → v3.0 轉 v2.1 → 裝 modality.json → 算 stats
#    ★ 一定要在 hf auth login 之後跑：資料集雖然是 public，但匿名下載會被 HF 以「每 IP」
#      限流（共享雲端 IP 很快撞 HTTP 429）。登入後就不會。
uv run hf download ChihHanShen/bimanual-so101-pickvials \
    --repo-type dataset --local-dir ~/Isaac-GR00T/datasets/bimanual-so101-pickvials
GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
    python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id bimanual-so101-pickvials --root ~/Isaac-GR00T/datasets
cp examples/SO101_bimanual/modality.json ~/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/modality.json
uv run python gr00t/data/stats.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 3) 驗證（兩個都要 OK 才能進 STEP 2）
DS=~/Isaac-GR00T/datasets/bimanual-so101-pickvials
python -c "import json; v=json.load(open('$DS/meta/info.json'))['codebase_version']; assert v=='v2.1', f'dataset is {v}'; print(f'dataset {v} OK')"
test -f $DS/meta/relative_stats.json && echo "relative_stats.json OK"
```

> **下載中斷/失敗**:`rm -rf ~/Isaac-GR00T/datasets/bimanual-so101-pickvials` 清掉半套,重跑第 2 步。

---

## STEP 2 — Dry-run(~15 分鐘, ~$3)

用跟 STEP 3 **相同的上傳機制**(`UPLOAD_TO_HUB_REPO`),才驗得到正式訓練真正會走的路徑:

```bash
USE_WANDB=0 NUM_GPUS=4 \
MAX_STEPS=200 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=640 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/_dryrun
```

**通過標準:** 跑完 200 步沒 OOM · `_dryrun/` 有 `checkpoint-100`/`-200`
· log 出現 `Uploading checkpoint-200 to ...` 且**沒有** `Upload of ... failed`(權限 401/403 會在這裡現形)
· HF 上出現私有的 `...-dryrun` repo,底下有 `checkpoint-100/`、`checkpoint-200/`
· **VRAM ≤ 90 GB/卡**(OOM → STEP 3 改 `GLOBAL_BATCH_SIZE=512`)

> ⚠️ callback **不會**因上傳失敗而中斷訓練(那是刻意的:Hub 掛掉不該殺死 47 小時的 run)。
> 所以權限問題**只會出現在 log**,訓練仍會 exit 0。**必須自己看 log**,不能只看有沒有跑完。

**記下 step time** — `50000 × step_time × $10.51/hr` 是正式訓練的帳單。**這是唯一能反悔的時機**:
`MAX_STEPS` 一旦開跑就不能改(cosine 以它為週期退火到 0,resume 接不回來)。

```bash
rm -rf ~/Isaac-GR00T/checkpoints/_dryrun
```

---

## STEP 3 — 正式訓練

```bash
tmux new -s train             # 斷線後 tmux attach -t train

cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

NUM_GPUS=4 \
GLOBAL_BATCH_SIZE=640 \
MAX_STEPS=50000 SAVE_STEPS=5000 SAVE_TOTAL_LIMIT=10 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune \
  --wandb-project so101-bimanual \
  --experiment-name pickvials-n1p7-run2
```

- 離開 tmux(訓練繼續):`Ctrl-b` 然後 `d`。
- **`UPLOAD_TO_HUB_REPO`(本 fork 新增)** — 每存一個 checkpoint 就在背景推上 HF,
  所以**不需要結尾的 `&& hf upload`**。只在 rank 0 上傳、失敗只記 log 不中斷、
  結束前等最後一個傳完。私有 repo 加 `UPLOAD_TO_HUB_PRIVATE=1`。
- ⚠️ **`MAX_STEPS` 必須是 `SAVE_STEPS` 的倍數**(50000 ÷ 5000 ✓)。callback 只上傳
  `checkpoint-*`;訓練結束時 `trainer.save_model()` 另外寫一份到 `output_dir` 根目錄,
  那份**不會**被上傳。倍數關係成立時它跟 `checkpoint-50000` 內容相同,所以不影響;
  改成非倍數(如 49000)就會漏掉最終模型。
- HF repo 上會是 `checkpoint-5000/` … `checkpoint-50000/`,每個都自足可載入
  (`CheckpointFormatCallback` 已把 `experiment_cfg/` + `processor/` 複製進去)。
  載入時指到 `<repo>/checkpoint-50000`。
- `NUM_GPUS=4` 會自動開 DeepSpeed ZeRO-2(單卡時是關的),不用設定。

---

## STEP 4 — 選 checkpoint

```bash
for ckpt in 10000 20000 30000 40000 50000; do
  echo "=== checkpoint-$ckpt ==="
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 16 --steps 400
done
```

**以 sim rollout 的 success rate 拍板。** open-loop MSE 只是「有沒有在學」的健康檢查
(`rising` = 訓練壞了,不是過擬合),**不是** checkpoint 選擇器。

---

## STEP 5 — 手動重傳(只在 log 出現 `Upload of ... failed` 時)

補傳單一 checkpoint。**`path_in_repo` 要跟 callback 一致**(`checkpoint-<step>`,不是 `.`),
否則 repo 會長出兩套結構:

```bash
CK=~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/checkpoint-50000
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  "$CK" "$(basename "$CK")" --repo-type model --exclude '*optimizer.pt'
```

---

## 超參數(資料集:159 episodes / 176,671 有效 steps / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `GLOBAL_BATCH_SIZE` | **640** | ⚠️ **是總量不是每卡** → 640÷4 = **160/卡**。設 128 會變 32/卡 |
| `MAX_STEPS` | **50000** | 32.0 M 樣本 ≈ robocasa 官方 30.7 M;181 epochs ≈ LIBERO 134 的 1.35×。⚠️ **一次定生死** |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` | **5000 / 10** | 50K÷10。設 2000 會產 25 個而只留 10 → 早期取樣點被刪 |
| `learning_rate` | 1e-4(預設) | 官方 batch **32→1024** 全用同一 LR(6 個 example)→ 對 batch 不敏感 |
| `gradient_accumulation_steps` | 1(預設) | 只在顯存不夠時用。不增加吞吐量,固定時間下只給一半的權重更新 |
| `EPISODE_SAMPLING_RATE` / `SHARD_SIZE` / `DATALOADER_NUM_WORKERS` | **0.1 / 1024 / 4 — 預設,不動** | 官方 6 個 example 都沒覆寫。RAM 只吃 ~135 GB。(分析見 `TRAINING_ANALYSIS.md` §2,**本次不採用**) |
| `tune_visual` / `tune_llm` | False(預設) | 單任務 159 軌跡,解凍 backbone 會過擬合 |
| action_configs | 手臂 RELATIVE / 夾爪 ABSOLUTE | 官方 SO100 的刻意設計。全改 ABSOLUTE 會讓 `relative_stats.json` 失效 |

**實際在訓練的**:VLM backbone(LLM+vision)**凍結**;action head 全部 + `vlln` + VL self-attention。
`tune_vlln` 只在 model config、CLI 關不掉 → 「只練 action head」的說法不精確。

**參數解決不了的**:episode 是 **37.5 秒 / 30fps**,需 **~140 次**連續開環 chunk 預測
(LIBERO-10「long」只有 13.4 秒 / ~33 次)。若練足 50K 仍不理想,下一步是**以 10–15 fps 重新匯出**,
不是再加 steps。見 `TRAINING_ANALYSIS.md` §9.2。
