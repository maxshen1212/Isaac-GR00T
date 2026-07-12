# GR00T N1.7 · SO-101 雙臂微調 Cheatsheet

在 NVIDIA Brev(1× RTX PRO 6000 96GB)上,從零到訓練完成並上傳的完整執行順序。

**路徑基準**

| 項目 | 路徑 / 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(fork `maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集 | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials` |
| 輸出 checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_finetune` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 repo | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials` |

---

## STEP 0 — 環境設定(交給 Brev 開機 Setup Script,不用手動跑)

把整份 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script → Paste Script**。VM 開好後 Brev 會自動執行(clone repo、裝依賴、下載+轉換資料集、算 stats、驗證 GPU),所以**不用手動 curl / bash**。

> ⚠️ **部署前必做**:先把 `examples/finetune.sh` 的改動 **push 到 `origin/n1.7-graphen`**。
> Brev 腳本會 `git clone` 遠端分支,訓練跑的是 clone 下來的 `finetune.sh`;若遠端是舊版,
> `SAVE_TOTAL_LIMIT` 環境變數會被忽略(退回寫死的 5)。

SSH 進去後,先驗證開機腳本跑完了、而且東西在你的家目錄:

```bash
# 1) 環境是否建好（Brev 頁面也看得到 setup log；若還在跑就等它完成）
ls ~/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
test -f ~/Isaac-GR00T/datasets/bimanual-so101-pickvials/meta/relative_stats.json && echo "dataset+stats OK"

# 2) uv 是否在 PATH（setup 若以 root 身分跑，uv/repo 可能落在 /root 而非你的家目錄）
which uv || ls ~/.local/bin/uv
```

> ⚠️ **落點陷阱**:若第 2 步 `uv` 找不到,檢查 `sudo ls /root/Isaac-GR00T`——若在那,代表 setup 以 root 執行,
> repo/venv/uv 都在 `/root`。解法:後續步驟改用 `sudo -i` 切成 root 操作。

兩項都 OK → 進 STEP 1。此步 **不需 HF token**(資料集公開);腳本冪等,重跑會自動跳過已完成的部分。

---

## STEP 1 — 登入 HF + wandb(訓練前一次)

```bash
cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# HF：先到 https://huggingface.co/nvidia/GR00T-N1.7-3B 接受授權，再用 WRITE token 登入
hf auth login                 # 或 export HF_TOKEN=<token>

# wandb（不想用就跳過，並在 STEP 3 指令最前面加 USE_WANDB=0）
wandb login <YOUR_API_KEY>    # key: https://wandb.ai/authorize
```

---

## STEP 2 — Dry-run:用極小 step 驗證整條流程(~10 分鐘)

> 正式訓練前先跑這個,把整條 `訓練 → 存檔 → 自動上傳` 用 10 個 step 走一遍。
> 驗證:模型下載/載入、資料集+modality 對得上、**batch 64 不 OOM**、checkpoint 存檔、`&&` 自動上傳 + HF 寫入權限。
> 順便把 ~7GB base model 快取起來,正式訓練(STEP 3)就不用再下載。

```bash
cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

hf auth whoami                # 先確認 HF 已登入且有寫入權限

# 輸出丟 /tmp（用完即棄）；上傳到「私有的 -dryrun 測試 repo」，不污染正式 repo。
# 用跟正式訓練「相同的 GLOBAL_BATCH_SIZE=64」→ 這裡不 OOM，正式跑才安全。
USE_WANDB=0 CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
MAX_STEPS=10 SAVE_STEPS=5 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=64 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /tmp/so101_dryrun \
&& uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-dryrun \
     /tmp/so101_dryrun . --repo-type model --private --exclude '*optimizer.pt'
```

**通過標準(全部要出現):**

- 訓練沒 OOM,跑完 10 步,終端出現 `Model saved to /tmp/so101_dryrun`
- `/tmp/so101_dryrun` 底下有 `checkpoint-5`、`checkpoint-10`
- `&&` 之後的 `hf upload` 成功(沒有 401/403 權限錯),HF 上出現私有的 `...-dryrun` repo

**(可選)順便驗證 STEP 4 的 eval 指令能跑:**

```bash
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /tmp/so101_dryrun/checkpoint-10 \
  --traj-ids 0 --execution-horizon 16 --steps 50
# 註：dry-run 只練 10 步，MSE 會很大很正常，這裡只驗「指令能跑通」，不是看數值。
```

**清理 dry-run 產物:**

```bash
rm -rf /tmp/so101_dryrun
# 私有測試 repo 想刪的話，到 HF 網頁 Settings 刪 `...-dryrun`（或用當前 hf CLI 支援的刪除指令）。
```

通過後再進 STEP 3 正式訓練。

---

## STEP 3 — 訓練 + 訓練成功自動上傳

> **務必在 tmux 裡跑**,否則 SSH 一斷,8–10 小時的訓練就死了。

```bash
tmux new -s train             # 斷線後用 `tmux attach -t train` 重新接上

cd ~/Isaac-GR00T
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
MAX_STEPS=30000 SAVE_STEPS=5000 SAVE_TOTAL_LIMIT=6 GLOBAL_BATCH_SIZE=64 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune \
  --wandb-project so101-bimanual \
  --experiment-name pickvials-n1p7-run1 \
&& uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
     ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune . \
     --repo-type model --exclude '*optimizer.pt'
```

- 離開 tmux(訓練繼續):`Ctrl-b` 然後按 `d`。重新接上:`tmux attach -t train`。
- `&&` = 只有訓練成功(exit 0)才上傳;訓練失敗不會亂傳。
- `--exclude '*optimizer.pt'` = 略過只有續訓才用得到的 optimizer(省一半上傳量)。想要可續訓的完整備份就拿掉它。

---

## STEP 4 — 訓練後:健康檢查 + 選 checkpoint

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

# 健康檢查：MSE 應隨 step「單調下降」。掃幾個看趨勢即可（不是挑最低點）
for ckpt in 10000 20000 30000; do
  echo "=== checkpoint-$ckpt ==="
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 16 --steps 400
done
```

**選擇定案**:官方預設用終點 `checkpoint-30000`,最終以 **實機 rollout 的 task success rate** 拍板。
open-loop MSE 只是「訓練有沒有在學」的健康檢查(`rising` = 訓練壞掉,不是過擬合),**不是** checkpoint 選擇器。

---

## STEP 5 — 只在 STEP 3 的自動上傳失敗時,手動重傳

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"
ls ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune          # 先確認有哪些 checkpoint
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune . \
  --repo-type model --exclude '*optimizer.pt'
```

---

## 速查表

| 項目 | 值 / 說明 |
|---|---|
| 訓練時間(估) | ~8–10 小時(1× RTX PRO 6000, batch 64, 30K steps) |
| Checkpoint | 6 個(5K/10K/15K/20K/25K/30K),~60GB 本地 / ~42GB 上傳(排除 optimizer) |
| 磁碟 | 500GB 綽綽有餘(checkpoint ~60GB + venv/模型/資料集開銷 ~40GB) |
| 斷線保護 | **務必在 tmux 裡跑** |
| 想改回可續訓的完整備份 | STEP 3/5 拿掉 `--exclude '*optimizer.pt'` |
| 關閉 wandb | STEP 3 指令最前面加 `USE_WANDB=0`,並移除 `--wandb-project` |
| 顯存還有餘裕想更平滑 | `GLOBAL_BATCH_SIZE=96`,搭配 `MAX_STEPS=20000`(一樣 ~11 epochs) |
| 想練更久 | 加大 `MAX_STEPS`;中途斷線要接回同一數值則加 `--resume-from-checkpoint` |

## 超參數為什麼是這組(對應本資料集:159 episodes / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `MAX_STEPS` | 30000 | ≈11 epochs(= steps×batch/176700)。錨的是「掃過幾遍你的資料」,不是照抄官方 8 卡的 step 數 |
| `GLOBAL_BATCH_SIZE` | 64 | 96GB 塞得下、梯度穩定;官方 batch 32–640 都用同一 LR |
| `learning_rate` | 1e-4(預設) | 官方全系列驗證過,不隨 batch 改 |
| `grad_accum` | 1(預設) | 顯存夠,不需要;用了反而減少 weight update 次數 |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 5000 / 6 | 6 個 checkpoint 蓋滿整段,全留不被輪替刪 |
| `tune_visual` / `tune_llm` | False(預設) | 單任務 159 軌跡,解凍 backbone 會過擬合 |
