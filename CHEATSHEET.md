# GR00T N1.7 · SO-101 雙臂微調 Cheatsheet

4× RTX PRO 6000 96GB / 60 CPU / 567 GiB RAM @ $10.51/hr。**用官方預設參數**(不碰 shard/rate)。

> 🚫 這台**不能暫停、只能刪除**,信用點用完 = **整台連 checkpoint 一起刪除、不可復原**。
> 所以:訓練**務必在 tmux 裡**,**務必帶 `UPLOAD_TO_HUB_REPO`**(STEP 3)邊訓邊上傳,STEP 3 結尾會驗證上傳成功後**自動 `brev delete` 省錢**(因此 STEP 4a 改成從 HF 下載 checkpoint)。

| 項目 | 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(`maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集 | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials`(240 ep / 單任務) |
| Checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_finetune` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials` |

---

## STEP 0 — 環境(Brev Setup Script 自動跑)

把 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script**。SSH 進去後驗證:

```bash
ls ~/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
nvidia-smi -L | wc -l    # 4
which uv || ls ~/.local/bin/uv
```

> `uv` 找不到 → `sudo ls /root/Isaac-GR00T`,setup 若以 root 跑東西都在 `/root`,後續改用 `sudo -i`。

---

## STEP 1 — 登入 + 資料集(一次)

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

# 先在網頁接受 gated repo（沒授權會在載入模型時 403）：
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B    ★ VLM backbone,必要
#   https://huggingface.co/nvidia/GR00T-N1.7-3B        base model
hf auth login                 # 要 WRITE token（checkpoint 要上傳）
wandb login <API_KEY>         # 不用就在指令加 USE_WANDB=0 並移除 --wandb-project

# 資料集：下載 → v3→v2 → modality.json → stats（★ 要在 hf auth login 之後,否則匿名下載撞 429）
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

# 驗證
DS=~/Isaac-GR00T/datasets/bimanual-so101-pickvials
python -c "import json; assert json.load(open('$DS/meta/info.json'))['codebase_version']=='v2.1'; print('dataset v2.1 OK')"
test -f $DS/meta/relative_stats.json && echo "relative_stats.json OK"
```

> 下載失敗 → `rm -rf $DS` 清掉半套,重跑下載那段。

---

## STEP 2 — Dry-run(~15 分鐘,用正式訓練的完整配置)

```bash
USE_WANDB=0 NUM_GPUS=4 \
MAX_STEPS=200 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=640 \
DATALOADER_NUM_WORKERS=4 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/_dryrun
```

**通過標準（全部要對）:**
- 跑完 200 步沒 OOM,`_dryrun/` 有 `checkpoint-100`/`-200`
- log 出現 `Uploading checkpoint-200 to ...` 且**沒有** `Upload of ... failed`(權限 401/403 會在這現形)
- HF 出現私有 `...-dryrun` repo,底下有 `checkpoint-100/`、`checkpoint-200/`
- **VRAM ≤ 90 GB/卡**(`nvidia-smi`;OOM → STEP 3 改 `GLOBAL_BATCH_SIZE=512`)

> ⚠️ callback 上傳失敗**只記 log、不中斷訓練**(Hub 掛掉不該殺死 47h 的 run),所以**必須自己看 log**,不能只看有沒有跑完。

**記下 step time**(進度條的 `s/it`):`50000 × step_time × $10.51/hr` 是正式訓練帳單。跟餘額對不上要現在反悔——`MAX_STEPS` 一旦開跑就不能改(cosine 以它退火,resume 接不回)。

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
DATALOADER_NUM_WORKERS=4 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_finetune \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-run2

# ↓↓↓ 訓練+上傳確認完成後自動刪除機器省錢(brev CLI 需先 login --token) ↓↓↓
# 本地目錄 + HF 都存在才刪:只查 HF 會被舊 run 留下的同名 checkpoint 騙過(run2 是平的路徑)
LOCAL_CKPT=~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/pickvials-n1p7-run2/checkpoint-50000
LAST_CKPT=checkpoint-50000
if [ -d "$LOCAL_CKPT" ] && uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
    --include "$LAST_CKPT/config.json" --local-dir /tmp/verify_upload >/tmp/verify_upload.log 2>&1; then
  brev delete <INSTANCE_NAME>        # 名稱用 brev ls 查
else
  echo "!!! 驗證沒過,機器不刪——先跑 STEP 5 補傳,確認上傳後再手動 brev delete"
fi
```

> 上段要貼進 tmux 裡跑,不能單獨存 `.sh`(`tmux new -s train` 會接管終端機)。存檔跑改用 `tmux new -d -s train "bash train_and_delete.sh"`(拿掉 `tmux new` 那行)。

- 離開 tmux(訓練繼續):`Ctrl-b` 然後 `d`。`UPLOAD_TO_HUB_REPO` 邊存邊背景傳 HF,只 rank0 上傳,失敗記 log 不中斷。
- 帶 `--experiment-name` 時本機路徑多一層:`<output-dir>/<experiment-name>/checkpoint-N`。
- HF 上傳路徑:**run2(這次)是平的** `checkpoint-N`;**之後新 run 已改成巢狀** `<experiment-name>/checkpoint-N`(避免不同 run 撞 step 數互相覆蓋,見 `gr00t/experiment/utils.py` `HubUploadCallback`)。
- `NUM_GPUS=4` 自動開 DeepSpeed ZeRO-2。

---

## STEP 4 — 選 checkpoint

**4a**(open-loop)只能當健檢,不能排名——官方 `finetune_new_embodiment.md` 只到這步。**4b/4c**(rollout success rate)才是排名依據,repo 沒有官方文件寫這步,是延伸做法。

### 4a. Open-loop 健檢

> 訓練機已被自動刪除,在**任何一台裝了這個 repo + 資料集**的機器上先從 HF 下載 checkpoint 再健檢:

```bash
REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials
mkdir -p ~/Isaac-GR00T/eval_plots ~/models/bimanual-pickvials
for ckpt in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do
  echo "=== checkpoint-$ckpt ==="
  uv run hf download $REPO \
    --include "checkpoint-$ckpt/*" --exclude "checkpoint-$ckpt/global_step*" \
    --local-dir ~/models/bimanual-pickvials
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ~/models/bimanual-pickvials/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 16 --steps 400 \
    --save-plot-path ~/Isaac-GR00T/eval_plots/checkpoint-$ckpt.jpeg
done
```

> `--exclude "checkpoint-$ckpt/global_step*"` 擋掉 28GB 的 DeepSpeed optimizer(推論不需要,見 STEP 4b)。資料集(`~/Isaac-GR00T/datasets/bimanual-so101-pickvials`)若這台機器上沒有,先照 STEP 1 下載+轉換那段跑一次。

看 `Average MSE`/`MAE` 是否隨 step 大致下降(flat/rising = 沒在學,可剔除)。**每個 checkpoint 要帶不同 `--save-plot-path`**(沒帶會 fallback 到同一個 `/tmp/open_loop_eval/traj_0.jpeg`,互相覆蓋)。機器沒畫面,圖要 `scp -r shadeform@<host>:~/Isaac-GR00T/eval_plots ./` 傳回本機看。

### 4b. Sim rollout success rate 排名

健檢後挑 3-4 個間隔(如 40000/45000/50000)實跑 rollout。**另一個 repo、另一台有 Isaac Sim 的機器**,指令詳見 `Sim-to-Real-SO-101-Workshop/run_cheatsheet.md`「Eval」段:

```bash
# --exclude 擋掉 28GB 的 DeepSpeed optimizer,推論不需要
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  --include "checkpoint-50000/*" --exclude "checkpoint-50000/global_step*" \
  --local-dir ~/models/bimanual-pickvials

# 終端機 A(~/Isaac-GR00T)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials/checkpoint-50000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 終端機 B(~/env_isaaclab):印 Success Rate: X/Y (Z%)
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval --num_episodes 10 \
    --policy_host localhost --policy_port 5555
```

換 checkpoint 就重下載/重起 server/重跑,10 集抓相對排名即可。

### 4c. 真機 eval 拍板(co-trained 模型才需要)

4b 選出前 1-2 名,用 `so101_eval.py`(NVIDIA Strategy 2 流程)跑真機驗證,以真機成功率做最終決定。

> ⚠️ 不要預設 step 數最大 = 最好:real 資料佔比小,後段可能對那一小撮過擬合(sim 漲、真機反而退步)。務必實際比較,峰值常落在中段。

---

## STEP 5 — 手動重傳(log 出現 `Upload of ... failed` 時)

```bash
EXP=pickvials-n1p7-run2
CK=~/Isaac-GR00T/checkpoints/so101_bimanual_finetune/$EXP/checkpoint-50000
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials \
  "$CK" "$(basename "$CK")" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

> `path_in_repo` 要對齊原本上傳方式(見 STEP 3):run2 用平的;之後新 run 要改成 `"$EXP/$(basename "$CK")"`。`--exclude 'global_step*'` 擋 DeepSpeed optimizer(`*optimizer.pt` 擋不到,run2 自動上傳漏了這個,白傳 28.78GB)。

---

## 參數(官方預設,資料集 240 ep / 239,223 有效 steps / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `GLOBAL_BATCH_SIZE` | **640** | 官方 full-scale(8×RTX6000)值。⚠️ **是總量不是每卡** → 640÷4 = 160/卡 |
| `MAX_STEPS` | **50000** | 32M 樣本 ≈ robocasa 官方;134 epochs ≈ LIBERO。⚠️ **一次定生死**(cosine 退火,不能 resume 延長) |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` | **5000 / 10** | 50K÷10。設 2000 會產 25 個而只留 10 |
| `DATALOADER_NUM_WORKERS` | **4** | 官方值 |
| `learning_rate` | 1e-4(預設) | 官方 batch 32→640 全用同一 LR |
| `gradient_accumulation_steps` | 1(預設) | 顯存夠不需要 |
| `EPISODE_SAMPLING_RATE` / `SHARD_SIZE` | **預設,不動** | 官方所有 example 都不覆寫。改動會 OOM |
| `tune_*` / action_configs | 預設,不動 | 官方設計 |

**實際在訓練的**:VLM backbone(LLM+vision)**凍結**;action head + `vlln` + VL self-attention 訓練(51.54% 參數)。

**參數解決不了的**:episode 是 33.7 秒 / 30fps,推論時需 ~140 次連續開環 chunk 預測(LIBERO「long」只 13.4 秒/~33 次),長程複合誤差大。若練足 50K 成功率仍不理想,下一步是**以 10–15 fps 重新匯出資料**,不是加 steps。詳見 `TRAINING_ANALYSIS.md` §9.2。
