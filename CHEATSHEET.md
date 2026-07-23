# GR00T N1.7 · SO-101 雙臂 Sim+Real Co-training Cheatsheet

8× H100 80GB SXM / 128 vCPU / ~1.5 TiB RAM @ $37.00/hr(Nebius,instance type `gpu-h100-sxm.8gpu-128vcpu-1600gb`)。**用官方預設參數**(不碰 shard/rate)。

> ✅ 這台**可以 stop/restart 不掉資料**(已在 Nebius 上驗證過)。訓練仍建議在 tmux 裡跑(防斷線)+ 帶 `UPLOAD_TO_HUB_REPO`(STEP 3)邊訓邊備份到 HF,方便其他機器直接下載做 eval——但**不用再驗證上傳成功後自動砍機器**:訓練跑完直接 `brev stop` 省錢即可,本機 checkpoint 還在,之後要補傳/繼續 eval 隨時 `brev start` 開回來。

**Co-training 作法**(sim-and-real co-training, arXiv:2503.24361;repo 機制見 `README.md`「multiple dataset paths with mixture weighting」):兩套資料集用 `:` 串接餵給 `--dataset-path`,同一個 run 內按權重混合取樣。預設取樣權重 ∝ frame 數 → **sim ≈ 79.6% / real ≈ 20.4%**,已屬 paper 建議的 sim-dominant 區間,先不動(調整旋鈕見底部參數表 `DS_WEIGHTS_ALPHA`)。

| 項目 | 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(`maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集(real) | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps`(49 ep / 22,157 frames) |
| 資料集(sim) | `~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps`(260 ep / 86,442 frames) |
| Checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain` |

---

## STEP 0 — 環境(Brev Setup Script 自動跑)

把 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script**。SSH 進去後驗證:

```bash
ls ~/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
nvidia-smi -L | wc -l    # 8
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

# 資料集 ×2（real+sim）：下載 → v3→v2 → modality.json → stats（★ 要在 hf auth login 之後,否則匿名下載撞 429）
# 兩套 camera keys/task 文字/joint 都已對齊,共用同一份 modality.json 與 config
for name in bimanual-so101-pickvials-real-10fps bimanual-so101-pickvials-sim-10fps; do
  uv run hf download ChihHanShen/$name \
      --repo-type dataset --local-dir ~/Isaac-GR00T/datasets/$name
  GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
      python scripts/lerobot_conversion/convert_v3_to_v2.py \
      --repo-id $name --root ~/Isaac-GR00T/datasets
  cp examples/SO101_bimanual/modality.json ~/Isaac-GR00T/datasets/$name/meta/modality.json
  uv run python gr00t/data/stats.py \
      --dataset-path ~/Isaac-GR00T/datasets/$name \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py
done

# 驗證（兩套都要過）
for DS in ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps \
          ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps; do
  python -c "import json; assert json.load(open('$DS/meta/info.json'))['codebase_version']=='v2.1'; print('$DS v2.1 OK')"
  test -f $DS/meta/relative_stats.json && echo "$DS relative_stats.json OK"
done
```

> 下載失敗 → `rm -rf ~/Isaac-GR00T/datasets/<該資料集>` 清掉半套,重跑迴圈（另一套已完成的不用動）。

---

## STEP 2 — Dry-run(~15 分鐘,用正式訓練的完整配置)

```bash
DS_REAL=~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps
DS_SIM=~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps

USE_WANDB=0 NUM_GPUS=8 \
MAX_STEPS=200 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=640 \
DATALOADER_NUM_WORKERS=4 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/_dryrun
```

> `--dataset-path` 用 `:` 串接多套資料集是官方作法(`examples/robocasa/README.md`、`finetune_config.py` docstring),**引號必加**。

**通過標準（全部要對）:**
- 跑完 200 步沒 OOM,`_dryrun/` 有 `checkpoint-100`/`-200`
- **起跑 log 印出 mixture 表**(每個 dataset 一行,`mix_ratio` 欄):real ≈ 20% / sim ≈ 80%(取樣 ∝ frame 數,`gr00t/data/dataset/factory.py`)——兩行都要在,少一行 = 有一套沒被讀到
- log 出現 `Uploading checkpoint-200 to ...` 且**沒有** `Upload of ... failed`(權限 401/403 會在這現形)
- HF 出現私有 `...-cotrain-dryrun` repo,底下有 checkpoint(巢狀路徑,見 STEP 3),且 checkpoint 內**沒有 `global_step*/`**(有 = exclude 沒生效,正式跑會白傳 ~28GB/checkpoint)
- **VRAM ≤ 75 GB/卡**(`nvidia-smi`;H100 只有 80GB,門檻比 96GB 的 RTX6000 收緊;OOM → STEP 3 改 `GLOBAL_BATCH_SIZE=512`)

> ⚠️ callback 上傳失敗**只記 log、不中斷訓練**(Hub 掛掉不該殺死 47h 的 run),所以**必須自己看 log**,不能只看有沒有跑完。

### （選用）調 `DATALOADER_NUM_WORKERS`

影片解碼(`torchcodec`,`device="cpu"`)吃的是 vCPU 核心——每張卡開 `DATALOADER_NUM_WORKERS` 個 worker process,worker 內部 ffmpeg 又是 auto 多執行緒(`num_ffmpeg_threads=0`),所以「多開幾個 worker」不一定更快,可能反而讓 CPU 搶核心。公式只能抓起始值,實測才準:

**起始值** = `floor(vCPU 數 ÷ NUM_GPUS) − 1`(留 1 核給每張卡的主 training process)。例如這台 128 vCPU / 8 GPU → 起始值 ≈ 15(比官方預設 `4` 寬鬆很多)。

```bash
# 另開兩個 terminal 陪跑 dry-run
watch -n1 nvidia-smi     # 盯 GPU-Util 欄
uptime                    # 盯 load average(1 分鐘均值)
```

把 STEP 2 的 `DATALOADER_NUM_WORKERS` 依序改成幾組值(如 4 → 8 → 15 → 20)各跑一次 200 步 dry-run,對照:

- **GPU-Util 常態 90%+ 且 load average ≤ 總 vCPU 數**:這組可用,可以再往上加試試看還能不能更快
- **load average 明顯超過總 vCPU 數**:已經 oversubscribe,worker 之間在搶核心,調低
- **GPU-Util 常掉很低、load average 沒超標**:worker 數不夠餵資料,調高

挑進度條 `s/it` 最低、同時 load average 沒爆的那組,寫回 STEP 3 的 `DATALOADER_NUM_WORKERS`。

**記下 step time**(進度條的 `s/it`,用調好 worker 數之後的那組):`25000 × step_time × $37.00/hr` 是正式訓練帳單。跟餘額對不上要現在反悔——`MAX_STEPS` 一旦開跑就不能改(cosine 以它退火,resume 接不回)。

```bash
rm -rf ~/Isaac-GR00T/checkpoints/_dryrun
```

---

## STEP 3 — 正式訓練

```bash
tmux new -s train             # 斷線後 tmux attach -t train

cd ~/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

DS_REAL=~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps
DS_SIM=~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps

NUM_GPUS=8 \
GLOBAL_BATCH_SIZE=640 \
MAX_STEPS=25000 SAVE_STEPS=2500 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=4 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-run3

# ↓↓↓ 訓練跑完直接停機省錢(這台可 stop/restart 不掉資料,不用再賭上傳驗證後自動刪除) ↓↓↓
brev stop <INSTANCE_NAME>        # 名稱用 brev ls 查;之後要補傳/繼續 eval,brev start 開回來即可
```

> 上段要貼進 tmux 裡跑,不能單獨存 `.sh`(`tmux new -s train` 會接管終端機)。存檔跑改用 `tmux new -d -s train "bash train_and_stop.sh"`(拿掉 `tmux new` 那行)。

- 離開 tmux(訓練繼續):`Ctrl-b` 然後 `d`。`UPLOAD_TO_HUB_REPO` 邊存邊背景傳 HF,只 rank0 上傳,失敗記 log 不中斷。
- 帶 `--experiment-name` 時本機路徑多一層:`<output-dir>/<experiment-name>/checkpoint-N`。
- HF 上傳路徑是**巢狀** `<experiment-name>/checkpoint-N`(`gr00t/experiment/utils.py` `HubUploadCallback`,避免不同 run 撞 step 數互相覆蓋)。舊 run2 repo 是平的 `checkpoint-N`,和這次的 `-cotrain` repo 互不影響。
- 自動上傳預設 exclude `*optimizer.pt` **和 `global_step*`**(DeepSpeed shards,~28GB/checkpoint,只有 resume 才需要)——run2 曾因漏擋 `global_step*` 白傳 28.78GB,已改進 `HubUploadCallback` 預設。
- Normalization 不用另外處理:mixture dataset 會把兩套的 stats 按取樣權重 merge 後存進 checkpoint(`sharded_mixture_dataset.py` `merge_statistics`),推論端照常載入。
- `NUM_GPUS=8` 自動開 DeepSpeed ZeRO-2;8× H100 走 NVLink/SXM 互聯,梯度 all-reduce 比 PCIe 卡快。
- 機器只是 **stop**、沒有刪除,本機 checkpoint 跟資料集都還在——log 裡看到 `Upload of ... failed` 才需要跑 STEP 5,不影響是否能 `brev stop`。

---

## STEP 4 — 選 checkpoint

**4a**(open-loop)只能當健檢,不能排名——官方 `finetune_new_embodiment.md` 只到這步。**4b/4c**(rollout success rate)才是排名依據,repo 沒有官方文件寫這步,是延伸做法。

### 4a. Open-loop 健檢

> 訓練機是 **stop 而非刪除**,本機 checkpoint 還在——可以 `brev start` 開回訓練機直接健檢,也可以在**任何一台裝了這個 repo + 資料集**的機器上先從 HF 下載 checkpoint 再健檢(擇一即可)：

```bash
REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain
EXP=pickvials-n1p7-run3
mkdir -p ~/Isaac-GR00T/eval_plots ~/models/bimanual-pickvials-cotrain
for ckpt in 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000; do
  echo "=== checkpoint-$ckpt ==="
  uv run hf download $REPO \
    --include "$EXP/checkpoint-$ckpt/*" --exclude "$EXP/checkpoint-$ckpt/global_step*" \
    --local-dir ~/models/bimanual-pickvials-cotrain
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path ~/models/bimanual-pickvials-cotrain/$EXP/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 16 --steps 400 \
    --save-plot-path ~/Isaac-GR00T/eval_plots/checkpoint-$ckpt.jpeg
done
```

> `--exclude ".../global_step*"` 是防禦性的(自動上傳已預設不傳 `global_step*`,正常情況 HF 上不會有)。**open-loop 用 real-10fps**(健檢的是對真機軌跡的 MSE;sim 那套交給 4b 的 rollout)。資料集若這台機器上沒有,先照 STEP 1 下載+轉換那段跑一次(只跑 real 那套即可)。

看 `Average MSE`/`MAE` 是否隨 step 大致下降(flat/rising = 沒在學,可剔除)。**每個 checkpoint 要帶不同 `--save-plot-path`**(沒帶會 fallback 到同一個 `/tmp/open_loop_eval/traj_0.jpeg`,互相覆蓋)。機器沒畫面,圖要 `scp -r shadeform@<host>:~/Isaac-GR00T/eval_plots ./` 傳回本機看。

### 4b. Sim rollout success rate 排名

健檢後挑 3 個間隔(如 15000/20000/25000)實跑 rollout。**另一個 repo、另一台有 Isaac Sim 的機器**,指令詳見 `Sim-to-Real-SO-101-Workshop/run_cheatsheet.md`「Eval」段:

```bash
# --exclude 防禦性擋 global_step*(自動上傳已預設不傳)
EXP=pickvials-n1p7-run3
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  --include "$EXP/checkpoint-25000/*" --exclude "$EXP/checkpoint-25000/global_step*" \
  --local-dir ~/models/bimanual-pickvials-cotrain

# 終端機 A(~/Isaac-GR00T)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-cotrain/$EXP/checkpoint-25000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 終端機 B(~/env_isaaclab):印 Success Rate: X/Y (Z%)
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval --num_episodes 10 \
    --policy_host localhost --policy_port 5555
```

換 checkpoint 就重下載/重起 server/重跑,10 集抓相對排名即可。

### 4c. 真機 eval 拍板(本次是 co-trained,**必跑**)

4b 選出前 1-2 名,用 `so101_eval.py`(NVIDIA Strategy 2 流程)跑真機驗證,以真機成功率做最終決定。

> ⚠️ 不要預設 step 數最大 = 最好:real 資料佔比小,後段可能對那一小撮過擬合(sim 漲、真機反而退步)。務必實際比較,峰值常落在中段。

---

## STEP 5 — 手動重傳(log 出現 `Upload of ... failed` 時)

```bash
EXP=pickvials-n1p7-run3
CK=~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain/$EXP/checkpoint-25000
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  "$CK" "$EXP/$(basename "$CK")" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

> `path_in_repo` 用巢狀 `"$EXP/$(basename "$CK")"`,對齊自動上傳路徑(見 STEP 3)。兩個 `--exclude` 對齊 `HubUploadCallback` 的預設 exclude(`*optimizer.pt` + `global_step*`),手動補傳才不會把 ~28GB 的 DeepSpeed shards 傳上去。

---

## 參數(官方預設,real 49 ep + sim 260 ep / 108,599 frames / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `GLOBAL_BATCH_SIZE` | **640** | 官方 full-scale 值,8×H100 正好對齊官方 8-GPU 參照點。⚠️ **是總量不是每卡** → 640÷8 = 80/卡 |
| `MAX_STEPS` | **25000** | 16M 樣本 ≈ **147 epochs**(108,599 frames),對齊舊 run 的 134-epoch 錨(≈LIBERO)。沿用 50K 會到 295 epochs,real 佔比小、易過擬合。⚠️ **一次定生死**(cosine 退火,不能 resume 延長) |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` | **2500 / 10** | 25K÷10,共 10 個 checkpoint(2500–25000),留足中段候選點抓峰值 |
| `DS_WEIGHTS_ALPHA` | **不設(預設)** | 預設取樣權重 ∝ frame 數 → sim 79.6%/real 20.4%,已屬 sim-dominant(arXiv:2503.24361:sim 為主、比例要調不是越高越好;paper 的 α=0.99 是 sim:real=50:1 的情境,我們只有 3.9:1,不適用)。4c 真機 eval 不佳才掃:`0`→50/50、`1.61`→90/10(權重 = `len^α`,`factory.py`) |
| `DATALOADER_NUM_WORKERS` | **4** | 官方值 |
| `learning_rate` | 1e-4(預設) | 官方 batch 32→640 全用同一 LR |
| `gradient_accumulation_steps` | 1(預設) | 顯存夠不需要 |
| `EPISODE_SAMPLING_RATE` / `SHARD_SIZE` | **預設,不動** | 官方所有 example 都不覆寫。改動會 OOM |
| `tune_*` / action_configs | 預設,不動 | 官方設計 |

**實際在訓練的**:VLM backbone(LLM+vision)**凍結**;action head + `vlln` + VL self-attention 訓練(51.54% 參數)。

**上一輪的痛點已處理**:舊 run 的問題是 30fps episode 需 ~140 次連續開環 chunk 預測,長程複合誤差大(`TRAINING_ANALYSIS.md` §9.2 建議 10–15 fps 重匯出)——這兩套 10fps 資料集就是該方案:sim episode ≈332 frames、real ≈452 frames,execution-horizon 16 下約 **21/28 次 chunk**,與 LIBERO-long(~33 次)同量級。若 25K 練完成功率仍不理想,下一步是掃 `DS_WEIGHTS_ALPHA` 或補收 real 資料,不是加 steps。
