# GR00T N1.7 · SO-101 雙臂 Sim+Real Co-training Cheatsheet

8× H100 74GB SXM5 / 112 CPU / 894 GiB RAM @ $28.04/hr(SFCompute,`h100.ib`)。用官方預設參數(不碰 shard/rate)。

> 🚫 **不能暫停/重啟,只能刪除**(欠費 Brev 會自動連 checkpoint 一起刪掉;帳號已設自動儲值)。訓練務必在 tmux 裡跑 + 帶 `UPLOAD_TO_HUB_REPO` 當備份。跑完手動確認 log 無 `Upload of ... failed` 後才 `brev delete`。

**Co-training**:兩套資料集用 `:` 串接餵 `--dataset-path`,同一 run 內按權重混合取樣。權重見底部 `DS_WEIGHTS_ALPHA`——**2026-08-13 起預設值會變成 real 主導,必須手動設**。

| 項目 | 值 |
|---|---|
| Repo | `~/Isaac-GR00T`(`maxshen1212/Isaac-GR00T` @ `n1.7-graphen`) |
| 資料集(real 10fps) | `datasets/bimanual-so101-pickvials-real-10fps`(101 ep / 19,963 frames → **18,448 steps / 19 shards**) |
| 資料集(sim 10fps) | `datasets/bimanual-so101-pickvials-sim-10fps`(100 ep / 8,965 frames → **7,465 steps / 8 shards**) |
| 資料集(sim 30fps) | `datasets/bimanual-so101-pickvials-sim`(100 ep / 26,802 frames → **25,302 steps / 25 shards**)*run 2 專用* |
| Checkpoint | `~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain` |
| Base model | `nvidia/GR00T-N1.7-3B` |
| 上傳目標 | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain` |

> **steps** = `Σ(每集幀數 − action_horizon + 1)` = 總幀數 − 集數×15,是實際可抽的樣本數。
> **shards** = `ceil(steps / 1024)`,`len(dataset)` 回傳的就是它,`DS_WEIGHTS_ALPHA` 是對它取冪次。

---

## 這一輪要跑的三個 run

單變數設計:**run 1 是共同對照**,1↔2 只變 fps、1↔3 只變有沒有 real 資料。

| run | experiment-name | 資料集 | `MAX_STEPS` | `SAVE_STEPS` | `DS_WEIGHTS_ALPHA` | 掃幾遍 |
|---|---|---|---|---|---|---|
| **1** sim-only 10fps | `pickvials-n1p7-r1-sim10` | sim-10fps | **2500** | 250 | 不設 | 214 |
| **2** sim-only 30fps | `pickvials-n1p7-r2-sim30` | sim(30fps) | **6500** | 500 | 不設 | 164 |
| **3** co-train 50/50 | `pickvials-n1p7-r3-cotrain` | real-10fps + sim-10fps | **6000** | 500 | **`0`** | sim 257 / real 104 |

三個 run 的 epoch 數都對齊在 ~170,`GLOBAL_BATCH_SIZE=640` 不變(官方 Training Tips:
*"Maximize batch size for your hardware and train for a few thousand steps"*——batch 要大,但步數是**幾千**)。

⚠️ **不能用「跑 25k 然後挑早期 checkpoint」代替**:LR 是 cosine 退火到 `MAX_STEPS`,
25000 步那個 run 的 checkpoint-2500 沒有經過退火收尾,跟完整跑 2500 步不是同一件事。`MAX_STEPS` 要一開始就選對。

⚠️ 官方註明同樣設定重跑有 **5–6% 變異**(非確定性影像增強)。三個 run 差距在 5–6% 以內 = 噪音,不能下結論。

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

# 資料集 ×3（run1/2/3 全部用到）：下載 → v3→v2 → modality.json → stats（★ 要在 hf auth login 之後,否則匿名下載撞 429）
# bimanual-so101-pickvials-sim 是 30fps 原始版,只有 run 2 要;跑 run1/run3 可以只拉前兩套。
for name in bimanual-so101-pickvials-real-10fps bimanual-so101-pickvials-sim-10fps bimanual-so101-pickvials-sim; do
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

# 驗證（每套都要過）
for DS in ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps \
          ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
          ~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim; do
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

USE_WANDB=1 NUM_GPUS=8 \
DS_WEIGHTS_ALPHA=0 \
MAX_STEPS=200 SAVE_STEPS=100 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=640 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/_dryrun
```

> `--dataset-path` 用 `:` 串接多套資料集是官方作法,**引號必加**。

**通過標準（全部要對）:**
- 跑完 200 步沒 OOM,`_dryrun/` 有 `checkpoint-100`/`-200`
- 起跑 log 印出 **shard 數**,兩行都要在而且數字對得上:
  `Generated 19 shards for ...real-10fps` / `Generated 8 shards for ...sim-10fps`
  → **對不上就別往下跑**,`DS_WEIGHTS_ALPHA` 是對 shard 數取冪,shard 變了 alpha 就要重算(見底部)
- log 出現 `Applied ds_weights_alpha=0 across 2 datasets`,mixture 為 **sim ≈ 50% / real ≈ 50%**
  (⚠️ 不設 alpha 的話是 **real 70% / sim 30%**——這是新資料的預設,不是舊文件寫的 sim80/real20)
- log 出現 `Uploading checkpoint-200 to ...` 且**沒有** `Upload of ... failed`
- HF 出現私有 `...-cotrain-dryrun` repo,checkpoint 內**沒有** `global_step*/`(有 = exclude 沒生效,正式跑會白傳 ~28GB/checkpoint)
- **VRAM ≤ 62 GB/卡**(這台每卡只有 74GB;OOM → STEP 3 改 `GLOBAL_BATCH_SIZE=512`)

> ⚠️ callback 上傳失敗**只記 log、不中斷訓練**,必須自己看 log,不能只看有沒有跑完。

**（選用）調 `DATALOADER_NUM_WORKERS`**:起始值 = `vCPU 數 ÷ NUM_GPUS − 1`(這台 112/8 ≈ 13)。陪跑 dry-run 時開兩個 terminal 盯 `watch -n1 nvidia-smi`(GPU-Util)和 `uptime`(load average),依序測幾組值(4→8→13→18):GPU-Util 常態 90%+ 且 load ≤ 總 vCPU 數 → 可用,還能再往上加;load 明顯超標 → 調低;GPU-Util 常掉很低 → 調高。挑 `s/it` 最低那組寫回 STEP 3。

**記下 step time**:`MAX_STEPS × step_time × $28.04/hr` 是該 run 的帳單(run1 2500 / run2 6500 / run3 6000 步)。`MAX_STEPS` 一旦開跑就不能改(cosine 退火,resume 接不回)。

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
DS_SIM30=~/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim

# ---- run 1:sim-only 10fps(對照組)----
NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 \
MAX_STEPS=2500 SAVE_STEPS=250 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r1-sim10

# ---- run 2:sim-only 30fps(只變 fps)----
NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 \
MAX_STEPS=6500 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_SIM30" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r2-sim30

# ---- run 3:co-train 50/50(只變有沒有 real)----
NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 \
DS_WEIGHTS_ALPHA=0 \
MAX_STEPS=6000 SAVE_STEPS=500 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r3-cotrain
```

> run 1/2 是單一資料集,`DS_WEIGHTS_ALPHA` 設了也沒用(程式碼要 `len(all_datasets) > 1` 才套用),不必帶。

> 上段要貼進 tmux 裡跑,不能單獨存 `.sh`。存檔跑改用 `tmux new -d -s train "bash train.sh"`(拿掉 `tmux new` 那行)。

**訓練跑完(手動,不自動化)**:確認 log 出現最後一個 checkpoint 的 `Uploading checkpoint-<MAX_STEPS> to ...` 且**沒有** `Upload of ... failed`,再手動:

```bash
brev delete <INSTANCE_NAME>        # 名稱用 brev ls 查——這台不能 stop,只能刪
```

> 有 `Upload of ... failed` 先跑 STEP 5 補傳、確認 HF 上有完整 checkpoint 再刪,不要邊猜邊刪。

- 離開 tmux(訓練繼續):`Ctrl-b` 然後 `d`。只 rank0 上傳,失敗記 log 不中斷。
- 帶 `--experiment-name` 時本機路徑多一層:`<output-dir>/<experiment-name>/checkpoint-N`;HF 上傳路徑同樣是巢狀 `<experiment-name>/checkpoint-N`。
- 自動上傳預設 exclude `*optimizer.pt` 和 `global_step*`(DeepSpeed shards,只有 resume 才需要)。
- Normalization 不用另外處理:mixture dataset 會把兩套的 stats merge 後存進 checkpoint,推論端照常載入。
- `NUM_GPUS=8` 自動開 DeepSpeed ZeRO-2;8× H100 走 NVLink/SXM5,梯度 all-reduce 比 PCIe 卡快。

---

## STEP 4 — 選 checkpoint

**4a**(open-loop)只能當健檢,不能排名。**4b/4c**(rollout success rate)才是排名依據。

### 4a. Open-loop 健檢

> 訓練機用完會被手動 `brev delete`,在**任何一台裝了這個 repo + 資料集**的機器上先從 HF 下載 checkpoint 再健檢:

```bash
REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain
EXP=pickvials-n1p7-r3-cotrain                      # run1: -r1-sim10 / run2: -r2-sim30
CKPTS="500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000"   # = SAVE_STEPS 的倍數
# run1(SAVE_STEPS=250,到 2500):CKPTS="250 500 750 1000 1250 1500 1750 2000 2250 2500"
# run2(SAVE_STEPS=500,到 6500):上面再加 6500
mkdir -p ~/Isaac-GR00T/eval_plots ~/models/bimanual-pickvials-cotrain
for ckpt in $CKPTS; do
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
done 2>&1 | tee ~/Isaac-GR00T/eval_plots/eval_output.txt
```

> **open-loop 用 real-10fps**(健檢的是對真機軌跡的 MSE;sim 那套交給 4b)。**每個 checkpoint 要帶不同 `--save-plot-path`**(沒帶會互相覆蓋)。機器沒畫面,圖用 `scp -r shadeform@<host>:~/Isaac-GR00T/eval_plots ./` 傳回本機看。

> **這也是唯一能實測「步數是不是訂太多」的判準**:`Average MSE`/`MAE` 隨 step **持續下降** = 還在學;
> **走平** = 收斂了;**開始回升 = 過擬合**,最低點那個 checkpoint 就是答案。
> 若最低點落在很前面(例如 run1 的 750 步),代表 `MAX_STEPS` 還可以再砍——但**下一輪重跑才有意義**,
> 因為 cosine 退火沒在那個點收尾。反之若到最後還在降,才考慮加步數。

### 4b. Sim rollout success rate 排名

健檢後挑 3 個間隔實跑 rollout。**另一台有 Isaac Sim 的機器**,指令詳見 `Sim-to-Real-SO-101-Workshop/run_cheatsheet.md` §6(Eval — Sim):

```bash
EXP=pickvials-n1p7-r3-cotrain; CK=6000
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  --include "$EXP/checkpoint-$CK/*" --exclude "$EXP/checkpoint-$CK/global_step*" \
  --local-dir ~/models/bimanual-pickvials-cotrain

# 終端機 A(~/Isaac-GR00T)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/graphen/models/bimanual-pickvials-cotrain/$EXP/checkpoint-$CK \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 終端機 B(~/env_isaaclab):印 Success Rate: X/Y (Z%)
#   ⚠️ --steps_per_action 每個 run 不一樣,見下表
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval --num_episodes 10 \
    --policy_host localhost --policy_port 5555 --steps_per_action 3
```

換 checkpoint 就重下載/重起 server/重跑,10 集抓相對排名即可。

### ⚠️ 三個 run 的 eval 參數不同(最容易出錯的地方)

env 現在是 **30 Hz**(`decimation=4`),`--steps_per_action` = 錄製 fps ÷ 訓練 fps:

| run | 訓練資料 fps | `lerobot_eval_dual --steps_per_action` | `eval_so101_dual.py` 的 `fps` |
|---|---|---|---|
| 1 sim-10fps | 10 | **3** | 10 |
| 2 sim-30fps | **30** | **1** | **30** |
| 3 co-train 10fps | 10 | **3** | 10 |

**run 2 忘了改就會被跑成 1/3 速,量到的數字直接作廢。**
run 2 另有結構差異:16 步 chunk 在 30fps 下只有 0.53 秒,一集要連續開環預測約 140 次
(10fps 只要 21–28 次)。run 2 若明顯較差,這是首要嫌疑,不是「fps 本身比較爛」。

### 4c. 真機 eval 拍板(本次是 co-trained,**必跑**)

4b 選出前 1-2 名,用 **`gr00t/eval/real_robot/SO101_bimanual/eval_so101_dual.py`** 跑真機成功率做最終決定。
那支就是官方 `SO100/eval_so100.py` 的雙臂版(235 行,官方 291 行),同樣是
connect → `while True`:取 obs → policy → 執行 chunk。只差兩件事:12 維 + 3 相機的 adapter、
以及 10 Hz 控制頻率。**不動上游 `SO100/`**——它 pin 的 lerobot commit 更舊,且是單臂結構。

**三個坑:**
1. **控制頻率 = 該 checkpoint 的訓練資料集 fps**(查 `meta/info.json`),不是錄製的 30 fps。官方 client
   一律拿資料採集頻率當控制頻率(DROID 15、SO100 1/30),不內插。
   → **run 1 / run 3 用 `fps=10`(script 預設);run 2 是 30fps 資料訓的,必須改成 `fps=30`。**
   10 Hz 下 16 步 chunk 覆蓋 1.6 秒,遠大於推論延遲(實測 0.070 s)→ 不會有 stop-and-go;
   30 Hz 下只覆蓋 0.53 秒,餘裕仍有 7.6 倍但小很多。**用錯會差 3 倍速,這比其他任何問題都危險。**
2. **server 已把 relative 轉成絕對關節目標**(`decode_action` 用 client 送的 state)。
   client 不可再加回 state,也不可套 `SO101_USD_MAPPING`(那是 sim USD 單位的產物)。
3. **有 `--model-path` 時 `--modality-config-path` 被靜默忽略**,config 來自 checkpoint 的 processor。

**兩個環境不可混用**:server 用 Isaac-GR00T 的 `.venv`(`torch 2.9.0`/`transformers 4.57.3`),
client 沿用 lerobot 的 `.venv`。**server 借用 lerobot venv 會讓前處理跟訓練不同**(它是 5.5.4/2.11.0)。

```bash
# client 一次性設定
cd /home/graphen/sim2real/lerobot
uv pip install msgpack==1.1.0 msgpack-numpy==0.4.8
VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps -e /home/graphen/sim2real/Isaac-GR00T
```

> Isaac-GR00T 首次 `uv sync` 會卡在 `torchcodec-...aarch64.whl`(未 smudge 的 LFS pointer)→
> `git lfs install --local && git lfs pull --include="scripts/deployment/dgpu/wheels/*.whl"`
> server 首次啟動另會下載 gated backbone `nvidia/Cosmos-Reason2-2B`(~5GB),載完 GPU 佔 7.3 GB。

```bash
# 終端機 A — server
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path ~/models/bimanual-pickvials-cotrain/pickvials-n1p7-r3-cotrain/checkpoint-6000 \
    --embodiment-tag NEW_EMBODIMENT --port 5555

# 終端機 B — client(旗標一律「底線」,draccus;4a 的 open_loop_eval.py 走 tyro 用連字號,別互抄)
cd gr00t/eval/real_robot/SO101_bimanual
PY=/home/graphen/sim2real/lerobot/.venv/bin/python

# 第一次帶電:限位設到最小,手放電源開關。0.5°/write @10Hz ≈ 5°/s,慢到可以用手擋。
# 會大量 clamp、機器人嚴重落後 policy,正常。只確認運動方向對、Ctrl-C 收得乾淨。
$PY eval_so101_dual.py \
  --max_relative_target '{"shoulder_pan":0.5,"shoulder_lift":0.5,"elbow_flex":0.5,"wrist_flex":0.5,"wrist_roll":0.5,"gripper":1.0}'

# 確認沒問題後全速跑,**用 lerobot-replay 重播真人 demo 並排比速度**。
# 比 demo 快 = fps 設錯了,這比任何其他問題都危險。
$PY eval_so101_dual.py
```

硬體參數(port / 校正目錄 / 三台相機序號 / task 字串)都是 script 預設值,平常不用帶。
換 checkpoint 只重啟 server,client 不動。

**跑法**:一次執行 = 一集。script 是 `while True`,**不會自己停**——看著它做,成功或明顯卡住就
**Ctrl-C**,人工把試管擺回去,再跑一次。成功率自己記(紙筆或試算表都行)。

- 判定用二元「90 秒內 4 支全進架子」(對齊 sim 的 all-or-nothing)。**90 秒是你自己看錶的協定,
  script 不會強制**;順手記 `vials_placed` (0-4) 當低雜訊的第二指標。
- 操作者/硬體失誤(試管掉出桌面、人碰到手臂)那次不計入分母。
- **N=20**。N=10 時 50% 的 95% CI 約 ±31 個百分點,分不出 30% 和 70%。
  三個 run 各 20 集 = 60 次真機 eval,而且**差距 <5–6% 就是官方講的 run 間噪音,不能下結論**。
- 換 run 只改 server `--model-path`(client 不動),**但 run 2 要同時改 client 的 `fps=30`**。
  比較:run1↔run3 = co-training 有沒有用;run1↔run2 = 10fps vs 30fps。

> **⚠️ 開始量之前先確認管線本身是通的。** 真機 eval 已經連續兩次拿到 0%。若 client 有 bug
> (相機 key、action 縮放、relative action),**三個 run 都會讀到 0%,你分不出是模型還是管線壞了**。
> 先用任一 checkpoint 走完「首次帶電 1 集 → 全速單集跟 `lerobot-replay` 真人 demo 並排比速度」,
> 確認機器人動作合理,再開始正式計分。

> **⚠️ Ctrl-C 會讓手臂軟掉**:`disable_torque_on_disconnect` 是 lerobot 預設的 `True`,
> 所以停止時 torque 會斷、手臂下墜(跟 `lerobot-record` 收尾一樣)。差別在這裡的跑法就是
> **在動作中途按 Ctrl-C**——夾著試管停手的話它會掉。要停之前先讓它把手放低,或接受這個行為。

> **⚠️ 相機與馬達搶 USB**:`SOFollower.connect()` 是先開三台 RealSense 才 `configure()` 寫馬達,
> 而 Feetech 的 write 預設 `num_retry=0`——單一封包損毀就會讓 connect 整個爆掉
> (`Failed to write 'Lock' on id_=6`)。實測 12 顆馬達在沒有相機串流時全部正常。
> 再遇到就重跑一次;一直失敗就用 `lsusb -t` 看三台 D435i 是不是掛在同一個 root hub 分開接。
> (相機 640×480@30 是寫死在 `make_robot()` 裡的,沒有 CLI 旗標,要降 fps 得改那一行。)

> **舊的 50% sim 基準線已作廢,不要拿來對照**(ROADMAP 2026-08-11):`decimation` 由 2 改成 4
> (env 60 Hz → 30 Hz),舊 checkpoint 現在會以正確速率執行,數字必然不同;而且舊資料全部刪了。
> 這一輪的 run 1 就是新的 sim-only 基準線。真機 vs sim 對照時仍有兩個不對等要註明:
> ① sim 自動判定(`confirm_steps=25`)vs 真機人判;② sim 22.5 秒 vs 真機 90 秒。

> ⚠️ 不要預設 step 數最大 = 最好,峰值常落在中段。這一輪 `MAX_STEPS` 已按資料量重算(見底部),
> 但仍要靠 4a 的 MSE 曲線確認峰值位置。


---

## STEP 5 — 手動重傳(log 出現 `Upload of ... failed` 時)

```bash
EXP=pickvials-n1p7-r3-cotrain
CK=~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain/$EXP/checkpoint-6000
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  "$CK" "$EXP/$(basename "$CK")" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

> `path_in_repo` 用巢狀 `"$EXP/$(basename "$CK")"`,對齊自動上傳路徑。兩個 `--exclude` 才不會把 ~28GB 的 DeepSpeed shards 傳上去。

---

## 參數(2026-08-13 重算:real 101 ep + sim 100 ep / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `GLOBAL_BATCH_SIZE` | **640** | 官方 Training Tips「maximize batch size for your hardware」。⚠️ 是總量不是每卡 → 640÷8 = 80/卡 |
| `MAX_STEPS` | **依 run:2500 / 6500 / 6000** | 官方同一句話的後半:「train for **a few thousand** steps」。三個 run 對齊 ~170 epochs。⚠️ 一次定生死,不能 resume 延長 |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` | **250(run1)/ 500 / 10** | 步數變少,存點也要跟著變密才抓得到峰值 |
| `DS_WEIGHTS_ALPHA` | **`0`(run 3 必設)** | 見下方專節。**不設 = real 70% / sim 30%** |
| `DATALOADER_NUM_WORKERS` | **13** | 見 STEP 2 調參(112 vCPU ÷ 8 GPU − 1) |
| `learning_rate` | 1e-4(預設) | 官方 batch 32→640 全用同一 LR |
| `state_dropout_prob` | 0.2(CLI 預設) | 隨機丟 state 提升泛化。雙臂任務**很依賴** 12 維 state(要知道另一手在哪),官方說這種情況該**調低**;但它也是對抗過擬合的旋鈕,4a 顯示過擬合時優先動它,再考慮砍 steps |
| `gradient_accumulation_steps` | 1(預設) | 顯存夠不需要 |
| `EPISODE_SAMPLING_RATE` / `SHARD_SIZE` | **預設,不動** | 改動會 OOM,而且會改變 shard 數 → alpha 要重算 |
| `tune_*` / action_configs | 預設,不動 | 官方設計 |

**實際在訓練的**:VLM backbone(LLM+vision)**凍結**;action head + `vlln` + VL self-attention 訓練(51.54% 參數)。

### `DS_WEIGHTS_ALPHA` 怎麼算

權重是 **shard 數的冪次**(`factory.py`:`weights = len(ds)^alpha / len(ds[0])^alpha`,而
`len(dataset)` = shard 數,不是 frame 數)。`len(ds[0])^alpha` 在算比例時會消掉 →
**`--dataset-path` 誰寫前面不影響結果**。

$$p_{sim}=\frac{L_{sim}^{\alpha}}{L_{sim}^{\alpha}+L_{real}^{\alpha}} \quad\Longrightarrow\quad \alpha=\frac{\ln\!\big(\tfrac{p}{1-p}\big)}{\ln\!\big(L_{sim}/L_{real}\big)}$$

目前 `L_sim = 8`、`L_real = 19`:

| alpha | sim% | real% | |
|---|---|---|---|
| `1`(不設 = 預設) | 29.6 | 70.4 | ⚠️ **real 主導**——舊文件寫的 sim80/real20 已不成立 |
| **`0`** | **50** | **50** | ✅ 本輪用這個(兩邊 episode 數 100:101,等於每集曝光相同) |
| `-1.60` | 80 | 20 | |
| `-2.54` | 90 | 10 | arXiv:2503.24361 的預設值 |

**為什麼是負的**:sim 現在是**小**的那個資料集(8 < 19),要讓它主導就得反轉大小關係。
舊資料 sim 是大邊(86,442 vs 22,157)所以舊文件寫正的 `1.61`——**照舊值填會得到 sim 6%**。

**⚠️ real 卡在 shard 邊界上**:18,448 / 1024 = 18.016,只超過 18 這個整數 **16 steps**。
刪掉任何一集就會掉到 18 shards,alpha 要改成 `-2.71`(若要 90/10)。**每次都以 dry-run log 印的
shard 數為準**,不要沿用寫死的數字。

### 為什麼本輪不用 sim-dominant

arXiv:2503.24361 說「1:1 suboptimal、90/10 最好」,但那是在 **20 集 real + 1000 集 sim** 上測的
(real 只佔 2%),而且作者**沒有**針對不同 real 資料量重調過比例。本輪是 101:100,regime 完全不同;
更關鍵的是**這裡 sim 才是小資料集**(100 集,論文用 1000 集)——把 sim 拉到 90% 等於對 7,465 個
unique step 掃近 2000 遍,那是在背答案。所以走 `alpha=0`。

**10fps 資料已解決舊版痛點**:30fps 需 ~140 次連續開環 chunk 預測、長程複合誤差大;10fps 在
execution-horizon 16 下約 21/28 次 chunk,與 LIBERO-long 同量級。run 2 就是要實測這件事。
**練完不理想時,下一步是掃 `DS_WEIGHTS_ALPHA`、調 `state_dropout_prob` 或補收 real 資料,不是加 steps。**
