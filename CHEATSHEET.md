# GR00T N1.7 · SO-101 雙臂 Sim+Real Co-training Cheatsheet

8× H100 74GB SXM5 / 112 CPU / 894 GiB RAM @ $28.04/hr(SFCompute,`h100.ib`)。用官方預設參數(不碰 shard/rate)。

> 🚫 **不能暫停/重啟,只能刪除**(欠費 Brev 會自動連 checkpoint 一起刪掉;帳號已設自動儲值)。訓練務必在 tmux 裡跑 + 帶 `UPLOAD_TO_HUB_REPO` 當備份。跑完手動確認 log 無 `Upload of ... failed` 後才 `brev delete`。

**Co-training**:兩套資料集用 `:` 串接餵 `--dataset-path`,同一 run 內按權重混合取樣。預設權重 ∝ frame 數 → sim≈80% / real≈20%(sim-dominant,符合 arXiv:2503.24361 建議),先不動;要調見底部 `DS_WEIGHTS_ALPHA`。

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

USE_WANDB=1 NUM_GPUS=8 \
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
- 起跑 log 印出 mixture 表:real ≈ 20% / sim ≈ 80%,兩行都要在
- log 出現 `Uploading checkpoint-200 to ...` 且**沒有** `Upload of ... failed`
- HF 出現私有 `...-cotrain-dryrun` repo,checkpoint 內**沒有** `global_step*/`(有 = exclude 沒生效,正式跑會白傳 ~28GB/checkpoint)
- **VRAM ≤ 62 GB/卡**(這台每卡只有 74GB;OOM → STEP 3 改 `GLOBAL_BATCH_SIZE=512`)

> ⚠️ callback 上傳失敗**只記 log、不中斷訓練**,必須自己看 log,不能只看有沒有跑完。

**（選用）調 `DATALOADER_NUM_WORKERS`**:起始值 = `vCPU 數 ÷ NUM_GPUS − 1`(這台 112/8 ≈ 13)。陪跑 dry-run 時開兩個 terminal 盯 `watch -n1 nvidia-smi`(GPU-Util)和 `uptime`(load average),依序測幾組值(4→8→13→18):GPU-Util 常態 90%+ 且 load ≤ 總 vCPU 數 → 可用,還能再往上加;load 明顯超標 → 調低;GPU-Util 常掉很低 → 調高。挑 `s/it` 最低那組寫回 STEP 3。

**記下 step time**:`25000 × step_time × $28.04/hr` 是正式訓練帳單。`MAX_STEPS` 一旦開跑就不能改(cosine 退火,resume 接不回)。

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
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir ~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-run3
```

> 上段要貼進 tmux 裡跑,不能單獨存 `.sh`。存檔跑改用 `tmux new -d -s train "bash train.sh"`(拿掉 `tmux new` 那行)。

**訓練跑完(手動,不自動化)**:確認 log 出現最後一個 checkpoint 的 `Uploading checkpoint-25000 to ...` 且**沒有** `Upload of ... failed`,再手動:

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
done 2>&1 | tee ~/Isaac-GR00T/eval_plots/eval_output.txt
```

> **open-loop 用 real-10fps**(健檢的是對真機軌跡的 MSE;sim 那套交給 4b)。看 `Average MSE`/`MAE` 是否隨 step 大致下降(flat/rising = 沒在學,可剔除)。**每個 checkpoint 要帶不同 `--save-plot-path`**(沒帶會互相覆蓋)。機器沒畫面,圖用 `scp -r shadeform@<host>:~/Isaac-GR00T/eval_plots ./` 傳回本機看。

### 4b. Sim rollout success rate 排名

健檢後挑 3 個間隔(如 15000/20000/25000)實跑 rollout。**另一台有 Isaac Sim 的機器**,指令詳見 `Sim-to-Real-SO-101-Workshop/run_cheatsheet.md`「Eval」段:

```bash
uv run hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  --include "pickvials-n1p7-run3/checkpoint-25000/*" --exclude "pickvials-n1p7-run3/checkpoint-25000/global_step*" \
  --local-dir ~/models/bimanual-pickvials-cotrain

# 終端機 A(~/Isaac-GR00T)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /home/graphen/models/bimanual-pickvials-sim/pickvials-n1p7-run2/checkpoint-50000 \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 終端機 B(~/env_isaaclab):印 Success Rate: X/Y (Z%)
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval --num_episodes 10 \
    --policy_host localhost --policy_port 5555
```

換 checkpoint 就重下載/重起 server/重跑,10 集抓相對排名即可。

### 4c. 真機 eval 拍板(本次是 co-trained,**必跑**)

4b 選出前 1-2 名,用 **`gr00t/eval/real_robot/SO101_bimanual/eval_so101_dual.py`** 跑真機成功率做最終決定。
那支就是官方 `SO100/eval_so100.py` 的雙臂版(235 行,官方 291 行),同樣是
connect → `while True`:取 obs → policy → 執行 chunk。只差兩件事:12 維 + 3 相機的 adapter、
以及 10 Hz 控制頻率。**不動上游 `SO100/`**——它 pin 的 lerobot commit 早於 `bi_so_follower` 重構。

**三個坑:**
1. **控制頻率 10 Hz**(= 訓練資料集 fps,已查 `meta/info.json`),不是錄製的 30 fps。官方 client
   一律拿資料採集頻率當控制頻率(DROID 15、SO100 1/30),不內插。用 30 Hz 會快 3 倍。
   16 步 chunk 因此覆蓋 1.6 秒,遠大於推論延遲(實測 0.070 s)→ 不會有 stop-and-go。
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
    --model-path ~/models/bimanual-pickvials-cotrain/pickvials-n1p7-run3/checkpoint-25000 \
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
- 跑完換純 sim checkpoint(只改 server `--model-path`)重跑一輪,得 co-train vs sim-only 差距。

> **⚠️ Ctrl-C 會讓手臂軟掉**:`disable_torque_on_disconnect` 是 lerobot 預設的 `True`,
> 所以停止時 torque 會斷、手臂下墜(跟 `lerobot-record` 收尾一樣)。差別在這裡的跑法就是
> **在動作中途按 Ctrl-C**——夾著試管停手的話它會掉。要停之前先讓它把手放低,或接受這個行為。

> **⚠️ 相機與馬達搶 USB**:`SOFollower.connect()` 是先開三台 RealSense 才 `configure()` 寫馬達,
> 而 Feetech 的 write 預設 `num_retry=0`——單一封包損毀就會讓 connect 整個爆掉
> (`Failed to write 'Lock' on id_=6`)。實測 12 顆馬達在沒有相機串流時全部正常。
> 再遇到就重跑一次;一直失敗就用 `lsusb -t` 看三台 D435i 是不是掛在同一個 root hub 分開接。
> (相機 640×480@30 是寫死在 `make_robot()` 裡的,沒有 CLI 旗標,要降 fps 得改那一行。)

> **跟 sim 的 50% 對照時三個不對等要註明**:① sim 是 60 Hz,16 步 chunk 在 0.27 秒跑完,
> 比資料代表的 1.6 秒快 6 倍(physics-time vs data-time,不該轉移);② sim 自動判定
> (`confirm_steps=25`)vs 真機人判;③ sim 22.5 秒 vs 真機 90 秒。50% 是參考點,不是嚴格 baseline。

> ⚠️ 不要預設 step 數最大 = 最好:real 資料佔比小,後段可能對那一小撮過擬合。務必實際比較,峰值常落在中段。


---

## STEP 5 — 手動重傳(log 出現 `Upload of ... failed` 時)

```bash
EXP=pickvials-n1p7-run3
CK=~/Isaac-GR00T/checkpoints/so101_bimanual_cotrain/$EXP/checkpoint-25000
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  "$CK" "$EXP/$(basename "$CK")" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

> `path_in_repo` 用巢狀 `"$EXP/$(basename "$CK")"`,對齊自動上傳路徑。兩個 `--exclude` 才不會把 ~28GB 的 DeepSpeed shards 傳上去。

---

## 參數(官方預設,real 49 ep + sim 260 ep / 108,599 frames / 單任務)

| 參數 | 值 | 理由 |
|---|---|---|
| `GLOBAL_BATCH_SIZE` | **640** | 官方 full-scale 值。⚠️ 是總量不是每卡 → 640÷8 = 80/卡 |
| `MAX_STEPS` | **25000** | ≈147 epochs(108,599 frames),對齊 134-epoch 錨(≈LIBERO)。⚠️ 一次定生死,不能 resume 延長 |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT` | **2500 / 10** | 10 個 checkpoint,留足中段候選點抓峰值 |
| `DS_WEIGHTS_ALPHA` | **不設(預設)** | 預設權重已 sim-dominant(sim80/real20)。4c 真機 eval 不佳才掃:`0`→50/50、`1.61`→90/10 |
| `DATALOADER_NUM_WORKERS` | **4** | 見 STEP 2 調參 |
| `learning_rate` | 1e-4(預設) | 官方 batch 32→640 全用同一 LR |
| `gradient_accumulation_steps` | 1(預設) | 顯存夠不需要 |
| `EPISODE_SAMPLING_RATE` / `SHARD_SIZE` | **預設,不動** | 改動會 OOM |
| `tune_*` / action_configs | 預設,不動 | 官方設計 |

**實際在訓練的**:VLM backbone(LLM+vision)**凍結**;action head + `vlln` + VL self-attention 訓練(51.54% 參數)。

**10fps 資料已解決舊版痛點**:30fps 需 ~140 次連續開環 chunk 預測、長程複合誤差大;現在 execution-horizon 16 下約 21/28 次 chunk,與 LIBERO-long 同量級。25K 練完不理想,下一步是掃 `DS_WEIGHTS_ALPHA` 或補收 real 資料,不是加 steps。
