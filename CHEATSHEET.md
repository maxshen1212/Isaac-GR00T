# GR00T N1.7 · SO-101 雙臂 Sim+Real Co-training Cheatsheet

8× H100 74GB SXM5 / 112 CPU / 894 GiB RAM @ $28.04/hr(SFCompute,`h100.ib`)。

> 🚫 **不能暫停/重啟,只能刪除**(欠費 Brev 會自動連 checkpoint 一起刪掉)。帶 `UPLOAD_TO_HUB_REPO` 當備份,跑完確認 log 無 `Upload of ... failed` 後才 `brev delete`。

| 項目               | 值                                                                                              |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| Repo               | `$ROOT/Isaac-GR00T`(`maxshen1212/Isaac-GR00T` @ `n1.7-graphen`;`$ROOT` 各腳本開頭定義,預設 `~`) |
| 資料集(real 10fps) | `datasets/bimanual-so101-pickvials-real-10fps`(101 ep,18,448 steps / 19 shards)                 |
| 資料集(sim 10fps)  | `datasets/bimanual-so101-pickvials-sim-10fps`(100 ep,7,465 steps / 8 shards)                    |
| 資料集(sim 30fps)  | `datasets/bimanual-so101-pickvials-sim`(100 ep,25,302 steps / 25 shards)_run 2 專用_            |
| Checkpoint         | `$ROOT/Isaac-GR00T/checkpoints/so101_bimanual_cotrain`                                          |
| Base model         | `nvidia/GR00T-N1.7-3B`                                                                          |
| 上傳 repo 前綴     | `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-`(三 run 各自獨立 repo,見下表)                 |

---

## 這一輪三個 run(各開一個 tmux,同時跑)

單變數設計:run 1 是對照組,1↔2 只變 fps、1↔3 只變有沒有 real 資料。三個 run 共用同一組超參數。

| run              | tmux | experiment-name             | 資料集                 | GPU       | port  | `DS_WEIGHTS_ALPHA` | 上傳 repo(接前綴) |
| ---------------- | ---- | --------------------------- | ---------------------- | --------- | ----- | ------------------ | ----------------- |
| 1 sim-only 10fps | `r1` | `pickvials-n1p7-r1-sim10`   | sim-10fps              | `0,1`     | 29500 | 不設               | `sim-10fps`       |
| 2 sim-only 30fps | `r2` | `pickvials-n1p7-r2-sim30`   | sim(30fps)             | `2,3`     | 29501 | 不設               | `sim`             |
| 3 co-train 50/50 | `r3` | `pickvials-n1p7-r3-cotrain` | real-10fps + sim-10fps | `4,5,6,7` | 29502 | `0`                | `cotrain`         |

共用:`GLOBAL_BATCH_SIZE=128`、`MAX_STEPS=10000`、`SAVE_STEPS=1000`、`SAVE_TOTAL_LIMIT=10`、`--save-only-model`(不存 optimizer/`global_step*`,這台不能 resume 也不需要)。

⚠️ `MAX_STEPS` 開跑後不能改(cosine 退火到它,resume 接不回)。⚠️ 卡數只能 1/2/4/8(`GLOBAL_BATCH_SIZE % NUM_GPUS == 0` 有 assert)。⚠️ 官方註明同樣設定重跑有 5–6% 變異,三個 run 差距在這範圍內算噪音。

---

## STEP 0 — 環境(Brev Setup Script 自動跑)

把 `NVIDIA_Brev_script.sh` 貼到 Brev 部署頁的 **Setup Script**。SSH 進去後驗證:

```bash
ROOT=~

ls $ROOT/Isaac-GR00T/.venv/bin/activate && echo "venv OK"
nvidia-smi -L | wc -l    # 8
which uv || ls $ROOT/.local/bin/uv
```

> `uv` 找不到 → `sudo ls /root/Isaac-GR00T`,setup 若以 root 跑東西都在 `/root`,後續改用 `sudo -i`。

---

## STEP 1 — 登入 + 資料集(一次)

```bash
ROOT=~
cd $ROOT/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

# 先在網頁接受 gated repo（沒授權會在載入模型時 403）：
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B    ★ VLM backbone,必要
#   https://huggingface.co/nvidia/GR00T-N1.7-3B        base model
hf auth login                 # 要 WRITE token（checkpoint 要上傳）
wandb login <API_KEY>         # 不用就在指令加 USE_WANDB=0 並移除 --wandb-project

# 資料集 ×3：下載 → v3→v2 → modality.json → stats（★ 要在 hf auth login 之後,否則匿名下載撞 429）
for name in bimanual-so101-pickvials-real-10fps bimanual-so101-pickvials-sim-10fps bimanual-so101-pickvials-sim; do
  uv run hf download ChihHanShen/$name \
      --repo-type dataset --local-dir $ROOT/Isaac-GR00T/datasets/$name
  GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
      python scripts/lerobot_conversion/convert_v3_to_v2.py \
      --repo-id $name --root $ROOT/Isaac-GR00T/datasets
  cp examples/SO101_bimanual/modality.json $ROOT/Isaac-GR00T/datasets/$name/meta/modality.json
  uv run python gr00t/data/stats.py \
      --dataset-path $ROOT/Isaac-GR00T/datasets/$name \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py
done

# 驗證（每套都要過）
for DS in $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps \
          $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
          $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim; do
  python -c "import json; assert json.load(open('$DS/meta/info.json'))['codebase_version']=='v2.1'; print('$DS v2.1 OK')"
  test -f $DS/meta/relative_stats.json && echo "$DS relative_stats.json OK"
done
```

> 下載失敗 → `rm -rf $ROOT/Isaac-GR00T/datasets/<該資料集>` 清掉半套,重跑迴圈。

---

## STEP 2 — Dry-run(~15 分鐘,2 卡配置,跟 run 1/2 一致)

```bash
ROOT=~
DS_REAL=$ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps
DS_SIM=$ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps

USE_WANDB=1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 \
DS_WEIGHTS_ALPHA=0 \
MAX_STEPS=300 SAVE_STEPS=150 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=128 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $ROOT/Isaac-GR00T/checkpoints/_dryrun
```

**通過標準:**

- 跑完 300 步沒 OOM,`_dryrun/` 有 `checkpoint-150`/`-300`,**沒有** `global_step*/` 或 `optimizer.pt`
- log 印出 `Generated 19 shards for ...real-10fps` / `Generated 8 shards for ...sim-10fps`(對不上就別往下跑,shard 數變了 alpha 要重算,見底部)
- log 出現 `Applied ds_weights_alpha=0 across 2 datasets`,mixture ≈ sim 50% / real 50%
- log 出現 `Uploading checkpoint-300 to ...` 且**沒有** `Upload of ... failed`
- **記下 VRAM/卡跟 `s/it`**(每卡 64 個樣本,決定 STEP 3 三個 job 併行安不安全、算得出帳單)

**併行壓測**(STEP 3 就是這樣跑,三個 job 共用 112 vCPU):把上面指令改 `MAX_STEPS=100`,同時開三份分別指到 `0,1`/`2,3`/`4,5,6,7`(port 29500/29501/29502)。`s/it` 比單獨跑掉超過 ~20% → 調低 `DATALOADER_NUM_WORKERS`;`uptime` load average 不該超過 112。

```bash
ROOT=~
rm -rf $ROOT/Isaac-GR00T/checkpoints/_dryrun
```

---

## STEP 3 — 正式訓練(三個 tmux session,各自前景跑)

### Run 1 — `tmux new -s r1`

```bash
ROOT=~
cd $ROOT/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 MASTER_PORT=29500 \
GLOBAL_BATCH_SIZE=128 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim-10fps \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $ROOT/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r1-sim10
```

### Run 2 — `tmux new -s r2`

```bash
ROOT=~
cd $ROOT/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=2,3 NUM_GPUS=2 MASTER_PORT=29501 \
GLOBAL_BATCH_SIZE=128 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $ROOT/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r2-sim30
```

### Run 3 — `tmux new -s r3`

```bash
ROOT=~
cd $ROOT/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

DS_REAL=$ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps
DS_SIM=$ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps

CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 MASTER_PORT=29502 DS_WEIGHTS_ALPHA=0 \
GLOBAL_BATCH_SIZE=128 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=13 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS_REAL:$DS_SIM" \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $ROOT/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual --experiment-name pickvials-n1p7-r3-cotrain
```

各自 `Ctrl-b` `d` 離開(訓練繼續)。回頭看用 `tmux attach -t r1`(或 r2/r3)。

**跑完(手動,不自動化)**:三個 session 都要各自確認出現 `Uploading checkpoint-10000 to ...` 且**沒有** `Upload of ... failed`,再:

```bash
brev delete <INSTANCE_NAME>        # 名稱用 brev ls 查——這台不能 stop,只能刪
```

> 有 `Upload of ... failed` 先跑 STEP 5 補傳、確認 HF 上有完整 checkpoint 再刪。

---

## STEP 4 — 選 checkpoint

流程:**4a 下載 → 4b open-loop 健檢(篩)→ 4c sim rollout(排名)→ 4d 真機(拍板)**。
4b 的 MSE 只能篩掉明顯沒學好的,**不能排名**;排名一律看 4c/4d 的 success rate。

### 每個 run 的座標(整個 STEP 4 通用)

| run | `REPO`(接前綴) | `EXP` | `MODEL` | 訓練 fps | 4c `--steps_per_action` | 4d `--fps` |
|---|---|---|---|---|---|---|
| 1 | `sim-10fps` | `pickvials-n1p7-r1-sim10` | `bimanual-pickvials-sim-10fps` | 10 | 3 | 10(預設) |
| 2 | `sim` | `pickvials-n1p7-r2-sim30` | `bimanual-pickvials-sim-30fps` | 30 | **1** | **30** |
| 3 | `cotrain` | `pickvials-n1p7-r3-cotrain` | `bimanual-pickvials-cotrain` | 10 | 3 | 10(預設) |

前綴 = `ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-`。checkpoint 一律落在
`$ROOT/Isaac-GR00T/models/$MODEL/$EXP/checkpoint-<CK>`。

⚠️ `REPO` 跟 `EXP` 必須同一列,配錯會靜默抓不到檔(見 4a)。
⚠️ 控制頻率必須等於**訓練** fps,不是錄製的 30 fps;run 2 兩欄都要改,忘了差 3 倍速、數字作廢。

### 4a. 下載 checkpoint

一個 6.5 G,一個 run 10 個 = 65 G,三個 run ≈ 195 G。先 `df -h .`。

```bash
ROOT=~/sim2real
REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim
EXP=pickvials-n1p7-r2-sim10
MODEL=bimanual-pickvials-sim-10fps
CKPTS="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000"

mkdir -p $ROOT/Isaac-GR00T/models/$MODEL
for ckpt in $CKPTS; do
  echo "=== download checkpoint-$ckpt ==="
  uv run hf download $REPO \
    --include "$EXP/checkpoint-$ckpt/*" \
    --exclude "$EXP/checkpoint-$ckpt/global_step*" \
    --local-dir $ROOT/Isaac-GR00T/models/$MODEL
done

# ★ 一定要驗:EXP 打錯時 hf download 只印「Fetching 0 files」就當成功結束
for ckpt in $CKPTS; do
  test -f $ROOT/Isaac-GR00T/models/$MODEL/$EXP/checkpoint-$ckpt/config.json \
    && echo "✅ checkpoint-$ckpt" || echo "❌ checkpoint-$ckpt"
done
```

> 全 ❌ = `REPO`/`EXP` 配錯;零星 ❌ = 網路斷,重跑迴圈(已存在的會跳過)。沒驗就往下跑,`open_loop_eval` 會把不存在的路徑當成 HF repo id,噴 `HFValidationError`。

### 4b. Open-loop 健檢

```bash
ROOT=~/sim2real
EXP=pickvials-n1p7-r1-sim10
MODEL=bimanual-pickvials-sim-10fps
CKPTS="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000"

mkdir -p $ROOT/Isaac-GR00T/open_loop_eval/$MODEL
for ckpt in $CKPTS; do
  echo "=== checkpoint-$ckpt ==="
  uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path $ROOT/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps\
    --embodiment-tag NEW_EMBODIMENT \
    --model-path $ROOT/Isaac-GR00T/models/$MODEL/$EXP/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 8 --steps 400 \
    --save-plot-path $ROOT/Isaac-GR00T/open_loop_eval/$MODEL/checkpoint-$ckpt.jpeg
done 2>&1 | tee $ROOT/Isaac-GR00T/open_loop_eval/$MODEL/eval_output.txt

# 三個 run 都跑完後一次比
grep -H "Average MSE" $ROOT/Isaac-GR00T/open_loop_eval/*/eval_output.txt
```

> 固定用 real-10fps 當比較基準(對真機軌跡的 MSE)。`Average MSE` 持續下降=還在學,走平=收斂,回升=過擬合。圖跟 log 收在 `open_loop_eval/$MODEL/`,三個 run 不互相覆蓋。

### 4c. Sim rollout success rate(排名)

健檢後挑 3 個間隔實跑。指令詳見 `Sim-to-Real-SO-101-Workshop/run_cheatsheet.md` §6:

```bash
ROOT=~/sim2real
EXP=pickvials-n1p7-r1-sim10
MODEL=bimanual-pickvials-sim-10fps
CK=6000

# 終端機 A — server($ROOT/Isaac-GR00T)
source $ROOT/Isaac-GR00T/.venv/bin/activate
cd Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path $ROOT/Isaac-GR00T/models/$MODEL/$EXP/checkpoint-$CK \
    --embodiment-tag new_embodiment \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 終端機 B — client(~/env_isaaclab):印 Success Rate: X/Y (Z%)
source ~/env_isaaclab/bin/activate
cd Sim-to-Real-SO-101-Workshop
# ⚠️ 不要加 uv run:workshop 沒有 .venv,uv 會現場另建一個空環境然後找不到這支指令。
lerobot_eval_dual --task Lerobot-So101-Dual-Vials-To-Rack-Eval \
  --num_episodes 20 \
  --policy_host localhost \
  --policy_port 5555 \
  --steps_per_action 3 \
  --action_horizon 8
```

> `--steps_per_action 3` = 訓練資料(10 fps)相對錄製資料(30 fps)的降採樣倍率;改用 30 fps 資料訓練就要設 1。
>
> `--action_horizon 8` = receding-horizon 8/16(GR00T 自己 LIBERO 的配法)。模型一次預測 16 步,
> 只執行前 8 步就重新觀測、丟棄後 8 步。open-loop 從 48 env step(≈1.6 s)縮到 24 step(≈0.8 s),
> 抓偏了比較快能修。注意這個旗標**不改模型預測長度**,chunk 長度由 checkpoint 決定,它只是執行上限。
>
> 真機(§4d)配對用 `--execution_horizon 8`,兩邊一定要一致,否則 sim 排名不代表真機結果。

### 4d. 真機 eval 拍板(必跑)

**兩個環境不可混用**:server 用 Isaac-GR00T `.venv`(py3.12),client 用 `~/env_isaaclab`(py3.11)。
client 端要 LeRobot 才能開真機,所以跑在 env_isaaclab,不是 Isaac-GR00T 的 `.venv`。

> 一次性設定**已經做完**,不用再跑:`~/env_isaaclab` 裡已有 editable 的 `gr00t`、`msgpack`、`msgpack-numpy`。
> 換機器重建時才需要:`source ~/env_isaaclab/bin/activate` 後
> `uv pip install msgpack msgpack-numpy==0.4.8` +
> `uv pip install --no-deps -e /home/graphen/sim2real/Isaac-GR00T`。

```bash
# 終端機 A — server(Isaac-GR00T .venv)
ROOT=~/sim2real
EXP=pickvials-n1p7-r3-cotrain
MODEL=bimanual-pickvials-cotrain
CK=8000

cd $ROOT/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path $ROOT/Isaac-GR00T/models/$MODEL/$EXP/checkpoint-$CK \
    --embodiment-tag NEW_EMBODIMENT --port 5555

# 終端機 B — client(~/env_isaaclab,旗標用底線 draccus)
source ~/env_isaaclab/bin/activate
cd ~/sim2real/Isaac-GR00T/gr00t/eval/real_robot/SO101_bimanual

# 第一次帶電:限位設到最小,手放電源開關,只確認方向對、Ctrl-C 收得乾淨
python eval_so101_dual.py \
  --execution_horizon 8 \
  --max_relative_target '{"shoulder_pan":0.5,"shoulder_lift":0.5,"elbow_flex":0.5,"wrist_flex":0.5,"wrist_roll":0.5,"gripper":1.0}'

# 確認沒問題後全速跑,用 lerobot-replay 重播真人 demo 並排比速度
python eval_so101_dual.py --execution_horizon 8
```

> `--execution_horizon 8` 對應 sim eval(§4c)的 `--action_horizon 8`,兩邊必須一致,
> 否則 sim 的 checkpoint 排名不代表這裡的結果。模型一次預測 16 步,只執行前 8 步就重新規劃。
> (`--action_horizon` 在這支已 deprecated,會自動轉成 `--execution_horizon` 並印警告。)

⚠️ 有 `--model-path` 時 `--modality-config-path` 被忽略,config 來自 checkpoint。
⚠️ Ctrl-C 會讓手臂軟掉(torque 斷),夾著試管時避免中途按。

**跑法**:一次執行 = 一集,script 不會自己停,成功或卡住就 Ctrl-C。判定「90 秒內 4 支全進架子」,操作者/硬體失誤不計入分母。**N=20**(N=10 時 95% CI 太寬,分不出 30% 和 70%)。差距 <5–6% 是雜訊,不能下結論。

---

## STEP 5 — 手動重傳(log 出現 `Upload of ... failed` 時)

```bash
ROOT=~
REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain   # 依失敗的是哪個 run 換成 -sim-10fps / -sim / -cotrain
EXP=pickvials-n1p7-r3-cotrain
CK=$ROOT/Isaac-GR00T/checkpoints/so101_bimanual_cotrain/$EXP/checkpoint-10000
uv run hf upload $REPO \
  "$CK" "$EXP/$(basename "$CK")" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

---

## 參數對照表

| 參數                                                | 值                  | 理由                                                   |
| --------------------------------------------------- | ------------------- | ------------------------------------------------------ |
| `GLOBAL_BATCH_SIZE`                                 | 128(⚠️ 總量,÷ 卡數) | 資料集小(7,465–25,302 步),640 更新次數太少,見下        |
| `MAX_STEPS`                                         | 10000(三 run 一致)  | 落在官方 SO100(batch32×10k)和 LIBERO(batch640×20k)之間 |
| `SAVE_STEPS`/`SAVE_TOTAL_LIMIT`                     | 1000 / 10           | 剛好存滿,不被 rotate 掉                                |
| `--save-only-model`                                 | 要帶                | 不能 resume,optimizer state 純浪費磁碟                 |
| `DS_WEIGHTS_ALPHA`                                  | `0`(run 3 必設)     | 不設 = real 70% / sim 30%,`0` = 50/50。公式見下        |
| `DATALOADER_NUM_WORKERS`                            | 13                  | 112 vCPU ÷ 8 總 rank − 1                               |
| `learning_rate`                                     | 1e-4(預設)          | 官方 batch 32–1024 全用同一個值,不用縮放               |
| `state_dropout_prob`                                | 0.2(CLI 預設)       | 4a 顯示過擬合時優先調這個,而不是砍 steps               |
| 其他(`EPISODE_SAMPLING_RATE`/`SHARD_SIZE`/`tune_*`) | 預設,不動           | 動了會改變 shard 數,alpha 要重算                       |

**為什麼 batch 128 不是 640**:batch 只決定一次更新看幾個樣本,epoch 固定時 batch 越大更新次數越少。7,465 步的 run 1 若用 batch 640 只剩 2,500 次更新——比官方小資料集 recipe(SO100 batch32×10k)少 4 倍。128 才同時滿足「更新夠」和「不背答案」。

**實際在訓練的**:VLM backbone 凍結;action head + `vlln` + VL self-attention(51.54% 參數)。

### `DS_WEIGHTS_ALPHA`

$$p_{sim}=\frac{L_{sim}^{\alpha}}{L_{sim}^{\alpha}+L_{real}^{\alpha}} \quad\Longrightarrow\quad \alpha=\frac{\ln\!\big(\tfrac{p}{1-p}\big)}{\ln\!\big(L_{sim}/L_{real}\big)}$$

`L_sim=8`、`L_real=19`(shard 數,不是 frame 數;`--dataset-path` 順序不影響結果):

| alpha          | sim%   | real%                                         |
| -------------- | ------ | --------------------------------------------- |
| `1`(不設=預設) | 29.6   | 70.4                                          |
| **`0`**        | **50** | **50** ✅ 本輪用這個(兩邊 episode 數 100:101) |
| `-1.60`        | 80     | 20                                            |
| `-2.54`        | 90     | 10                                            |

⚠️ real 卡在 shard 邊界(18,448/1024=18.016,只超 16 steps),刪掉任何一集會掉到 18 shards,alpha 要重算——**每次以 dry-run log 印的 shard 數為準**。
