# GR00T N1.7 · SO-101 雙臂 fine-tune cheatsheet

登入 → 資料集 → dry-run → 訓練 → open-loop 健檢。**訓練完的 sim / 真機 rollout eval 不在這裡**,
在 [run_cheatsheet.md](../Sim-to-Real-SO-101-Workshop/run_cheatsheet.md) §6–§7。

| § | 做什麼 | 機器 |
| --- | --- | --- |
| [1](#1-login) | 環境檢查 + HF / W&B 登入 | 訓練機 |
| [2](#2-prepare-dataset) | 下載 → v3→v2 → modality → stats | 訓練機 |
| [3](#3-dry-run) | 300 步試跑 | 訓練機 |
| [4](#4-train) | 正式訓練 + 確認上傳 | 訓練機 |
| [5](#5-open-loop-check) | 下載 checkpoint,MSE 篩掉沒學好的 | 本機 |

全文用 **run 1(sim-only 10fps)** 寫死。換 run 改五處:`--experiment-name`、
`CUDA_VISIBLE_DEVICES`+`NUM_GPUS`、`MASTER_PORT`、`UPLOAD_TO_HUB_REPO`、`--dataset-path`
(併行時後四項要錯開)。co-train 的差異在 [§4](#4-train)。

> 🚫 **訓練機不能暫停,只能刪**(欠費會連 checkpoint 一起刪)。一定要帶 `UPLOAD_TO_HUB_REPO`,
> 確認 log 沒有 `Upload of ... failed` 才 `brev delete`。
> 📌 訓練機用 `$HOME/`(帳號名不確定),本機 §5 寫死 `/home/graphen/`。訓練機**不要改成 `~/`** ——
> `--dataset-path "a:b"` 這種冒號串接,`~` 不會展開。

| 資料集 | 規模 | fps |
| --- | --- | --- |
| `bimanual-so101-pickvials-real-10fps` | 101 ep,18,448 steps / **19 shards** | 10 |
| `bimanual-so101-pickvials-sim-10fps` | 100 ep,7,465 steps / **8 shards** | 10 |
| `bimanual-so101-pickvials-sim` | 100 ep,25,302 steps / **25 shards** | 30 |

---

## §1 Login

```bash
cd $HOME/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"
nvidia-smi -L | wc -l && which uv        # 卡數對、uv 在

# 先在網頁接受 gated repo(沒授權會在載入模型時 403):
#   https://huggingface.co/nvidia/Cosmos-Reason2-2B   ★ VLM backbone,必要
#   https://huggingface.co/nvidia/GR00T-N1.7-3B       base model
hf auth login     # 要 WRITE token(checkpoint 要上傳)
wandb login       # 不用就在 §3/§4 加 USE_WANDB=0 並移除 --wandb-project
```

> `uv` 找不到 → setup script 若以 root 跑,東西都在 `/root`,改用 `sudo -i`。

---

## §2 Prepare Dataset

★ 一定要在 `hf auth login` **之後**,否則匿名下載撞 429。co-train 就把整段換 `real-10fps` 再跑一次。

```bash
cd $HOME/Isaac-GR00T && source .venv/bin/activate

# download
uv run hf download ChihHanShen/bimanual-so101-pickvials-sim-10fps --repo-type dataset \
    --local-dir $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps

# lerobot v3 → v2
GIT_LFS_SKIP_SMUDGE=1 uv run --project scripts/lerobot_conversion --python 3.10 \
    python scripts/lerobot_conversion/convert_v3_to_v2.py \
    --repo-id bimanual-so101-pickvials-sim-10fps --root $HOME/Isaac-GR00T/datasets

# 蓋 modality.json + 算 relative action stats
cp examples/SO101_bimanual/modality.json \
   $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps/meta/modality.json
uv run python gr00t/data/stats.py \
    --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py

# 驗證,兩行都要 OK
python -c "import json;assert json.load(open('$HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps/meta/info.json'))['codebase_version']=='v2.1';print('v2.1 OK')"
test -f $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps/meta/relative_stats.json && echo "relative_stats OK"
```

> 下載到一半失敗 → `rm -rf $HOME/Isaac-GR00T/datasets/<該資料集>` 清掉半套再重跑。

---

## §3 Dry-run

```bash
cd $HOME/Isaac-GR00T && source .venv/bin/activate

USE_WANDB=1 NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 \
MAX_STEPS=300 SAVE_STEPS=150 SAVE_TOTAL_LIMIT=2 GLOBAL_BATCH_SIZE=128 \
DATALOADER_NUM_WORKERS=8 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim-10fps-dryrun \
UPLOAD_TO_HUB_PRIVATE=1 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $HOME/Isaac-GR00T/checkpoints/_dryrun
```

**四項都要過**:

- 300 步沒 OOM;`_dryrun/` 有 `checkpoint-150`/`-300`,**沒有** `global_step*/` 或 `optimizer.pt`
- log 的 `Generated N shards` 跟上面資料集表對得上(對不上 alpha 要重算)
- log 有 `Uploading checkpoint-300`,**沒有** `Upload of ... failed`
- 記下 `s/it` 跟 VRAM/卡 —— 估訓練時間與帳單,也決定能不能多 run 併行

co-train 多看一項:`Applied ds_weights_alpha=0 across 2 datasets`,mixture ≈ 50/50。

```bash
rm -rf $HOME/Isaac-GR00T/checkpoints/_dryrun
```

---

## §4 Train

`tmux new -s r1` 裡面跑。`Ctrl-b` `d` 離開,`tmux attach -t r1` 回來。

```bash
cd $HOME/Isaac-GR00T && source .venv/bin/activate && export PATH="$HOME/.local/bin:$PATH"

NUM_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=29500 \
GLOBAL_BATCH_SIZE=128 MAX_STEPS=10000 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=10 \
DATALOADER_NUM_WORKERS=8 \
UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim-10fps \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps \
  --modality-config-path examples/SO101_bimanual/so101_bimanual_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --save-only-model \
  --output-dir $HOME/Isaac-GR00T/checkpoints/so101_bimanual_cotrain \
  --wandb-project so101-bimanual \
  --experiment-name pickvials-n1p7-r1-sim10
```

⚠️ `MAX_STEPS` 開跑後不能改(cosine 退火到它,resume 接不回)。
⚠️ `GLOBAL_BATCH_SIZE` 是**總量**會除以卡數;卡數只能 1/2/4/8(有 assert)。

**co-train(real + sim 50/50)改這三處**,其餘照抄:

```bash
DS_WEIGHTS_ALPHA=0 \                    # 1) 開頭多這個環境變數
  --dataset-path "$HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps:$HOME/Isaac-GR00T/datasets/bimanual-so101-pickvials-sim-10fps" \
                                        # 2) ★ 一定要雙引號 + $HOME,用 ~ 第二個路徑不會展開
  UPLOAD_TO_HUB_REPO=ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-cotrain \
  --experiment-name pickvials-n1p7-r3-cotrain      # 3) 換 repo 跟實驗名
```

**收尾(手動)**:log 要有 `Uploading checkpoint-10000` 且**沒有** `Upload of ... failed`,才能刪機器。

```bash
brev delete <INSTANCE_NAME>     # 名稱用 brev ls 查——不能 stop,只能刪

# 上傳失敗才跑這段補傳,確認 HF 上完整了再刪機器
uv run hf upload ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim-10fps \
  "$HOME/Isaac-GR00T/checkpoints/so101_bimanual_cotrain/pickvials-n1p7-r1-sim10/checkpoint-10000" \
  "pickvials-n1p7-r1-sim10/checkpoint-10000" --repo-type model \
  --exclude 'global_step*' --exclude '*optimizer.pt'
```

---

## §5 Open-loop Check

**在本機跑,不是訓練機。** 用對真機軌跡的 MSE 篩掉明顯沒學好的 checkpoint。
⚠️ MSE **只能篩,不能排名** —— 排名看 [run_cheatsheet.md](../Sim-to-Real-SO-101-Workshop/run_cheatsheet.md) §6–§7 的 success rate。

```bash
source .venv/bin/activate

# 5-1  下載(一個 6.5 G,10 個 = 65 G,先 df -h .)
mkdir -p /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-sim-10fps
for ckpt in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  hf download ChihHanShen/gr00t-n1.7-so101-bimanual-pickvials-sim-10fps \
    --include "pickvials-n1p7-r1-sim10/checkpoint-$ckpt/*" \
    --exclude "pickvials-n1p7-r1-sim10/checkpoint-$ckpt/global_step*" \
    --local-dir /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-sim-10fps
done

# 5-2  ★ 驗大小,每個都要 6.5G。7M = repo/實驗名打錯或下載中斷,只有 config 沒有權重
du -sh /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-sim-10fps/pickvials-n1p7-r1-sim10/checkpoint-*

# 5-3  逐個算 MSE。⚠️ 不要加 uv run —— 它會重解依賴去抓 flash-attn 的 GitHub wheel,
#      連不到就每個 checkpoint 卡 74 秒然後失敗(那顆 wheel 還是 aarch64,x86 裝不起來)
mkdir -p /home/graphen/sim2real/Isaac-GR00T/open_loop_eval/bimanual-pickvials-sim-10fps
for ckpt in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  echo "=== checkpoint-$ckpt ==="
  python gr00t/eval/open_loop_eval.py \
    --dataset-path /home/graphen/sim2real/Isaac-GR00T/datasets/bimanual-so101-pickvials-real-10fps \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /home/graphen/sim2real/Isaac-GR00T/models/bimanual-pickvials-sim-10fps/pickvials-n1p7-r1-sim10/checkpoint-$ckpt \
    --traj-ids 0 --execution-horizon 8 --steps 400 \
    --save-plot-path /home/graphen/sim2real/Isaac-GR00T/open_loop_eval/bimanual-pickvials-sim-10fps/checkpoint-$ckpt.jpeg
done 2>&1 | tee /home/graphen/sim2real/Isaac-GR00T/open_loop_eval/bimanual-pickvials-sim-10fps/eval_output.txt

# 5-4  所有 run 一起比
grep -H "Average MSE" /home/graphen/sim2real/Isaac-GR00T/open_loop_eval/*/eval_output.txt
```

MSE 持續下降 = 還在學,走平 = 收斂,回升 = 過擬合(優先調 `state_dropout_prob`,不要砍 `MAX_STEPS`)。

⚠️ 比較基準要固定:**同一個 dataset + 同一個 `--execution-horizon`**。horizon 越大 open-loop 漂移越久、
MSE 天生越高,**不同 horizon 的數字不能互相比**。`8` 對齊 §6–§7 的 eval 設定,改了就要全部重跑。

---

## 附錄 — 參數為什麼是這些值

| 參數 | 值 | 理由 |
| --- | --- | --- |
| `GLOBAL_BATCH_SIZE` | 128(總量) | 資料集小(100 episodes) 640 的更新次數太少 |
| `MAX_STEPS` | 10000 | 落在官方 SO100(batch32×10k)與 LIBERO(batch640×20k)之間 |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 1000 / 10 | 剛好存滿 10 個,不被 rotate 掉 |
| `--save-only-model` | 要帶 | 不能 resume,optimizer state 純浪費磁碟 |
| `DS_WEIGHTS_ALPHA` | `0`(只有 co-train) | 不設 = real 70% / sim 30%;`0` = 50/50 |
| `DATALOADER_NUM_WORKERS` | 8 | vCPU 數 ÷ 總 rank − 1 |

實際在訓練的:VLM backbone 凍結,只動 action head + `vlln` + VL self-attention(51.54% 參數)。
⚠️ 官方註明同樣設定重跑有 5–6% 變異,run 之間差距在這範圍內都算噪音。

**`DS_WEIGHTS_ALPHA`** —— `L` 是 **shard 數**(不是 frame 數),以 `L_sim=8`、`L_real=19` 為例:
`α=1`(不設)→ sim 29.6%;**`α=0` → 50/50**;`α=-1.60` → sim 80%。

$$p_{sim}=\frac{L_{sim}^{\alpha}}{L_{sim}^{\alpha}+L_{real}^{\alpha}} \quad\Longrightarrow\quad \alpha=\frac{\ln\!\big(\tfrac{p}{1-p}\big)}{\ln\!\big(L_{sim}/L_{real}\big)}$$

⚠️ shard 數會變:real 卡在邊界(18,448/1024 = 18.016),刪任何一集就掉到 18 shards,alpha 要重算 ——
**一律以 dry-run log 印的 shard 數為準**。
