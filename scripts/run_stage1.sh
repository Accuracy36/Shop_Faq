#!/usr/bin/env bash
set -euo pipefail
# 第一阶段全部训练（在项目根目录执行：bash scripts/run_stage1.sh）。已有 history.json 的设置会跳过。
# 显卡与其他程序共用（8 GB），因此限制本进程的显存（MEM），RoBERTa 启用梯度检查点（不改变计算结果）。
PY=${PY:-python}
MEM=${MEM:-0.32}
run() {
  name=$1; shift
  if [ -f "models/$name/history.json" ]; then echo "skip $name"; return; fi
  echo "=== $name $(date +%H:%M:%S)"
  PYTHONUNBUFFERED=1 $PY src/train_matcher.py --name "$name" --mem-frac $MEM "$@" > "results/logs/$name.log" 2>&1
  tail -1 "results/logs/$name.log" | cut -c1-200
}
ROBERTA="--arch roberta --epochs 1 --lr 3e-5 --evals-per-epoch 4 --grad-ckpt"
RBT3="--arch rbt3 --epochs 2 --lr 5e-5 --evals-per-epoch 2"
run s1_roberta_seed1     $ROBERTA --seed 1
run s1_rbt3_seed1        $RBT3 --seed 1
run s1_transformer_seed1 --arch transformer --epochs 10 --lr 5e-4 --seed 1
run s1_bilstm_seed1      --arch bilstm --epochs 10 --lr 1e-3 --seed 1
run s1_roberta_seed2     $ROBERTA --seed 2
run s1_roberta_seed3     $ROBERTA --seed 3
run s1_rbt3_seed2        $RBT3 --seed 2
run s1_rbt3_seed3        $RBT3 --seed 3
run s1_rbt3_cosent       $RBT3 --seed 1 --loss cosent
run s1_rbt3_infonce      $RBT3 --seed 1 --loss infonce
run s1_rbt3_cls          $RBT3 --seed 1 --pooling cls
echo "=== stage1 done $(date +%H:%M:%S)"
