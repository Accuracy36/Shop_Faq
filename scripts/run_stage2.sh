#!/usr/bin/env bash
set -euo pipefail
# 第二阶段（领域适配）和全部评测。在项目根目录执行：bash scripts/run_stage2.sh
PY=${PY:-python}
MEM=${MEM:-0.55}
FINAL="--real --oos 60 --real-index --reject"
CK="--grad-ckpt"
adapt() {   # adapt 名称 其余参数
  name=$1; shift
  if [ -f "models/$name/history.json" ]; then echo "skip $name"; return; fi
  echo "=== $name $(date +%H:%M:%S)"
  PYTHONUNBUFFERED=1 $PY src/adapt_domain.py --name "$name" --steps 200 --eval-every 25 --mem-frac $MEM "$@" \
    > "results/logs/$name.log" 2>&1
  tail -1 "results/logs/$name.log" | cut -c1-120
}
evaluate() {   # evaluate 结果名 其余参数
  name=$1; shift
  if [ -f "results/eval/$name.json" ]; then return; fi
  $PY src/evaluate.py --name "$name" "$@" 2>&1 | grep -v "Redirects\|Loading"
}

# ---- 不需训练的基线
evaluate tfidf --baseline tfidf
evaluate bm25 --baseline bm25
evaluate bm25_real --baseline bm25 --real-index --reject
evaluate raw_roberta --baseline raw_roberta
evaluate raw_rbt3 --baseline raw_rbt3

# ---- 第一阶段各模型：只用标准问+相似问作索引
for m in roberta_seed1 roberta_seed2 roberta_seed3 rbt3_seed1 rbt3_seed2 rbt3_seed3 transformer_seed1 bilstm_seed1 \
         rbt3_cosent rbt3_infonce rbt3_cls; do
  evaluate s1_$m --model s1_$m
done

# ---- 领域适配的逐项对比（RoBERTa，种子 1）
evaluate s1_roberta_seed1_realidx --model s1_roberta_seed1 --real-index
evaluate s1_roberta_seed1_realidx_reject --model s1_roberta_seed1 --real-index --reject
adapt s2_faqonly_roberta_seed1 --mode faq --init s1_roberta_seed1 --lr 1e-5 $CK
evaluate s2_faqonly_roberta_seed1 --model s2_faqonly_roberta_seed1
adapt s2_simcse_roberta_seed1 --mode simcse --init s1_roberta_seed1 --lr 1e-5 $CK
evaluate s2_simcse_roberta_seed1 --model s2_simcse_roberta_seed1
adapt s2_real_roberta_seed1 --mode faq --init s1_roberta_seed1 --lr 1e-5 --real --real-index $CK
evaluate s2_real_roberta_seed1 --model s2_real_roberta_seed1 --real-index
adapt s2_nostage1_roberta_seed1 --mode faq --init raw_roberta --lr 1e-5 $FINAL $CK
evaluate s2_nostage1_roberta_seed1 --model s2_nostage1_roberta_seed1 --real-index --reject

# ---- 完整方案：各编码器、各随机种子
for m in roberta_seed1 roberta_seed2 roberta_seed3; do
  adapt s2_final_$m --mode faq --init s1_$m --lr 1e-5 --seed ${m: -1} $FINAL $CK
  evaluate s2_final_$m --model s2_final_$m --real-index --reject --latency
done
for m in rbt3_seed1 rbt3_seed2 rbt3_seed3; do
  adapt s2_final_$m --mode faq --init s1_$m --lr 2e-5 --seed ${m: -1} $FINAL
  evaluate s2_final_$m --model s2_final_$m --real-index --reject --latency
done
for m in transformer_seed1 bilstm_seed1; do
  adapt s2_final_$m --mode faq --init s1_$m --lr 1e-4 $FINAL
  evaluate s2_final_$m --model s2_final_$m --real-index --reject --latency
done
evaluate s2_final_roberta_seed1_threshold_only --model s2_final_roberta_seed1 --real-index

# ---- 伪标注（半监督）：用完整方案的模型给 ECD 无标签语料打伪标签，再训练一次
if [ ! -f results/pseudo/pseudo_seed1.json ]; then
  $PY src/adapt_domain.py --mode mine --init s2_final_roberta_seed1 --real-index --name pseudo_seed1 --mem-frac $MEM \
    2>&1 | grep -v "Redirects\|Loading"
fi
evaluate s2_final_roberta_seed1_pseudoidx --model s2_final_roberta_seed1 --real-index --reject --extra-index pseudo_seed1
adapt s2_pseudo_roberta_seed1 --mode faq --init s1_roberta_seed1 --lr 1e-5 $FINAL --pseudo pseudo_seed1 $CK
evaluate s2_pseudo_roberta_seed1 --model s2_pseudo_roberta_seed1 --real-index --reject --extra-index pseudo_seed1
echo "=== stage2 done $(date +%H:%M:%S)"
