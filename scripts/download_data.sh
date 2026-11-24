#!/usr/bin/env bash
# 下载公开数据和预训练模型（在项目根目录执行：bash scripts/download_data.sh）。
# 数据来源：
#   LCQMC / BQ / AFQMC：Hugging Face 上 C-MTEB 整理的版本（datasets/C-MTEB/LCQMC、BQ、AFQMC）
#   ECD（E-commerce Dialogue Corpus，淘宝客服对话）：Zhang et al., COLING 2018，作者在
#     https://github.com/cooelf/DeepUtteranceAggregation 公布的 Google Drive 链接
#   预训练模型：hfl/chinese-roberta-wwm-ext、hfl/rbt3（Apache-2.0）
set -e
PY=${PY:-python}
hf() { mkdir -p "$(dirname "$2")"; curl -sS -L -o "$2" "https://huggingface.co/$1"; echo "$2"; }
hf datasets/C-MTEB/LCQMC/resolve/main/data/train-00000-of-00001-83493a6b9e29a4d0.parquet      data/raw/LCQMC/train.parquet
hf datasets/C-MTEB/LCQMC/resolve/main/data/validation-00000-of-00001-ae04bea7d65ea894.parquet data/raw/LCQMC/validation.parquet
hf datasets/C-MTEB/LCQMC/resolve/main/data/test-00000-of-00001-cb3f2553acbb3663.parquet       data/raw/LCQMC/test.parquet
hf datasets/C-MTEB/BQ/resolve/main/data/train-00000-of-00001-eec36d2dc3f2a1c2.parquet         data/raw/BQ/train.parquet
hf datasets/C-MTEB/BQ/resolve/main/data/validation-00000-of-00001-7f96017a1c50f024.parquet    data/raw/BQ/validation.parquet
hf datasets/C-MTEB/BQ/resolve/main/data/test-00000-of-00001-27441f0c8fdc9664.parquet          data/raw/BQ/test.parquet
hf datasets/C-MTEB/AFQMC/resolve/main/data/train-00000-of-00001-da0bf29c7c279c45.parquet      data/raw/AFQMC/train.parquet
hf datasets/C-MTEB/AFQMC/resolve/main/data/validation-00000-of-00001-b8fc393b5ddedac7.parquet data/raw/AFQMC/validation.parquet
hf datasets/C-MTEB/AFQMC/resolve/main/data/test-00000-of-00001-5a4d6d31cf610204.parquet       data/raw/AFQMC/test.parquet
for m in chinese-roberta-wwm-ext rbt3; do
  for f in config.json vocab.txt tokenizer.json tokenizer_config.json special_tokens_map.json pytorch_model.bin; do
    hf hfl/$m/resolve/main/$f models/pretrained/$m/$f
  done
done
mkdir -p data/raw/ECD
$PY -m gdown 154J-neBo20ABtSmJDvm7DK0eTuieAuvw -O data/raw/ECD/ECD.zip
# ECD.zip 的 SHA-256：41f09f0312b7611863794fb46d9f7f4061d8171901232ac3c17929295e4878b2
(cd data/raw/ECD && unzip -q -o ECD.zip && mv "E-commerce dataset"/* . && rmdir "E-commerce dataset")
echo done
