# 简易购物自动问答系统（电商客服 FAQ 自动应答）

《深度学习实践》大作业，进阶题 4。学号：PB24000178　姓名：周宇翔

代码仓库：[Accuracy36/Shop_Faq](https://github.com/Accuracy36/Shop_Faq)。
数据标注使用 AI 辅助，尚不声称经过多名人工标注者复核；请按 `data/faq/ANNOTATION.md` 抽查。

最终材料：[PDF 报告](report/PB24000178+周宇翔+简易购物自动问答系统.pdf)、
[Word 报告](report/PB24000178+周宇翔+简易购物自动问答系统.docx)、
[57.6 秒 Demo 视频](demo/video/PB24000178+周宇翔+简易购物自动问答系统.mp4)。
训练历史与模型配置随仓库保存；完整模型权重保留在本机，可按下文步骤重新训练。

用户输入一句购物相关的问题，系统在店铺的 FAQ 知识库中找到意思最接近的标准问并返回它的答案；
找不到足够接近的问题时不作答，提示转人工并给出“猜你想问”。
知识库属于一家虚构的店铺“青柚生活旗舰店”（零食、日用品），共 60 条 FAQ。

方法是双塔句向量检索：同一个编码器把用户问句和知识库中的句子编码成向量，按余弦相似度检索。
编码器分两个阶段训练：先在公开的中文句对数据上学通用的句子匹配，再用店铺 FAQ 和标注过的真实客服问句做领域适配。

<!-- RESULTS -->

## 最终结果（2026-11-20）

| 模型 | 完整方案真实 Top-1 (%) | 端到端准确率 (%) |
| --- | ---: | ---: |
| BM25（不训练） | 69.46 ± 0.00 | 70.10 ± 0.00 |
| BiLSTM，从零训练 | 58.62 ± 0.00 | 66.58 ± 0.00 |
| Transformer，从零训练 | 56.65 ± 0.00 | 66.83 ± 0.00 |
| RBT3（3 层） | 80.13 ± 0.57 | 76.05 ± 0.73 |
| RoBERTa-wwm-ext（12 层） | 85.88 ± 1.24 | 78.81 ± 0.81 |

RoBERTa 与 RBT3 为三个种子的均值 ± 样本标准差；其他配置只运行一次。
复用 11 组第一阶段模型，完成 13 组第二阶段训练（每组 200 步）和 33 份评测。
本次第二阶段、全部评测、伪标注和绘图用时 65.5 分钟；8 项一致性核验通过。
SimCSE 的最优检查点为第 0 步；伪标注筛选保留 1680 句、覆盖 60 类，但未提升端到端准确率。
Demo 固定使用 `s2_final_roberta_seed1`、真实问句索引和拒识样例，阈值由验证集选择。
视频由脚本驱动真实推理录制，不要求本人出镜或配音；最终文件在 `demo/video/`。

<!-- END RESULTS -->

## 目录

```
data/faq/            FAQ 知识库、构造测试集、真实问句的训练/验证/测试集及标注文件（ANNOTATION.md 为标注说明）
data/raw/            下载的公开数据（不入库，scripts/download_data.sh 可重新下载）
data/processed/      整理后的句对、ECD 用户发言、字表（不入库，可由 prepare_data.py 重新生成）
src/common.py        路径、文本规范化、读写
src/prepare_data.py  数据整理与抽样
src/data_stats.py    数据统计与核验
src/models.py        编码器（BiLSTM、自己实现的 Transformer、预训练 BERT 类模型）与损失函数（CoSENT、InfoNCE）
src/retrieval.py     FAQ 索引、TF-IDF / BM25 基线、评测指标
src/train_matcher.py 第一阶段：公开句对上的通用句子匹配
src/adapt_domain.py  第二阶段：领域适配（有监督对比学习、SimCSE、伪标注）
src/evaluate.py      评测
src/plot_results.py  汇总结果并画图
src/draw_diagrams.py 系统流程图、模型结构图
scripts/             下载数据、两个阶段的批量运行脚本
demo/app.py          网页 Demo（Flask）；demo/static/index.html 为页面
demo/record_demo.py  自动录制演示视频
results/             评测结果（eval/）、汇总表（tables.json）、图（figures/）
```

## 数据来源

| 数据 | 来源 | 规模 | 在本项目中的用途 |
| --- | --- | --- | --- |
| LCQMC | 百度知道问题对（Liu et al., COLING 2018），取 C-MTEB 整理的版本 | 训练 22.6 万对 | 第一阶段训练、句对评测 |
| BQ Corpus | 银行客服问题对（Chen et al., EMNLP 2018），同上 | 训练 10.0 万对 | 同上 |
| AFQMC | 蚂蚁金融问题对（CLUE），同上 | 训练 3.4 万对 | 同上 |
| ECD | 淘宝客服对话（Zhang et al., COLING 2018），作者公布的下载链接 | 100 万条“对话-回复”样本 | 抽取真实用户问句：标注后作训练集、验证集、测试集；无标签部分用于 SimCSE 和伪标注 |
| FAQ 知识库 | 为本项目编写（虚构店铺，使用 AI 辅助） | 60 条，每条 1 句标准问、6 句相似问、3 句测试问 | 知识库、构造测试集 |

三个句对数据集规范化（去标点等）后，训练集中有 2903 对与验证集或测试集重复，已从训练集中去掉。

## 复现步骤

```powershell
pip install -r requirements.txt
bash scripts/download_data.sh                 # 下载数据和预训练模型
python src/prepare_data.py faq                # 解析 FAQ 源文件
python src/prepare_data.py pairs              # 整理句对
python src/prepare_data.py ecd                # 抽取 ECD 用户发言和待标注的评测候选
python src/prepare_data.py finalize           # 按标注生成真实问句验证集、测试集，建立字表
python src/prepare_data.py trainsample        # 抽取待标注的训练样本
python src/prepare_data.py finalize_train     # 按标注生成真实训练集
python src/data_stats.py                      # 数据统计与核验
bash scripts/run_stage1.sh                    # 第一阶段训练（11 组）
bash scripts/run_stage2.sh                    # 第二阶段训练和全部评测
python scripts/run_stage2.py                  # Windows 等平台可使用的等价队列；失败即停止，完整结果跳过
python src/plot_results.py --tsne             # 汇总并画图
python src/check_results.py                   # 模型轮次、预测汇总、数据划分与种子数核验
python demo/app.py --model s2_final_roberta_seed1 --real-index --reject      # 启动 Demo
python demo/record_demo.py --model s2_final_roberta_seed1 --real-index --reject   # 录制演示视频
```

标注文件（`data/faq/ecd_eval_annotation.txt`、`ecd_train_annotation.txt`）已随仓库提供，重新执行抽样步骤会得到同样的候选句子。

## 引用的代码与模型

- 预训练模型 `hfl/chinese-roberta-wwm-ext`、`hfl/rbt3`（哈工大讯飞联合实验室，Apache-2.0），通过 `transformers` 库载入，未修改其结构。
- CoSENT 损失按苏剑林博客《CoSENT：比 Sentence-BERT 更有效的句向量方案》中的公式实现；InfoNCE、SimCSE 按原论文实现。
- 其余代码（数据处理、Transformer 编码器、训练与评测、检索、Demo）为本项目编写。

## 最终交付与复现注意

最终 Word/PDF 在 `report/`，MP4 在 `demo/video/`，轻量源码与结果包在项目根目录。
源码包文件名为 `PB24000178+周宇翔+简易购物自动问答系统_源码与结果.zip`，文件清单见包内 `ARCHIVE_MANIFEST.json`。
源码包包含标注数据、模型配置/训练历史、逐条评测、图表和报告生成脚本；不含原始下载数据、`.pt` 权重和缓存。
本机完整训练权重保留在 `models/`；重建数据和模型请使用新的工作目录。恢复 Demo 时需同时提供训练权重和 `models/pretrained/` 的分词器/模型配置。
本机使用 `D:\APPS\miniconda\envs\dl2026\python.exe`。Windows 第二阶段优先用 `python scripts/run_stage2.py`；
`.sh` 命令需要 Git Bash，不能误用指向 WSL 的 Windows bash.exe。两个第二阶段入口二选一，无需重复运行。
录屏先执行 `python -m playwright install chromium --only-shell`。本机浏览器安装于 `.runtime/playwright`，录屏脚本自动识别。
报告生成另需 `python-docx` 和 `numpy`，运行 `python report/build_report.py`；在 Word 中导出同内容 PDF。
报告中的仓库地址为 https://github.com/Accuracy36/Shop_Faq 。课程材料由作者核对后自行提交。
发布时遵守 `.gitignore`，不要把原始大数据、模型权重、浏览器缓存或临时渲染文件上传到普通 Git 仓库。
