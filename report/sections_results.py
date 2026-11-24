"""Final analysis generated exclusively from saved experiment results."""
import json
from pathlib import Path


def complete_report(doc, root, stats):
    results = root / 'results'
    fig = results / 'figures'
    tables = json.loads((results / 'tables.json').read_text(encoding='utf-8'))

    def read(name):
        p = results / name
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

    def cell(row, key, scale=100, digits=1):
        if not row or key not in row:
            return '—'
        value = row[key]
        m, s = value['mean'] * scale, value['std'] * scale
        return f'{m:.{digits}f}±{s:.{digits}f}' if row['n'] > 1 else f'{m:.{digits}f}'

    def picture(filename, caption, width=7.5):
        if (fig / filename).exists():
            doc.figure(fig / filename, width, caption)

    doc.h1('5  实验结果与分析')
    doc.para('5.1  编码器与两阶段训练', indent=False, keep=True)
    doc.para('以下表格均由results/eval中的实际评测汇总。百分数以0–100表示，MRR与AUROC也按百分数展示；'
             '三个随机种子的结果写为均值±样本标准差，单次实验不附标准差。横线表示该配置不适用或未产生该项结果。'
             '所有模型和回答阈值按验证集选定；测试集用于报告固定方案的表现，不据测试结果重新挑选种子。')
    rows = []
    for r in tables['encoders']:
        a, b = r['stage1'], r['final']
        rows.append([r['label'], cell(a, 'pair_auc'), cell(a, 'real_top1'), cell(a, 'e2e'),
                     cell(b, 'real_top1'), cell(b, 'e2e')])
    doc.table_caption('表6  编码器比较及领域适配前后结果 单位为百分数')
    doc.table(['编码器', '阶段一\n句对AUC', '阶段一\nTop1', '阶段一\n端到端', '完整方案\nTop1', '完整方案\n端到端'], rows,
              [2426, 1320, 1320, 1320, 1320, 1320], ['left'] + ['center'] * 5)
    rb = next(r for r in tables['encoders'] if r['label'].startswith('RoBERTa-wwm'))
    rt = next(r for r in tables['encoders'] if r['label'].startswith('RBT3'))
    doc.para(f'RoBERTa第一阶段的真实问句Top-1为 {cell(rb["stage1"], "real_top1")}%、端到端准确率为 '
             f'{cell(rb["stage1"], "e2e")}%；完整方案分别为 {cell(rb["final"], "real_top1")}%和 '
             f'{cell(rb["final"], "e2e")}%。RBT3完整方案端到端准确率为 {cell(rt["final"], "e2e")}%。'
             '这里“完整方案”同时包含领域训练、真实问句索引和拒识样例，因此前后差异不能全部归因于微调；逐项对照见5.3节。')
    doc.para('BM25的完整配置只增加真实问句索引与拒识样例，没有神经网络领域训练。随机初始化的BiLSTM和Transformer'
             '在该设置下仍落后于两个预训练编码器；这支持预训练在当前小规模领域数据上的价值，但不说明其对所有任务都最优。')
    picture('encoder_comparison.png', '图4  不同编码器在真实问句上的检索与端到端表现')
    picture('training_stage1.png', '图5  第一阶段训练损失与验证指标')
    picture('training_stage2.png', '图6  第二阶段验证指标与训练进程')
    rows = []
    for r in tables['encoders']:
        b = r['final']
        if b:
            rows.append([r['label'], cell(b, 'constructed_top1'), cell(b, 'real_mrr'),
                         cell(b, 'oos_reject'), cell(b, 'answer_precision'), cell(b, 'auroc')])
    doc.table_caption('表7  完整方案的覆盖与拒答表现 单位为百分数')
    doc.table(['编码器', '构造集\nTop1', '真实集\nMRR', '范围外\n拒答率', '已回答\n正确率', '范围内外\nAUROC'],
              rows, [2426, 1320, 1320, 1320, 1320, 1320], ['left'] + ['center'] * 5)
    doc.para('构造集只包含知识库范围内的问题且每条FAQ都有样本，真实集则包含上下文缺失、商品特定问题和范围外陈述。'
             '两者分布不同，不能因为构造集分数较高就推断真实客服场景同样可靠。Top-1只看范围内样本的排序，'
             '端到端指标还要求范围外拒答，回答正确率则只看系统选择回答的问句，三者需要结合使用。')

    doc.para('5.2  损失函数与池化方式', indent=False, keep=True)
    doc.table_caption('表8  RBT3第一阶段损失与池化对照 种子1')
    doc.table(['配置', '句对AUC', '真实Top1', '真实MRR', '端到端准确率'],
              [[r['label']] + [cell(r['stage1'], k) for k in ['pair_auc', 'real_top1', 'real_mrr', 'e2e']]
               for r in tables['loss']], [3426, 1400, 1400, 1400, 1400], ['left'] + ['center'] * 4)
    doc.para('CoSENT利用句对是否同义的排序约束，InfoNCE直接优化批内检索目标。不同指标的最优配置可能不同，'
             '说明通用句对判别能力并不等同于店铺FAQ检索能力。这里每项消融只有种子1，观察到的差异也可能包含随机波动，'
             '不据单次差异声称统计显著。')
    losses = tables['loss']
    doc.para(f'单用CoSENT的真实Top-1为 {cell(losses[0]["stage1"], "real_top1")}%，高于两损失相加的 '
             f'{cell(losses[2]["stage1"], "real_top1")}%；但两损失相加的端到端准确率为 '
             f'{cell(losses[2]["stage1"], "e2e")}%，高于单用CoSENT的 {cell(losses[0]["stage1"], "e2e")}%。'
             '平均池化和CLS池化的差异较小，当前单种子实验不足以建立稳定排序。')

    doc.para('5.3  索引 拒答与领域适配的逐项对照', indent=False, keep=True)
    doc.table_caption('表9  RoBERTa领域适配对照 种子1 单位为百分数')
    doc.table(['配置', '真实Top1', '端到端\n准确率', '范围外\n拒答率', '已回答\n正确率'],
              [[r['label']] + [cell(r['stage1'], k) for k in ['real_top1', 'e2e', 'oos_reject', 'answer_precision']]
               for r in tables['adaptation']], [3826, 1300, 1300, 1300, 1300], ['left'] + ['center'] * 4)
    a = tables['adaptation']
    doc.para(f'固定第一阶段模型，仅将真实问句加入索引，端到端准确率由 {cell(a[0]["stage1"], "e2e")}%'
             f'变为 {cell(a[1]["stage1"], "e2e")}%；再加入拒识样例为 {cell(a[2]["stage1"], "e2e")}%。'
             f'完整方案为 {cell(a[7]["stage1"], "e2e")}%。这些对照把索引变化与训练变化分别记录，'
             '避免把全部收益归到编码器本身。')
    doc.para('拒答阈值在验证集上选择，不在测试集重新调节。拒识样例是一种基于已知范围外语句的判别补充，'
             '它可能减少误答，也可能使本应回答的问句被拒绝；所以同时展示范围外拒答率和已回答正确率。'
             '无监督SimCSE只利用句子的两次Dropout表示，不直接接触FAQ标签，其效果由实际对照决定。')
    sim = json.loads((root / 'models/s2_simcse_roberta_seed1/history.json').read_text(encoding='utf-8'))
    doc.para(f'SimCSE实际训练了200步，但验证集选中的检查点是第 {sim["best"]["step"]} 步，因此表9中的结果与原第一阶段模型相同。'
             '这表示本次适配未优于起点，不能把“训练已完成”写成“性能已经提升”。')
    picture('similarity_distribution.png', '图7  范围内与范围外问句的相似度分布')
    picture('roc_threshold.png', '图8  范围内外判别曲线及预选阈值在测试集上的敏感性分析')

    doc.para('5.4  伪标注', indent=False, keep=True)
    pseudo = read('pseudo/pseudo_seed1.json')
    if pseudo:
        p = pseudo['stats']
        doc.para('伪标注的置信度阈值先在真实验证集上确定：保留样本不少于20句，且所选样本Top-1正确率至少95%。'
                 f'本次保存的伪标注统计为：候选 {p["candidates"]:,} 句，阈值 {p["threshold"]:.4f}，'
                 f'验证集选中 {p["dev_selected"]} 句、正确率 {100*p["dev_precision"]:.1f}%；'
                 f'最终保留 {p["selected"]} 句，覆盖 {p["faqs_covered"]} 条FAQ，每条最多30句。')
        doc.para('表9分别列出只将伪标注加入索引、以及同时加入训练和索引的结果。验证集上的高精度约束并不保证'
                 '无标签语料上的实际标签同样准确；语料中的高频寒暄、语义歧义和类别不平衡仍可能使伪标签引入噪声。'
                 '不以验证集筛选精度冒充伪标注全集的真实准确率。')
        doc.para(f'完整方案端到端准确率为 {cell(a[7]["stage1"], "e2e")}%，只扩充伪标注索引后为 '
                 f'{cell(a[9]["stage1"], "e2e")}%，伪标注同时参与训练和索引后为 {cell(a[10]["stage1"], "e2e")}%。'
                 '本次伪标注没有提升端到端准确率，因此Demo继续使用预先指定的无伪标注完整方案。')
    else:
        doc.para('本次伪标注未产生满足预设条件的结果文件，因而不继续训练依赖该文件的模型。'
                 '表9相关项保留为空；未通过降低95%精度要求来填满表格。具体退出信息见results/logs/completion_pseudo_mine.log。')

    doc.h1('6  误例 可视化与部署')
    errors = read('error_cases.json')
    if errors:
        c = errors['counts']
        doc.para('固定RoBERTa完整方案种子1，在真实测试集上的结果为：' + '，'.join(
                 f'{label}{c.get(key, 0)}句' for key, label in [('correct', '答对'), ('rejected', '正确拒答'),
                 ('wrong_answer', '范围内答错'), ('missed', '范围内漏答'), ('false_answer', '范围外误答')]) + '。')
        rows = []
        for key, label in [('wrong_answer', '范围内答错'), ('missed', '范围内漏答'), ('false_answer', '范围外误答')]:
            for r in errors.get(key, [])[:3]:
                rows.append([label, r['query'], r['gold'], r['predicted'], f'{r["confidence"]:.3f}'])
        doc.table_caption('表10  各类错误按测试集原顺序选取的前3例')
        doc.table(['类型', '用户问句', '主标签', '最高分FAQ', '得分'], rows,
                  [1300, 3226, 1800, 1800, 900], ['left', 'left', 'left', 'left', 'center'])
        doc.para('错误表中的“最高分FAQ”是排序第一的候选，不一定最终被系统回答；漏答样例仍会列出该候选。'
                 '得分为−1表示拒识样例比所有FAQ更相似，程序将决策分数设为拒答标记，并非余弦相似度恰为−1。'
                 '主标签只用于展示，评测时允许命中该问句任一可接受FAQ。错误不能只靠提高或降低阈值同时消除，'
                 '还涉及知识库覆盖范围、单轮输入缺少上下文、近似意图之间的区分以及标注歧义。')
    picture('category_outcomes.png', '图9  真实测试集按问题类别的结果')
    picture('tsne.png', '图10  固定样本句向量的t-SNE可视化')
    doc.para('t-SNE用于观察固定样本在二维投影中的聚散，不保留全部高维距离。图中局部聚集只能提供直观解释，'
             '不能替代独立测试集指标，也不据图选择模型。少样本类别的准确率同样应结合实际句数理解。')
    rows = []
    for r in tables['encoders']:
        if r['final'] and 'latency_cpu' in r['final']:
            rows.append([r['label'], cell(r['final'], 'parameters', 1/1e6, 2),
                         cell(r['final'], 'latency_cpu', 1, 2), cell(r['final'], 'latency_cuda', 1, 2)])
    doc.table_caption('表11  完整方案参数量与单请求推理时延')
    doc.table(['模型', '参数量 百万', 'CPU 毫秒', 'GPU 毫秒'], rows,
              [3426, 1600, 2000, 2000], ['left', 'center', 'center', 'center'])
    doc.para('时延在索引已建立后测量，先预热10个请求，再顺序测试100句真实问句；CPU固定线程数，GPU计时前后同步。'
             '包括编码与向量检索，不包括首次载入模型、建立索引和浏览器网络往返；它是本机低并发测量，不是生产环境吞吐量。')
    doc.para('较小的BiLSTM和自写Transformer在本次单句请求中CPU反而更快，说明启动与同步等固定开销会影响GPU收益。'
             f'RoBERTa的GPU平均时延为 {cell(rb["final"], "latency_cuda", 1, 2)} 毫秒，RBT3为 '
             f'{cell(rt["final"], "latency_cuda", 1, 2)} 毫秒；模型选择需要同时考虑准确率与实际请求规模。')
    picture('latency.png', '图11  CPU与GPU的单请求推理时延')
    doc.para('网页Demo固定使用完整方案RoBERTa种子1，启用真实样例索引和拒识样例；这个选择沿用实验计划，'
             '不按测试集挑选表现最好的种子。界面展示系统实际返回的FAQ答案、拒答和建议问题。演示脚本仅自动输入与点击，'
             '每次应答均调用已训练模型实时推理，不是预置答案视频。')

    doc.h1('7  复现 核验与局限')
    check = read('verification.json')
    if check:
        doc.para('结果核验：' + '；'.join(f'{k}：{"通过" if v else "未通过"}'
                 for k, v in check['checks'].items()) + '。详细记录见results/verification.json。')
    doc.para('复现顺序为安装requirements.txt、下载公开数据与预训练模型、整理数据、运行阶段一、阶段二和评测、汇总图表、'
             '启动Demo及录屏。README给出了具体命令，scripts目录保留批量运行入口。训练完成的history.json、评测json和逐条预测'
             '可用于核对报告中的数值；随机种子固定仍不保证不同硬件和软件版本逐位一致。')
    doc.para('公开句对和预训练模型用于迁移学习，知识库与标注文件为本项目整理。真实客服问句标签由AI助手逐条辅助标注，'
             '尚不能声称经过用户或多名人工标注者逐条复核。多标签规则降低了部分歧义，但仍存在主观判断偏差；'
             '因此报告给出的分数应理解为在这套标注口径下的表现，后续应进行人工抽查及独立复核。')
    doc.para('系统限于虚构店铺的60条FAQ，不能查询订单状态、库存或物流接口，也不能处理多轮对话中的指代。'
             '从ECD抽样得到的真实测试集规模有限，拒答策略只能覆盖已见或相似的范围外表达。高相似度不等于答案一定正确，'
             '上线仍需要知识库维护、转人工流程和实际业务数据验证。')
    doc.para('本项目完成了从公开句对训练到店铺领域适配、带拒答的FAQ检索及网页演示的流程。结果说明应同时评估'
             '句子匹配、知识库范围内排序和范围外拒答，并用逐项消融区分训练、索引与拒识样例的作用；'
             '方法的有效范围由当前数据、标注和实验设置限定。')
    doc.refs_heading('参考资料与代码来源')
    for text in [
        '[1] Liu X et al. LCQMC A Large scale Chinese Question Matching Corpus. COLING 2018. https://aclanthology.org/C18-1166/',
        '[2] Chen J et al. The BQ Corpus A Large scale Domain specific Chinese Corpus for Sentence Semantic Equivalence Identification. EMNLP 2018. https://aclanthology.org/D18-1536/',
        '[3] CLUE Benchmark. AFQMC数据与说明. https://github.com/CLUEbenchmark/CLUE',
        '[4] 本项目公开句对读取C-MTEB整理的数据版本，下载地址与处理步骤见scripts/download_data.sh和src/prepare_data.py。',
        '[5] ECD 电商客服对话语料，作者数据来源及原始说明保留在data/raw/ECD/README.md，下载入口见scripts/download_data.sh。',
        '[6] Cui Y et al. Chinese BERT with Whole Word Masking. https://github.com/ymcui/Chinese-BERT-wwm 预训练模型hfl/chinese-roberta-wwm-ext与hfl/rbt3通过transformers载入。',
        '[7] 苏剑林. CoSENT 比Sentence-BERT更有效的句向量方案. https://kexue.fm/archives/8847 按文中排序损失公式实现。',
        '[8] Gao T, Yao X, Chen D. SimCSE Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021. https://arxiv.org/abs/2104.08821',
        '[9] van den Oord A, Li Y, Vinyals O. Representation Learning with Contrastive Predictive Coding. 2018. https://arxiv.org/abs/1807.03748',
        '[10] Vaswani A et al. Attention Is All You Need. NeurIPS 2017. https://arxiv.org/abs/1706.03762',
        '[11] 自写Transformer、数据处理、训练评测、检索与Demo代码随项目提供；预训练模型结构未作修改。代码整理和标注使用AI辅助，实验数值来自实际执行。',
    ]:
        doc.ref(text)
