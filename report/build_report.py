"""生成大作业报告（Word）。排版沿用课程实验报告的模板；数字全部从 results/ 下的文件读取。

    python report/build_report.py            # 生成 report/ 下的 docx
第 1–4 节只依赖数据统计（results/data_stats.json）；第 5 节以后需要训练和评测结果，
由 report/sections_results.py 提供（训练完成后编写）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding='utf-8')

from common import FAQ_DIR, RESULTS, load_faq, read_json, read_jsonl  # noqa: E402
from docxkit import Report, omr, omsub, omfrac  # noqa: E402

TEMPLATE = Path(r'D:\!WorkSpace\2026 Fall Deep Learning\HWs\4\PB24000178_周宇翔_4.docx')
TITLE = '简易购物自动问答系统'
OUT = Path(__file__).resolve().parent / f'PB24000178+周宇翔+{TITLE}.docx'
GIT_URL = 'https://github.com/Accuracy36/Shop_Faq'
FIG = RESULTS / 'figures'
S = read_json(RESULTS / 'data_stats.json')
FAQ = load_faq()


def n(value):
    return f'{value:,}'


def wan(value):
    return f'{value / 10000:.1f} 万'


doc = Report(TEMPLATE)
doc.title(f'深度学习实践大作业 {TITLE}')
doc.idline('进阶题 4　电商客服 FAQ 自动应答　　学号：PB24000178　姓名：周宇翔')
doc.para(f'代码仓库：{GIT_URL}', indent=False)
doc.para(f'演示视频：PB24000178+周宇翔+{TITLE}.mp4（随文档一并提交）', indent=False)

# ================================================================ 1
doc.h1('1  选题与任务')
doc.para('选题为进阶题 4“开发一个简易购物自动问答系统（电商客服 FAQ 自动应答）”。店铺客服每天收到的问题大部分是重复的：'
         '发什么快递、什么时候发货、能不能退。这类问题有固定答案，可以整理成 FAQ（常见问题与标准答案）。'
         '自动应答要做的事是：用户用自己的话问一句，系统判断它问的是哪一条 FAQ，把那一条的答案回给用户；'
         '如果知识库里没有对应的问题，就不要硬答，转给人工。')
doc.para('难点有两个。第一，同一个意思有很多种说法，“发什么快递”“你家走哪个物流”“用哪家寄”字面上几乎不重合，'
         '按关键词匹配会漏掉。第二，真实对话里大约一半的句子并不是在问 FAQ（“好的那我拍了”“这个是纯棉的吗”），'
         '系统必须能判断“这句话我答不了”，否则答非所问比不回答更糟。')
doc.para('本项目把它做成一个句向量检索问题：训练一个编码器，把句子变成向量，意思相近的句子向量也相近；'
         '用户的问句和知识库里的句子用同一个编码器编码，按余弦相似度找最相近的一条，相似度不够高就拒答。'
         f'知识库属于一家虚构的店铺“青柚生活旗舰店”（零食、日用品），共 {len(FAQ)} 条 FAQ，为本项目编写。'
         '最后做成了一个网页 Demo，输入问题即可得到回答，并能看到模型给出的相似度。')

# ================================================================ 2
doc.h1('2  数据集说明')
doc.para('项目用到四类数据：公开的中文句对数据集（学通用的“两句话是不是一个意思”）、淘宝客服对话语料（提供真实的用户问句）、'
         '为本项目编写的 FAQ 知识库、以及从淘宝语料中抽取并由 AI 辅助标注的真实问句。表 1 列出来源和规模。')
pair = S['pairs']
doc.table_caption('表 1  数据来源与规模')
doc.table(
    ['数据', '来源', '规模', '用途'],
    [['LCQMC', '百度知道问题对，COLING 2018；取 Hugging Face 上 C-MTEB 整理的版本',
      f'训练 {n(pair["train"]["LCQMC"]["total"])} 对，验证 {n(pair["dev"]["LCQMC"]["total"])}，测试 {n(pair["test"]["LCQMC"]["total"])}',
      '第一阶段训练；句对评测'],
     ['BQ Corpus', '银行客服问题对，EMNLP 2018；同上',
      f'训练 {n(pair["train"]["BQ"]["total"])} 对，验证 {n(pair["dev"]["BQ"]["total"])}，测试 {n(pair["test"]["BQ"]["total"])}',
      '同上'],
     ['AFQMC', '蚂蚁金融问题对，CLUE 基准；同上',
      f'训练 {n(pair["train"]["AFQMC"]["total"])} 对，测试 {n(pair["test"]["AFQMC"]["total"])}', '同上'],
     ['ECD', '淘宝客服对话（E-commerce Dialogue Corpus），COLING 2018，作者公布的下载链接',
      f'训练部分 {n(S["ecd"]["train_sessions"])} 段对话，去重后 {n(S["ecd"]["train_distinct"])} 句用户发言',
      '抽取真实用户问句'],
     ['FAQ 知识库', '项目编写（虚构店铺）', f'{len(FAQ)} 条；标准问 {len(FAQ)} 句，相似问 {S["faq"]["index_texts"] - len(FAQ)} 句',
      '知识库与索引'],
     ['构造测试集', '项目编写', f'{S["constructed"]["n"]} 句（每条 FAQ 3 句）', '测试'],
     ['真实问句', '从 ECD 抽取，AI 辅助标注',
      f'训练 {S["real"]["train"]["n"]} 句，验证 {S["real"]["dev"]["n"]} 句，测试 {S["real"]["test"]["n"]} 句',
      '第二阶段训练、选阈值、测试']],
    [1100, 3126, 2900, 1900], ['center', 'left', 'left', 'left'])

doc.para('2.1  公开句对数据集', indent=False, keep=True)
doc.para('三个数据集的每条样本都是两句话加一个 0/1 标签（是否同义）。LCQMC 来自百度知道，话题很杂；'
         'BQ 和 AFQMC 来自银行和支付的客服场景，句式更接近客服对话，但都不是购物领域。'
         'AFQMC 官方测试集不公开标签，因此用它的验证集作测试集；LCQMC、BQ 沿用官方的训练、验证、测试划分。'
         f'三者合并后训练集共 {n(sum(v["total"] for v in pair["train"].values()))} 对，'
         f'正例占 {100 * sum(v["positive"] for v in pair["train"].values()) / sum(v["total"] for v in pair["train"].values()):.1f}%。')

doc.para('2.2  淘宝客服对话语料（ECD）', indent=False, keep=True)
doc.para('ECD 是从淘宝真实客服对话中整理出来的多轮对话数据，原本用于“给定对话上文，从候选中选出正确回复”的任务。'
         '每条样本是若干轮对话加一句回复。回复由客服发出，往前数第 1、3、5……句是用户说的话，我只取这些用户发言。'
         f'训练部分共 {n(S["ecd"]["train_sessions"])} 段对话，用户发言去重后有 {n(S["ecd"]["train_distinct"])} 句。'
         '这些句子是真实买家打出来的，口语、省略、错别字都有，没有标签。'
         '数据发布时已经分好词、去掉了标点，商品链接被替换成数字编号；含 6 位以上数字串的句子（商品编号）我都去掉了。')

doc.para('2.3  FAQ 知识库', indent=False, keep=True)
cats = S['faq']['categories']
doc.para(f'知识库共 {len(FAQ)} 条，分 {len(cats)} 类：' + '、'.join(f'{k} {v} 条' for k, v in cats.items()) + '。'
         '写之前我先统计了 ECD 训练部分出现次数最多的用户问句（“发什么快递”“什么时候发货”“保质期多久”“能优惠点吗”等），'
         '保证真实对话里最常见的问题都有对应的条目，再补上售后、发票、会员等店铺通常会有的规定。'
         '每条 FAQ 包含：一句标准问、一段答案、6 句相似问（同一个意思的其他说法，放进索引并参与训练）、'
         '3 句测试问（只用于测试，写法上故意与相似问拉开距离，带口语、语气词或多余的背景描述）。表 2 是一个例子。')
example = next(e for e in FAQ if e['key'] == 'choose_courier')
tests = [r['query'] for r in read_jsonl(FAQ_DIR / 'test_constructed.jsonl') if r['key'] == 'choose_courier']
doc.table_caption('表 2  一条 FAQ 的内容')
doc.table(['字段', '内容'],
          [['类别', example['category']], ['标准问', example['question']], ['答案', example['answer']],
           ['相似问（6 句）', '；'.join(example['similar'])], ['测试问（3 句）', '；'.join(tests)]],
          [1800, 7226], ['center', 'left'])

doc.para('2.4  真实问句的抽取与标注', indent=False, keep=True)
real = S['real']
doc.para('构造测试问的表达方式有限，不能完整代表真实用户，因此另从 ECD 中抽取两批真实问句。标注由 AI 辅助逐条完成；'
         '目前没有用户已人工逐条复核的记录，标注噪声是本实验的一项局限。')
doc.para(f'评测用的一批：取 ECD 验证、测试部分每段对话的最后一句用户发言（也就是客服接下来要回答的那一句），'
         f'去重后有 {n(S["ecd"]["eval_distinct"])} 句，随机抽 700 句逐条标注，剔除 2 句（含个人信息或粗口）后剩 698 句，'
         f'再随机分成验证集 {real["dev"]["n"]} 句和测试集 {real["test"]["n"]} 句。'
         f'训练用的一批：从 ECD 训练部分的用户发言中取出现次数最多的 200 句，再随机取 500 句，同样逐条标注，得到 {real["train"]["n"]} 句。')
doc.para('标注时每句话单独看、不看上文，判断把哪条 FAQ 的答案回给用户是合适的；没有合适的就标为“范围外”。'
         '范围外的句子主要有三种：离开上文看不出在问什么（“这个呢”“599”）、只针对某个具体商品（“这个是纯棉的吗”）、'
         '不需要回答的陈述（“我已经下单了”）。一句话落在两条相近的 FAQ 之间时，把可接受的 FAQ 都列出，评测时答其中任何一条都算对。'
         f'三批数据中范围内的句子分别是 {real["train"]["in_scope"]}、{real["dev"]["in_scope"]}、{real["test"]["in_scope"]} 句，'
         f'约占一半；带多个可接受标签的有 {real["train"]["multi_label"]}、{real["dev"]["multi_label"]}、{real["test"]["multi_label"]} 句。'
         '完整的标注规则在仓库的 data/faq/ANNOTATION.md 中。评测集的标注在任何模型评测之前完成，之后没有改过。')
doc.figure(FIG / 'data_overview.png', 7.2,
           '图 1  数据概览。上：各来源句长分布；中：真实问句类别构成；下：验证与测试问句最常涉及的12条FAQ')
doc.para('图 1 中间一幅显示，真实问句里物流配送和优惠活动两类最多，售后、发票类很少，约一半在范围外；'
         f'验证集和测试集一共只涉及 {len(set(r["key"] for s in ("dev", "test") for r in read_jsonl(FAQ_DIR / f"real_{s}.jsonl") if r["accept"]))} 条 FAQ。'
         '构造测试集则是每条 FAQ 各 3 句、不含范围外的句子。两个测试集互相补充：构造集覆盖全部 60 条 FAQ，真实集反映真实的问题分布和说法。')

doc.para('2.5  预处理', indent=False, keep=True)
doc.para('所有文本（训练句对、知识库、用户输入）都经过同一个规范化函数：Unicode NFKC 规范化（全角字符转半角）、英文转小写、'
         '去掉标点和符号、去掉中文之间的空白（英文和数字之间保留一个空格）、截断到 40 个字符。'
         '去标点是因为 ECD 发布时已经没有标点，如果知识库和线上输入带标点而真实语料不带，模型会把“有没有标点”当成特征。'
         f'三个句对数据集中约 99% 的句子不超过 40 个字符。预训练模型用它自带的分词器（按字切分，词表 21,128）；'
         f'从零训练的两个模型按字切分，字表取训练语料中出现不少于 3 次的字符，共 {n(S["char_vocab"])} 个（含补齐符和未登录符）。')

doc.para('2.6  划分策略与防止数据泄漏', indent=False, keep=True)
doc.para('各部分数据的用途严格分开，表 3 列出每一份数据在哪个环节使用。')
doc.table_caption('表 3  数据划分与用途')
doc.table(
    ['数据', '句数', '第一阶段训练', '第二阶段训练', '索引', '选模型/阈值', '测试'],
    [['公开句对训练集', n(sum(v['total'] for v in pair['train'].values())) + ' 对', '√', '', '', '', ''],
     ['公开句对验证集 / 测试集', f'{n(sum(v["total"] for v in pair["dev"].values()))} / {n(sum(v["total"] for v in pair["test"].values()))} 对',
      '', '', '', '句对阈值', '句对指标'],
     ['FAQ 标准问、相似问', str(S['faq']['index_texts']), '', '√', '√', '', ''],
     ['真实训练集', str(real['train']['n']), '', '√', '√（范围外的作拒识样例）', '', ''],
     ['真实验证集', str(real['dev']['n']), '', '', '', '√', ''],
     ['构造测试集', str(S['constructed']['n']), '', '', '', '', '√'],
     ['真实测试集', str(real['test']['n']), '', '', '', '', '√']],
    [2200, 1326, 1000, 1000, 1500, 1000, 1000], ['left'] + ['center'] * 6)
checks = S['checks']
doc.para('为了保证测试结果可信，做了三件事，并写成脚本（src/data_stats.py）自动核验：')
doc.para(f'（1）评测用的 700 句真实问句在抽出之后，先从 ECD 的无标签语料里删掉，再从剩下的语料中抽训练样本。'
         f'核验结果：真实训练集与验证集、测试集的重复句数为 {checks["real_train_dev_overlap"]} 和 {checks["real_train_test_overlap"]}，'
         f'评测问句出现在无标签语料中的句数为 {checks["eval_queries_in_ecd_unlabeled"]}。')
doc.para(f'（2）解析 FAQ 源文件时检查标准问、相似问、测试问规范化后两两不重复；'
         f'评测问句（构造集和真实集）出现在索引文本中的句数为 {checks["eval_queries_in_index"]}。')
removed = read_json(ROOT / 'data' / 'processed' / 'pairs_stats.json')
total_removed = sum(v['removed_as_held_out'] for v in removed.values())
doc.para(f'（3）公开句对数据集本身有重复：去掉标点后，训练集中有 {n(total_removed)} 对句子与验证集或测试集中的句对相同'
         f'（其中 LCQMC {n(removed["LCQMC/train"]["removed_as_held_out"])} 对；不做规范化时，训练集与测试集一字不差的重复句对也有 88 对）。'
         f'这些句对已从训练集中去掉，去掉后训练集与验证集、测试集的重复句对数为 {checks["pairs_train_dev_overlap"]} 和 {checks["pairs_train_test_overlap"]}。')

# ================================================================ 3
doc.h1('3  模型与方法')
doc.para('3.1  总体流程', indent=False, keep=True)
doc.figure(FIG / 'system.png', 7.2, '图 2  系统流程：（a）在线问答；（b）离线建立索引；（c）编码器的两阶段训练')
doc.para('图 2 是整个系统。在线部分（a）：用户问句规范化后送入编码器得到一个向量，与索引中每一句的向量算余弦相似度，'
         '每条 FAQ 取它名下所有句子（标准问、相似问、标注过的真实问句）的最高分作为该 FAQ 的得分，得分最高的 FAQ 就是检索结果；'
         '再决定答不答。离线部分（b）：把知识库里的句子用同一个编码器编码好存起来，知识库增删条目时只需要重新编码，不用重新训练。'
         '训练部分（c）分两个阶段，3.3 和 3.4 节分别说明。')
doc.para('之所以用“两句话分别编码再比较”的双塔结构，而不是把问句和每条 FAQ 拼在一起送进模型打分，是因为双塔的索引向量可以提前算好，'
         '线上每个问题只需要编码一次；拼接打分则要对每条候选各跑一次模型，知识库越大越慢。')

doc.para('3.2  句向量编码器', indent=False, keep=True)
doc.figure(FIG / 'model.png', 7.2, '图 3  模型结构：（a）双塔，两个塔共享全部参数；（b）一个 Transformer 编码层')
doc.para('图 3（a）是双塔结构：两句话经过同一个编码器，各得到一个向量，训练时用它们的余弦相似度计算损失。'
         '编码器内部是若干个 Transformer 编码层（图 3（b））：每个字先变成向量并加上位置信息，'
         '自注意力让每个字参考句中所有其他字来更新自己的表示，前馈网络再做一次逐位置的非线性变换，两处都带残差连接和层归一化。'
         '最后把所有有效位置的输出取平均，得到整句话的向量。为了比较，我一共实现了四种编码器，见表 4。')
doc.table_caption('表 4  四种编码器')
doc.table(
    ['编码器', '结构', '初始化', '句向量维度'],
    [['BiLSTM', '字嵌入 256 维；2 层双向 LSTM，隐藏 256 维；平均池化与最大池化拼接后线性映射', '随机初始化，从零训练', '256'],
     ['Transformer', '自己实现的编码器（src/models.py）：4 层，256 维，4 个头，前馈 1024 维，Pre-LN，可学习的位置嵌入；平均池化',
      '随机初始化，从零训练', '256'],
     ['RBT3', '3 层 Transformer，768 维，12 个头；平均池化', '哈工大讯飞联合实验室的预训练模型 hfl/rbt3', '768'],
     ['RoBERTa-wwm-ext', '12 层 Transformer，768 维，12 个头；平均池化', '预训练模型 hfl/chinese-roberta-wwm-ext', '768']],
    [1700, 4126, 2200, 1000], ['center', 'left', 'left', 'center'])
doc.para('前两个模型从零训练，用来回答“不用预训练模型能做到什么程度”；Transformer 编码器的多头自注意力、残差、层归一化、位置嵌入都是自己写的，'
         '没有调用 nn.TransformerEncoder。后两个使用 HFL 发布的中文预训练权重：RoBERTa-wwm-ext 有 12 层，RBT3 有 3 层。'
         '参数量与实际时延在结果部分分别统计。预训练模型通过 transformers 库载入，去掉预训练输出头。')

doc.para('3.3  第一阶段：通用句子匹配', indent=False, keep=True)
doc.para('第一阶段在公开句对上训练，目标是让同义句的向量相近、不同义句的向量远离。用了两个损失函数，训练时相加。')
doc.para('CoSENT 损失直接利用 0/1 标签：一个批次内，任何一个正例对的相似度都应该高于任何一个负例对。'
         '记第 i 个句对两句话的余弦相似度为 s，损失为')
def upright(text):
    return omr(text).replace('<m:r>', '<m:r><m:rPr><m:sty m:val="p"/></m:rPr>', 1)

def summation(lower, expression):
    return ('<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="subSup"/><m:supHide m:val="1"/>'
            '</m:naryPr><m:sub>' + omr(lower) + '</m:sub><m:sup/><m:e>' + expression + '</m:e></m:nary>')

doc.math(omsub('L', 'CoSENT') + omr(' = ') + upright('log') + omr('(') + omr('1 + ') +
         summation('i∈N, j∈P', upright('exp') + omr('[λ(') + omsub('s', 'i') + omr(' − ') + omsub('s', 'j') + omr(')]')) + omr(')'))
doc.para('其中 N、P 分别是批内负例对、正例对的下标集合，λ=20。')
doc.para('它只约束相似度的相对大小，不要求正例的相似度一定接近 1、负例接近 0，比直接回归标签更容易优化。')
doc.para('InfoNCE 损失（批内负例的对比损失）只用正例对：对正例对 (a, b)，把同一批次里其他句对的第二句都当作 a 的负例，'
         '要求 a 与 b 的相似度在所有候选中最高；再反过来对 b 做一遍。')
def exp_similarity(other):
    return upright('exp') + omr('[') + omfrac(upright('cos') + omr('(') + omsub('a', 'i') + omr(', ') + omsub('b', other) + omr(')'), omr('τ')) + omr(']')
doc.math(omsub('ℓ', 'i') + omr(' = − ') + upright('log') + omfrac(exp_similarity('i'), summation('k', exp_similarity('k'))))
doc.para('对正例锚点的损失取平均，再与反向（b 到 a）的损失取平均，得到 InfoNCE；温度 τ=0.05。')
doc.para('批内其他句子大多和 a 毫无关系，所以它教给模型的是“在一大堆句子里把同义的那句挑出来”，这正是检索要做的事。'
         '实现上有两个细节：标签为 0 的句对不作锚点，但它们的句子仍然作为候选；同一批内如果有和正例文本完全相同的句子，不算作负例。')

doc.para('3.4  第二阶段：领域适配', indent=False, keep=True)
doc.para('公开句对数据与店铺知识库的分布不同，第一阶段没有直接学习该店铺的范围内与范围外标签。'
         '第二阶段用店铺自己的数据继续训练：把每条 FAQ 看成一个类别，它的标准问、相似问以及标注为该 FAQ 的真实问句都属于这个类别。'
         f'每一步从 {len(FAQ)} 个类别里各随机取两句不同的话组成正例对，其他类别的句子是负例，损失仍是 InfoNCE。'
         '在此之上加了两点：')
doc.para('（1）范围外的真实问句作负例。每一步另外随机取 60 句标注为范围外的真实问句，加入所有锚点的候选集合；'
         '它们永远不是任何 FAQ 的正例，于是被推离所有 FAQ。同时加一项间隔损失，直接要求范围外句子与任何 FAQ 句子的余弦相似度不超过 0.3：')
doc.math(omsub('L', 'oos') + omr(' = mean( max( 0, cos(') + omsub('x', 'FAQ') + omr(', ') + omsub('o', 'k') + omr(') − 0.3 ) )'))
doc.para('（2）可接受的其他 FAQ 不作负例。如果一句真实问句标注了多个可接受的 FAQ，那么这些 FAQ 的句子不出现在它的负例里，避免把本来算对的答案推远。')
doc.para('第二阶段的数据只有几百句，很容易过拟合，所以用很小的学习率训练 200 步，每 25 步在真实验证集上评一次，保留验证分数最高的一次；'
         '第 0 步（不做适配）也参与比较。')

doc.para('3.5  检索与拒答', indent=False, keep=True)
doc.para('索引里放FAQ 的标准问、相似问和真实训练集中范围内的问句（挂在它的主标签 FAQ 名下）。'
         '一条 FAQ 的得分是它名下所有句子与用户问句相似度的最大值，所以真实用户的说法只要有一句在索引里，相近的问法就能被找到。')
doc.para('拒答有两道。第一道是阈值：最高分低于阈值就不作答，阈值在真实验证集上取使端到端准确率最高的值。'
         '第二道是拒识样例：把真实训练集中范围外的问句也编码好放在一边，如果用户问句与某个拒识样例的相似度比与任何 FAQ 句子的都高，'
         '说明它更像一句“已知答不了的话”，直接拒答。拒识样例相当于给“范围外”这一类也提供了例子，而不是只靠一个相似度阈值来划界。')

# ================================================================ 4
doc.h1('4  实验设置')
doc.para('4.1  评价指标', indent=False, keep=True)
doc.para('句对判定（公开句对的测试集）：用两句话的余弦相似度打分，报告 AUC，以及用验证集上准确率最高的阈值得到的测试集准确率。'
         '这组指标只用来说明第一阶段学到了什么。')
doc.para('FAQ 检索（只看范围内的问句）：Top-1 准确率，即得分最高的 FAQ 是可接受的 FAQ 的比例；Top-3；MRR@10，即正确 FAQ 排名倒数的平均值（排在 10 名以外记 0）。')
doc.para('端到端准确率（包含范围外的问句，是最接近实际使用效果的指标）：范围内的问句要回答且答对才算对，范围外的问句要拒答才算对，'
         '两者合起来除以总句数。同时报告回答的正确率（回答了的问句中答对的比例）和 AUROC（用最高相似度区分范围内、范围外问句的能力，0.5 相当于随机）。')
doc.para('通用问句回答率：从 LCQMC 测试集固定抽取500句，统计系统给出FAQ答案的比例。代码中沿用open_domain字段名。'
         '这批问句未逐条人工核实均与购物无关，因此该值只是开放域误答的代理指标，不能直接视为严格标注的范围外错误率。')

doc.para('4.2  软硬件环境', indent=False, keep=True)
doc.para('Windows 11，Python 3.12.14，PyTorch 2.10.0（CUDA 13.0），transformers 5.23.0，scikit-learn 1.9.1；'
         'NVIDIA GeForce RTX 5060 Laptop GPU（8 GB 显存）。早期实验曾与其他程序共用显卡；本次补充实验顺序执行。'
         '第一阶段日志中的显存比例上限为0.32，第二阶段队列采用0.55。RoBERTa训练使用梯度检查点，'
         '通过在反向传播时重算部分前向结果节省显存。耗时只作为实际运行记录，不据此评价各模型的速度优劣。')

doc.para('4.3  超参数', indent=False, keep=True)
doc.para('两个阶段的超参数见表 5。第一阶段学习率先在前5%的步数内线性升到设定值，再按余弦曲线降到0；第二阶段使用固定学习率。'
         '两个阶段均用混合精度训练，梯度范数裁剪到 1。')
doc.table_caption('表 5  超参数')
doc.table(
    ['', 'BiLSTM', 'Transformer', 'RBT3', 'RoBERTa'],
    [['第一阶段 学习率', '1e-3', '5e-4', '5e-5', '3e-5'],
     ['第一阶段 轮数（步数）', '10 轮（56,210）', '10 轮（56,210）', '2 轮（11,242）', '1 轮（5,621）'],
     ['第一阶段 验证间隔', '每轮', '每轮', '每半轮', '每 1/4 轮'],
     ['第二阶段 学习率', '1e-4', '1e-4', '2e-5', '1e-5'],],
    [2026, 1750, 1750, 1750, 1750], ['left', 'center', 'center', 'center', 'center'])
doc.para('其余设置各模型相同：优化器 AdamW，权重衰减 0.01（偏置和归一化层的参数不衰减）；第一阶段批大小为 64 对句子，'
         'CoSENT 的 λ=20，InfoNCE 的温度 τ=0.05；第二阶段训练 200 步，每步从 60 个类别中各取 2 句，另取 60 句范围外问句，'
         '每 25 步验证一次；句子最长 40 个字符。')
doc.para('模型选择只用真实验证集：第一阶段每次验证计算“MRR 与端到端准确率的平均”（此时索引只有标准问和相似问），'
         '保留最高的一次；第二阶段同样每 25 步验证一次，保留最高的一次，第 0 步（不做适配）也参与比较。'
         '回答阈值在真实验证集上选取。构造测试集和真实测试集用于训练完成后的统一评测，不参与选模型或调阈值。'
         'RoBERTa 和 RBT3 用随机种子 1、2、3 各训练一遍（两个阶段都换种子），表中给出均值±标准差；其余设置只训练一遍（种子 1）。')

doc.para('4.4  对比实验', indent=False, keep=True)
doc.para('实验分四组。（1）编码器对比：不训练的 TF-IDF、BM25 和未微调的 RoBERTa，以及四种编码器只做第一阶段、做完整方案的结果。'
         '（2）第一阶段的损失函数和池化方式：在 RBT3 上分别只用 CoSENT、只用 InfoNCE、两者相加，以及把平均池化换成取 [CLS] 位置的输出。'
         '（3）领域适配的逐项对比：在 RoBERTa 上从“只做第一阶段”出发，每次加入一项（真实问句进索引、拒识样例、各种第二阶段训练方式），'
         '看每一项各带来多少变化。（4）半监督的伪标注：用完整方案的模型给淘宝语料中的无标签问句打标签，'
         '取置信度高的加入训练或索引，看能否进一步提高。')

from sections_results import complete_report
complete_report(doc, ROOT, S)
doc.save(OUT)
print('saved', OUT)
