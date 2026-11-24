"""Monochrome, print-sized diagrams of the implemented FAQ system."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import RESULTS
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
FIG = RESULTS / 'figures'
def canvas(height):
    fig, ax = plt.subplots(figsize=(6.4, height))
    ax.set_xlim(0, 6.4); ax.set_ylim(0, height); ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99)
    return fig,ax
def box(ax,x,y,w,h,text,size=10):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor='black',linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.35)
def arrow(ax,x1,y1,x2,y2,text=''):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='black',lw=.8))
    if text: ax.text((x1+x2)/2+.08,(y1+y2)/2,text,ha='left',va='center',fontsize=9)
def system_figure():
    fig,ax=canvas(8.4)
    ax.text(.15,8.15,'（a）在线问答',fontsize=11)
    box(ax,1.65,7.48,3.1,.45,'用户输入单轮购物问句')
    box(ax,1.65,6.66,3.1,.55,'文本规范化 → 句向量编码')
    box(ax,1.,5.77,4.4,.60,'与索引计算余弦相似度\n每条FAQ取其名下语句的最高分')
    box(ax,.9,4.85,4.6,.60,'最高分达到验证集阈值\n且最相似语句不是拒识样例？')
    for y1,y2 in [(7.48,7.21),(6.66,6.37),(5.77,5.45)]: arrow(ax,3.2,y1,3.2,y2)
    box(ax,.3,3.96,2.55,.55,'返回FAQ标准答案')
    box(ax,3.5,3.96,2.55,.55,'拒答并建议转人工\n展示猜你想问')
    arrow(ax,2,4.85,1.55,4.51,'是'); arrow(ax,4.4,4.85,4.78,4.51,'否')
    ax.text(.15,3.57,'（b）离线建立索引',fontsize=11)
    box(ax,.25,2.64,2.75,.65,'标准问与相似问\n真实训练问句及拒识样例')
    box(ax,3.5,2.64,2.65,.65,'同一个编码器\n归一化后保存索引矩阵')
    arrow(ax,3.,2.97,3.5,2.97)
    ax.text(.15,2.20,'（c）两阶段训练与评测',fontsize=11)
    box(ax,.25,1.13,2.75,.80,'阶段一 359,787个公开句对\nLCQMC、BQ、AFQMC\nCoSENT + InfoNCE')
    box(ax,3.5,1.13,2.65,.80,'阶段二 店铺领域适配\nFAQ与698句真实训练问句\n对比学习及范围外负例')
    arrow(ax,3.,1.53,3.5,1.53)
    ax.text(3.2,.65,'验证集300句：选模型和阈值\n最终评测：构造集180句、真实测试集398句',ha='center',va='center',fontsize=10,linespacing=1.45)
    fig.savefig(FIG/'system.png',dpi=220);plt.close(fig)
def model_figure():
    fig,ax=canvas(8.6)
    ax.text(.15,8.35,'（a）共享参数的双塔句向量模型',fontsize=11)
    for x,text in [(.25,'句子A：发什么快递'),(3.5,'句子B：用哪家快递发货')]:
        box(ax,x,7.57,2.65,.45,text)
        box(ax,x,6.43,2.65,.78,'分词与嵌入\n共享同一个句子编码器')
        box(ax,x,5.35,2.65,.70,'池化与归一化\n得到句向量')
        arrow(ax,x+1.325,7.57,x+1.325,7.21);arrow(ax,x+1.325,6.43,x+1.325,6.05)
    ax.annotate('',xy=(3.5,6.82),xytext=(2.9,6.82),arrowprops=dict(arrowstyle='<->',lw=.8))
    box(ax,1.15,4.35,4.1,.63,'余弦相似度与训练损失\nCoSENT / InfoNCE')
    arrow(ax,1.575,5.35,2.2,4.98);arrow(ax,4.825,5.35,4.2,4.98)
    ax.text(.15,3.9,'（b）自写Transformer的一个Pre-LN编码层',fontsize=11)
    box(ax,.8,3.02,4.8,.55,'输入 X → LayerNorm → 多头自注意力')
    box(ax,.8,2.08,4.8,.55,'与 X 残差相加，得到 Y')
    box(ax,.8,1.14,4.8,.55,'Y → LayerNorm → Linear → GELU → Linear')
    box(ax,.8,.20,4.8,.55,'与 Y 残差相加，得到该层输出')
    for y1,y2 in [(3.02,2.63),(2.08,1.69),(1.14,.75)]:arrow(ax,3.2,y1,3.2,y2)
    fig.savefig(FIG/'model.png',dpi=220);plt.close(fig)
if __name__=='__main__':
    FIG.mkdir(parents=True,exist_ok=True);system_figure();model_figure()
    print('Saved print-sized system and model diagrams.')
