#import "@preview/ezexam:0.3.1": *

#show: setup.with(
    mode: EXAM,
    resume: false,
    heading-top: 0em,
    heading-bottom: .4em,
    line-height: .65em,
    par-spacing: .65em,
    enum-spacing: .65em,
    list-spacing: .65em,
)
#show strong: set text(weight: "bold")
#set par(justify: true)
#let Title = "山东大学计算机科学与技术学院机器学习与模式识别期末试题"
#let author = "arshtyi"
#let date = datetime.today()
#set document(title: Title, date: date, author: author)
#title(Title)
#exam-info(info: (
    班级: "24智能",
    教师: "邹逸飞",
    时间: datetime(year: 2026, month: 7, day: 4).display("[year].[month].[day]"),
    源码: link("https://github.com/arshtyi/SDU-Machine-Learning-And-Pattern-Recognition", "source"),
))
#set par(justify: true)
#show raw: set text(font: ("JetBrains Mono",))
#show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: .3em, y: 0em),
    outset: (x: 0em, y: .3em),
    radius: .2em,
)
#show math.equation: set text(font: "New Computer Modern Math")

= 简答
#question[
    #set enum(spacing: 1em)
    对于样本点$x=(0,1,2),y=(1,3,2)$，采用一元线性回归模型$hat(y)=w_1x+w_0$。
    + 利用公式$w_1=(sum_i (x_i - macron(x))(y_i - macron(y))) / (sum_i (x_i - macron(x))^2)$计算$w_1$和$w_0$。
    + 给出样本的预测值$hat(y)_i$与残差平方和。
    + 给出新样本$x=3$的预测值$hat(y)$。
    + 对于异常值$(3,10)$，该直线会发生怎样的变化？为什么平方损失对异常值敏感？
]
#question[
    对于已中心化的二维数据$(1,0),(-1,0),(0,2),(0,-2)$，其协方差矩阵为$S=1/n X^T X$。
    + 计算协方差矩阵$S$。
    + 计算两个特征值，并给出第一主成分方向。
    + 给出仅保留第一主成分时的解释方差比例。
    + 将$z=(1,1)$投影到第一主成分上并重构，给出重构误差平方。
    + PCA仅保留最大主成分是否一定有利于分类？为什么？
]
#question[
    两类$C_1={(2,1),(4,1)},C_2={(0,0),(0,2)}$。
    + 给出两类均值$mu_1,mu_2$。
    + 计算类内散度矩阵$S_w$，并求LDA投影方向$w=S_w^(-1)(mu_1-mu_2)$。
    + 将两类均值分别投影到LDA方向上，并以投影值的中点作为分类阈值，判断新数据$x=(1,1)$的类别。
    + LDA与PCA的优化目标有何本质区别？
]
#question[
    某文本分类场景下，二值特征$X$表示是否出现某个关键词，标签$Y$表示文本类别。统计如下：$X=1,Y=1:3;X=0,Y=1:1;X=1,Y=0:1;X=0,Y=0:5$，总样本数为$10$。
    + 计算$P(X),P(Y),P(X,Y)$。
    + 使用以$2$为底的对数计算互信息$I(X;Y)$。
    + 假设另一个特征的互信息接近于$0$，过滤式特征选择倾向于保留哪一个？
    + 给出过滤式特征选择的优点与局限。
]
#question[
    两类别的先验概率：$P(C_1)=0.6, P(C_2)=0.4$，类条件分布：$x bar C_1 tilde.op N(0,1), x bar C_2 tilde.op N(2,1)$。
    + 对于$x=0.8$，计算两类的未归一化后验分数$P(C_k)p(x|C_k)$，可忽略共同常数$1/sqrt(2pi)$，并给出类别判定。
    + 求两类后验概率相等的边界$x_0$。
    + 假设先验概率相等，则边界在哪里？为什么较大的先验概率会扩大该类的判定区域？
]
