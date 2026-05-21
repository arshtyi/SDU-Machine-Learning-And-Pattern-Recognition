#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let fonts = (main: "Source Han Serif SC", mono: "IBM Plex Mono", cjk: "Noto Serif CJK SC")
#let institute = "计算机科学与技术"
#let course = "机器学习"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "Experiment 9: Decision Tree"
#let time = "2"

#set document(title: title, author: author, date: date)
#set text(font: (fonts.main, fonts.cjk), size: zh(5), lang: "zh", region: "cn")
#set par(justify: true, first-line-indent: (amount: 2em, all: true))
#set page(
    paper: "a4",
    margin: (x: 35pt, y: 35pt),
    footer: align(center, context counter(page).display("- 1 -")),
)
#set heading(numbering: numbly("", "{2:1}.", "({3:1})"))
#show heading: set text(size: zh(-4))
#{
    set underline(offset: 2.5pt, extent: 2.5pt)
    show heading: it => align(center, text(tracking: .1em, size: zh(-2), it))
    heading(numbering: none, level: 1)[山东大学 #underline[#institute] 学院\ #underline[#course] 课程实验报告]
    set text(size: zh(-4))
    set table.cell(inset: .5em, align: left + horizon, stroke: 1pt)
    table(
        columns: (3fr, 2.5fr, 3fr),
        [学号：#id], [姓名：#author], [班级：#class],
    )
    v(0em, weak: true)
    table(
        columns: 1fr,
        [实验题目：#title],
    )
    v(0em, weak: true)
    table(
        columns: (1fr,) * 2,
        [实验学时：#time], [实验日期：#date.display("[year].[month].[day]")],
    )
}
#show raw: set text(font: (fonts.mono, fonts.cjk))
#show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: 0.3em, y: 0em),
    outset: (x: 0em, y: 0.3em),
    radius: 0.2em,
)
#show: codly-init
#codly(
    languages: codly-languages,
    zebra-fill: none,
    fill: luma(248),
    stroke: 0.5pt + rgb("bfbfbf"),
    radius: 4pt,
)
#set enum(numbering: numbly("{1:1})", "{2:a}."))
#set list(indent: 6pt, marker: sym.bullet.tri)

#let in-block(body) = {
    let is-level-1-heading(it) = (
        it.func() == heading
            and (
                it.at("level", default: none) == 1
                    or (it.at("offset", default: none) + it.at("depth", default: none) == 1)
            )
    )

    let text-block(it) = {
        v(0em, weak: true)
        block(
            width: 100%,
            inset: (x: 4pt, y: 1em),
            stroke: 1pt,
            breakable: true,
            it,
        )
    }

    let children = body.at("children", default: (body,))
    let content = ()
    let buf = ()

    for child in children {
        if is-level-1-heading(child) {
            if buf.len() > 0 {
                content.push(text-block(buf.join()))
                buf = ()
            }
            buf.push(child)
        } else if buf.len() > 0 {
            buf.push(child)
        } else {
            content.push(child)
        }
    }
    if buf.len() > 0 {
        content.push(text-block(buf.join()))
    }
    content.join()
}
#show: in-block

= 实验目的
- 理解决策树分类器的基本思想, 掌握基于特征划分和信息增益递归建树的方法.
- 在不使用现成机器学习库和决策树库的前提下, 从零实现二分类决策树.
- 使用 10 折交叉验证评估模型泛化能力, 并对决策树结构进行可视化展示.

= 硬件环境
- CPU: 9600x

= 软件环境
- Python: 3.11

= 实验步骤与内容
== 数据处理
读取数据,将前$11$列作为特征矩阵`X`,最后一列作为标签向量`Y`:
```python
def load_wine_data(path: Path = RES_DIR / "ex6Data.csv") -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    data = np.asarray(rows, dtype=float)
    if header[-1] != "quality" or data.shape[1] != 12:
        raise ValueError("expected 11 feature columns and one quality label column")
    return data[:, :-1], data[:, -1].astype(int)
```
结果如下:
```txt
samples=4898, features=11
class 0=3838, class 1=1060
```
#figure(image("../output/class_distribution.png"), caption: [葡萄酒质量二分类标签分布]) <fig:dist>
== 决策树
实现CART二叉决策树: 每个内部节点选择一个连续特征阈值将样本划分为两部分.候选阈值取相邻不同取值的中点, 划分质量使用 Gini 指数或信息熵, 最终选择带来最大 impurity decrease 的划分:
```python
def _best_split(self, x: np.ndarray, y: np.ndarray, parent_impurity: float):
    best_feature = -1
    best_threshold = 0.0
    best_gain = 0.0
    n_samples, n_features = x.shape
    total_pos = y.sum()

    for feature_index in range(n_features):
        order = np.argsort(x[:, feature_index], kind="mergesort")
        values = x[order, feature_index]
        labels = y[order]
        distinct_positions = np.nonzero(values[:-1] != values[1:])[0] + 1
        cumsum_pos = np.cumsum(labels)
```
并且加入以下停止条件:
+ 节点样本已经纯净.
+ 当前深度达到 `max_depth`.
+ 节点样本数小于 `min_samples_split`.
+ 划分后任一叶节点样本数小于 `min_samples_leaf`.
+ impurity decrease 小于 `min_impurity_decrease`.
== 交叉验证与结果
每次取 1 折作为测试集, 其余 9 折作为训练集, 重复 10 次后取平均准确率.为了比较不同树复杂度和划分准则的影响, 对 Gini 与 Entropy 两种准则、不同最大深度、叶节点最小样本数和内部节点最小划分样本数进行了测试.其中 `min_samples_leaf` 越大, 树越保守; `min_samples_split` 越小, 树越容易继续细分:
```txt
gini-depth3-leaf6-split12      mean=0.7944, std=0.0078
gini-depth5-leaf6-split12      mean=0.8056, std=0.0131
gini-depth9-leaf6-split12      mean=0.8107, std=0.0206
entropy-depth5-leaf6-split12   mean=0.8030, std=0.0143
entropy-depth9-leaf6-split12   mean=0.8085, std=0.0128
entropy-full-leaf6-split12     mean=0.8150, std=0.0191
entropy-full-leaf4-split12     mean=0.8254, std=0.0145
entropy-full-leaf3-split6      mean=0.8293, std=0.0124
entropy-full-leaf2-split2      mean=0.8301, std=0.0119
entropy-full-leaf1-split4      mean=0.8395, std=0.0147
entropy-full-leaf1-split2      mean=0.8418, std=0.0159
```
#figure(image("../output/cv_accuracy.png"), caption: [不同决策树配置的 10 折交叉验证准确率]) <fig:cv>
最佳配置为 `criterion=entropy`, `max_depth=None`, `min_samples_leaf=1`, `min_samples_split=2`, 10 折平均准确率为 0.8418:
```txt
[[3838    0]
 [   0 1060]]
training precision=1.0000, recall=1.0000, f1=1.0000
```
上面的混淆矩阵是在全量训练集上重新训练最佳模型后得到的, 用于展示模型的拟合能力; 判断泛化性能时仍以 10 折交叉验证准确率为主.由于该配置允许叶节点只包含 1 个样本并且内部节点只要有 2 个样本即可继续划分, 它的训练集表现达到 100%, 同时 10 折交叉验证也高于更保守的剪枝配置, 因而在本实验数据和评估方式下效果最好.
== 决策树可视化
展示前$3$层: 根节点首先使用 `alcohol <= 10.625` 进行划分, 说明酒精度是该数据集中最重要的早期划分特征之一.随后模型结合 `volatile acidity`, `residual sugar`, `sulphates`, `pH` 等特征继续划分样本:
#figure(image("../output/tree_top_levels.png"), caption: [决策树前 3 层结构可视化]) <fig:tree>
前几层文本结构:

```txt
alcohol <= 10.625 (gain=0.0940, samples=4898, pos=1060, p1=0.216, predict=0)
  T: volatile acidity <= 0.2025 (gain=0.0415, samples=2853, pos=260, p1=0.091, predict=0)
    T: residual sugar <= 12.55 (gain=0.0630, samples=652, pos=150, p1=0.230, predict=0)
      T: sulphates <= 0.835 (gain=0.0316, samples=510, pos=82, p1=0.161, predict=0)
        T: total sulfur dioxide <= 144.5 (gain=0.0234, samples=504, pos=76, p1=0.151, predict=0)
          T: Leaf(samples=327, pos=64, p1=0.196, predict=0)
          F: Leaf(samples=177, pos=12, p1=0.068, predict=0)
        F: Leaf(samples=6, pos=6, p1=1.000, predict=1)
      F: alcohol <= 9.15 (gain=0.2424, samples=142, pos=68, p1=0.479, predict=0)
        T: citric acid <= 0.305 (gain=0.4163, samples=80, pos=58, p1=0.725, predict=1)
    F: volatile acidity <= 0.3025 (gain=0.0131, samples=2201, pos=110, p1=0.050, predict=0)
```
= 结论分析与体会
本实验完成了从 CSV 数据读取、标签分布展示、决策树分类器从零实现、10 折交叉验证到树结构可视化的完整流程.实验结果表明, 决策树可以在不做复杂特征变换的情况下较好地处理葡萄酒质量二分类任务.Entropy 和 Gini 均能达到 0.80 左右的平均准确率, 其中不限制最大深度、设置 `min_samples_leaf=1` 与 `min_samples_split=2` 的 Entropy 决策树表现最好, 平均准确率达到 0.8418.

从树结构看, 模型优先选择 `alcohol`, `volatile acidity`, `residual sugar` 等特征进行划分, 这些特征对高质量葡萄酒识别有较强区分作用.调参结果说明, 更深的树能捕捉较细的局部规律, 因而交叉验证准确率继续提升; 但该配置训练集完全拟合, 也具有更高过拟合风险.如果更强调模型简洁性和稳定性, `entropy-full-leaf3-split6` 仍能达到 0.8293 的准确率且标准差更低, 可以作为更保守的折中方案.
