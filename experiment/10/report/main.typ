#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let fonts = (main: "Source Han Serif SC", mono: "IBM Plex Mono", cjk: "Noto Serif CJK SC")
#let institute = "计算机科学与技术"
#let course = "机器学习"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "Experiment 10: Random Forest"
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
    table(columns: 1fr, [实验题目：#title])
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

= 实验目的
- 理解决策树分类的基本思想, 观察树深度增加时决策边界如何逐步细分.
- 理解决策树容易过拟合的原因, 并通过不同随机子集训练的树观察模型不稳定性.
- 掌握 Bagging 集成方法和随机森林的基本用法, 对比单棵树、Bagging 与随机森林的分类边界.

= 硬件环境
- CPU: 9600x

= 软件环境
- Python: 3.11

= 实验步骤与内容
== 环境与数据
使用手册中的二维四分类样例数据. 数据由 `sklearn.datasets.make_blobs` 生成, 样本数为 $300$, 类别数为 $4$, `random_state=0`, `cluster_std=1.0`. 统一设置输出目录为 `output/`:

```python
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RANDOM_STATE = 0
def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(
        n_samples=300,
        centers=4,
        random_state=RANDOM_STATE,
        cluster_std=1.0,
    )
```

#figure(image("../output/animal_decision_tree.png"), caption: [手工决策树示意图]) <fig:animal>
#figure(image("../output/blob_dataset.png"), caption: [二维四分类数据集]) <fig:blob>

== 决策树分类器
决策树通过轴对齐的二分切分将特征空间划分为多个区域. 为了观察树的生长过程, 分别设置 `max_depth=1,2,3,4`, 绘制前四层决策边界:

```python
for depth, ax in zip([1, 2, 3, 4], axes.ravel(), strict=True):
    model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    visualize_classifier(model, x, y, ax, title=f"Decision tree depth = {depth}")
```

#figure(image("../output/decision_tree_depths.png"), caption: [决策树前四层分类边界]) <fig:depths>

随后训练不限制深度的单棵决策树. 从边界可以看到, 单棵树会形成较多细碎、弯折的局部区域, 这些区域往往反映了训练样本的局部噪声, 而不一定代表真实数据分布.

#figure(image("../output/decision_tree_full.png"), caption: [不限制深度的单棵决策树分类边界]) <fig:single-tree>

== 过拟合与随机子集
为了展示决策树对训练样本扰动的敏感性, 将数据随机划分为两半, 分别训练两棵树. 两棵树在类别中心附近通常较一致, 但在簇之间的不确定区域容易给出不同划分:

```python
rng = np.random.default_rng(RANDOM_STATE + 1)
halves = np.array_split(rng.permutation(len(y)), 2)
for part_number, indices in enumerate(halves, start=1):
    model = DecisionTreeClassifier(random_state=RANDOM_STATE + part_number)
    model.fit(x[indices], y[indices])
```

#figure(image("../output/random_subset_trees.png"), caption: [两个随机半样本训练出的决策树]) <fig:subset-trees>

如果把多个树的输出合并, 单个模型的偶然性会被削弱.先使用两棵子采样树做一个小型集成, 用于说明"多个弱稳定模型投票"这一思想:

#figure(image("../output/two_tree_ensemble.png"), caption: [两棵树子采样集成的分类边界]) <fig:two-tree>

== Bagging 集成
Bagging 通过反复从训练集抽样, 训练多棵彼此不同的决策树, 再对预测结果进行投票. 使用 `BaggingClassifier`, 基学习器为 `DecisionTreeClassifier`, 树数量为 $100$, 每棵树使用 $80%$ 样本:

```python
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=1),
    n_estimators=100,
    max_samples=0.8,
    random_state=1,
)
bag.fit(x, y)
```

#figure(image("../output/bagging_classifier.png"), caption: [Bagging 决策树集成分类边界]) <fig:bagging>

Bagging 的边界相比单棵树更平滑, 说明多个过拟合模型经过投票后可以降低单个模型方差, 从而缓解决策树对训练样本的敏感性.

== 随机森林
随机森林可以看作更强的随机化树集成. 它不仅对样本进行随机化, 还会在树节点划分时引入特征级随机性. 使用 ```python RandomForestClassifier(n_estimators=100, random_state=0)```:

```python
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
visualize_classifier(model, x, y, ax, title="Random Forest of 100 Trees")
```

#figure(image("../output/random_forest_classifier.png"), caption: [随机森林分类边界]) <fig:forest>

与单棵树相比, 随机森林对局部噪声不那么敏感; 与普通 Bagging 相比, 特征级随机性进一步降低了树之间的相关性, 使集成结果更稳定.

== 模型评价与输出
额外使用分层训练/测试划分比较三类模型的训练准确率、测试准确率、树深度和叶节点数量. 结果保存到 `output/model_accuracy.csv`, 说明保存到 `output/summary.md`:

```python
models = [
    ("single decision tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("bagging classifier", make_bagging_classifier(n_estimators=100, max_samples=0.8, random_state=1)),
    ("random forest", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
]
```

评价文件字段如下:

```txt
model,train_accuracy,test_accuracy,depth_summary,leaf_summary
```

= 结论分析与体会
本实验完成了随机森林实验手册中的完整流程: 首先构造二维四分类数据并绘制数据分布, 然后训练不同深度的决策树观察边界逐层细分, 再通过随机半样本训练的两棵树展示决策树过拟合和不稳定性, 最后使用 Bagging 与随机森林进行集成分类.

实验现象说明, 单棵不剪枝决策树可以非常细致地拟合训练数据, 但也容易形成不规则的局部边界. Bagging 通过样本扰动和投票平均降低方差, 随机森林进一步在特征选择层面引入随机性, 因而通常能得到更平滑、更稳定的分类边界. 这也体现了集成学习的核心思想: 将多个有差异的基学习器组合起来, 可以得到比单个学习器更可靠的预测结果.
