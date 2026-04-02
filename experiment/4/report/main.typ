#import "dependency.typ": *
#import "template.typ": *



#set page(height: auto)
#set par(justify: true)

#show: report.with(
    institute: "计算机科学与技术",
    course: "机器学习",
    student-id: "202400130242",
    student-name: "彭靖轩",
    class: "24智能",
    date: datetime.today(),
    lab-title: "Experiment 3: Linear Discriminant Analysis",
    exp-time: "2",
)

#show figure.where(kind: "image"): it => {
    set image(width: 67%)
    it
}

#show: zebraw.with(
    lang: false,
)

#exp-block([
    = 实验目的
])
#exp-block([
    = 硬件环境
    - CPU: 9600x
])
#exp-block([
    = 软件环境
    - Python 3.10
])
#exp-block()[
    = 实验步骤与内容
]#exp-block()[
    = 结论分析与体会
]
