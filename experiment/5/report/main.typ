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
    lab-title: "Experiment5: 数据预处理",
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
    - 熟悉数据预处理的基本方法
    - 通过对数据的观察和分析,掌握异常值的处理方法
    - 通过对数据的观察和分析,掌握特征处理的方法
    - 通过对数据的观察和分析,掌握特征编码的方法
    - 通过对数据的观察和分析,掌握无量纲化的方法
    - 通过对数据的观察和分析,掌握特征选择的方法
    - 通过对数据的观察和分析,掌握数据清洗的方法
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
    == 准备数据
    为了方便,每个阶段处理得到的数据均保存下来了.
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    sns.set_theme(style='whitegrid')
    pd.set_option('display.max_columns', 200)

    RAW_PATH = 'res/googleappstorev1.csv'
    STEP2_PATH = 'output/step2_clean.csv'
    STEP4_PATH = 'output/step4_size.csv'
    STEP5_PATH = 'output/step5_price.csv'
    STEP7_PATH = 'output/step7_type.csv'
    STEP8_PATH = 'output/step8_content.csv'
    STEP9_PATH = 'output/step9_category.csv'
    APPSTORE_V14_PATH = 'output/appstorev1.4.csv'
    FINAL_PATH = 'output/AppDataV2.csv'

    os.makedirs('output', exist_ok=True)
    ```
    == 缺失值查看
    查看缺失值的情况如@F1
    ```python
    raw_df = pd.read_csv(RAW_PATH, index_col=0)
    print('原始数据形状:', raw_df.shape)
    display(raw_df.head())
    display(raw_df.isnull().any())
    display(raw_df.isnull().sum())
    ```
    #figure(image("../output/1.png"))<F1>
    == 缺失值处理和异常值处理
    进行缺失值处理和异常值处理,得到@F2
    ```python
    df2 = pd.read_csv(RAW_PATH, index_col=0).copy()
    df2['Rating'] = df2['Rating'].fillna(df2['Rating'].median())
    df2 = df2.dropna().drop_duplicates().reset_index(drop=True)
    df2.to_csv(STEP2_PATH)
    print('Step2 输出:', STEP2_PATH, '形状=', df2.shape)
    display(df2.isnull().sum())
    df2.head()
    ```
    #figure(image("../output/2.png"))<F2>
    == Rating 异常值观察
    对 Rating 进行异常值观察,得到@F3 和@F4
    ```python
    df3 = pd.read_csv(STEP2_PATH, index_col=0)
    display(df3[['Rating']].describe())
    plt.figure(figsize=(8, 4))
    df3['Rating'].hist(bins=20, color='steelblue', edgecolor='black')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Rating Distribution')
    plt.show()
    ```
    #figure(image("../output/3.png"))<F3>
    #figure(image("../output/4.png"))<F4>
    == Size 特征处理 + 异常值观察
    对 Size 特征进行处理和异常值观察,得到@F5 和@F6
    ```python
    df4 = pd.read_csv(STEP2_PATH, index_col=0).copy()
    size_raw = df4['Size'].astype(str).str.strip()
    k_mask = size_raw.str.endswith('k')
    m_mask = size_raw.str.endswith('M')
    size_num = pd.Series(np.nan, index=df4.index, dtype='float64')
    size_num.loc[m_mask] = pd.to_numeric(size_raw.loc[m_mask].str.replace('M', '', regex=False), errors='coerce')
    size_num.loc[k_mask] = pd.to_numeric(size_raw.loc[k_mask].str.replace('k', '', regex=False), errors='coerce') / 1024
    df4['Size'] = size_num
    df4['Size'] = df4['Size'].fillna(df4.groupby('Category')['Size'].transform('mean'))
    df4['Size'] = df4['Size'].fillna(df4['Size'].median()).round(3)
    df4.to_csv(STEP4_PATH)
    print('Step4 输出:', STEP4_PATH, '形状=', df4.shape)
    display(df4['Size'].describe())
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df4, x='Size', y='Rating', s=12, alpha=0.5, color='orangered')
    plt.title('Size vs Rating')
    plt.show()
    ```
    #figure(image("../output/5.png"))<F5>
    #figure(image("../output/6.png"))<F6>
    == Price 特征处理 + 异常值观察
    对 Price 特征进行处理和异常值观察,得到@F7 和@F8
    ```python
    df5 = pd.read_csv(STEP4_PATH, index_col=0).copy()
    df5['Price'] = pd.to_numeric(df5['Price'].astype(str).str.replace('$', '', regex=False), errors='coerce').fillna(0.0)
    df5.to_csv(STEP5_PATH)
    print('Step5 输出:', STEP5_PATH, '形状=', df5.shape)
    display(df5['Price'].describe())
    display(df5.loc[df5['Price'] > 300, ['App', 'Category', 'Rating', 'Reviews', 'Price']].head(10))
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df5, x='Price', y='Rating', scatter_kws={'s': 12, 'alpha': 0.4}, line_kws={'color': 'black'})
    plt.title('Price vs Rating')
    plt.show()
    ```
    #figure(image("../output/7.png"))<F7>
    #figure(image("../output/8.png"))<F8>
    == Installs + Reviews:特征处理、无量纲化、异常值处理
    对 Installs 和 Reviews 进行特征处理、无量纲化和异常值处理,得到@F9 和@F10
    ```python
    df6 = pd.read_csv(STEP5_PATH, index_col=0).copy()
    display(df6[['Installs', 'Reviews']].head())

    df6['Installs'] = pd.to_numeric(
        df6['Installs'].astype(str).str.replace(',', '', regex=False).str.replace('+', '', regex=False),
        errors='coerce'
    )
    df6['Reviews'] = pd.to_numeric(df6['Reviews'], errors='coerce')
    df6['Installs'] = df6['Installs'].fillna(df6['Installs'].median())
    df6['Reviews'] = df6['Reviews'].fillna(df6['Reviews'].median())

    def iqr_clip(s):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    before_stats = df6[['Installs', 'Reviews']].describe()
    df6['Installs'] = iqr_clip(df6['Installs'])
    df6['Reviews'] = iqr_clip(df6['Reviews'])
    after_stats = df6[['Installs', 'Reviews']].describe()
    display(before_stats)
    display(after_stats)

    for col in ['Installs', 'Reviews']:
        log_col = f'{col}_log1p'
        norm_col = f'{col}_norm'
        df6[log_col] = np.log1p(df6[col])
        cmin, cmax = df6[log_col].min(), df6[log_col].max()
        df6[norm_col] = (df6[log_col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0

    display(df6[['Installs', 'Reviews', 'Installs_norm', 'Reviews_norm']].head())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(y=df6['Installs'], ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Installs (After IQR)')
    sns.boxplot(y=df6['Reviews'], ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Reviews (After IQR)')
    sns.histplot(df6['Installs_norm'], kde=True, ax=axes[1, 0], color='skyblue')
    axes[1, 0].set_title('Installs_norm')
    sns.histplot(df6['Reviews_norm'], kde=True, ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title('Reviews_norm')
    plt.tight_layout()
    plt.show()

    appstore_v14 = df6[['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']].copy()
    appstore_v14.to_csv(APPSTORE_V14_PATH)
    print('Step6 输出:', APPSTORE_V14_PATH, '形状=', appstore_v14.shape)
    ```
    #figure(image("../output/9.png"))<F9>
    #figure(image("../output/10.png"))<F10>
    == Type 编码
    对 Type 进行编码,得到@F11 和@F12
    ```python
    df7 = pd.read_csv(APPSTORE_V14_PATH, index_col=0).copy()
    plt.figure(figsize=(5, 3))
    df7['Type'].value_counts().plot(kind='bar', color=['#4C78A8', '#F58518'])
    plt.title('Type Distribution')
    plt.show()
    df7['Type'] = df7['Type'].map({'Free': 1, 'Paid': 2}).astype('Int64')
    df7.to_csv(STEP7_PATH)
    print('Step7 输出:', STEP7_PATH, '形状=', df7.shape)
    df7.head()
    ```
    #figure(image("../output/11.png"))<F11>
    #figure(image("../output/12.png"))<F12>
    == Content Rating 处理与编码
    对 Content Rating 进行处理与编码,得到@F13 ,@F14 和@F15
    ```python
    df8 = pd.read_csv(STEP7_PATH, index_col=0).copy()
    display(df8['Content Rating'].value_counts())
    display(df8[df8['Content Rating'] == 'Unrated'][['App', 'Category', 'Rating', 'Reviews', 'Installs']])
    df8 = df8[df8['Content Rating'] != 'Unrated'].reset_index(drop=True)

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df8, x='Content Rating', y='Rating')
    plt.xticks(rotation=45)
    plt.title('Content Rating vs Rating')
    plt.show()

    content_order = {'Everyone': 0, 'Everyone 10+': 1, 'Teen': 2, 'Mature 17+': 3, 'Adults only 18+': 4}
    df8['Content Rating'] = df8['Content Rating'].map(content_order).astype('Int64')
    df8.to_csv(STEP8_PATH)
    print('Step8 输出:', STEP8_PATH, '形状=', df8.shape)
    df8.head()
    ```
    #figure(image("../output/13.png"))<F13>
    #figure(image("../output/14.png"))<F14>
    #figure(image("../output/15.png"))<F15>
    == Category One-Hot 编码
    对 Category 进行 One-Hot 编码,得到@F16 ,@F17 和@F18
    ```python
    df9 = pd.read_csv(STEP8_PATH, index_col=0).copy()
    display(df9['Category'].value_counts().head(10))
    plt.figure(figsize=(18, 6))
    sns.barplot(data=df9, x='Category', y='Rating', hue='Type', estimator=np.mean, errorbar=None)
    plt.xticks(rotation=80)
    plt.title('Category-Type-Rating')
    plt.show()
    df9 = pd.get_dummies(df9, columns=['Category'], dtype='int8')
    df9.to_csv(STEP9_PATH)
    print('Step9 输出:', STEP9_PATH, '形状=', df9.shape)
    df9.head()
    ```
    #figure(image("../output/16.png"))<F16>
    #figure(image("../output/17.png"))<F17>
    #figure(image("../output/18.png"))<F18>
    == Genres 编码 + 删除 App
    对 Genres 进行编码,删除 App,得到@F19 和最终数据
    ```python
    df10 = pd.read_csv(STEP9_PATH, index_col=0).copy()
    df10['Genres'] = df10['Genres'].astype(str).apply(lambda x: x.split(';')[0].strip())
    genre_map = {g: i for i, g in enumerate(sorted(df10['Genres'].unique()))}
    df10['Genres'] = df10['Genres'].map(genre_map).astype('Int64')
    df10 = df10.drop(columns=['App'])
    df10.to_csv(FINAL_PATH)
    print('最终输出:', FINAL_PATH, '形状=', df10.shape)
    df10.head()
    ```
    #figure(image("../output/19.png"))<F19>
]
#exp-block()[
    = 结论分析与体会
    通过本次实验,我掌握了数据预处理的基本方法,包括缺失值处理、异常值处理、特征处理、特征编码、无量纲化和特征选择等方法.通过对数据的观察和分析,我学会了如何处理异常值和缺失值,如何进行特征处理和编码,以及如何进行无量纲化和特征选择.这些技能对于后续的机器学习模型训练和评估非常重要,能够帮助我们更好地理解数据,提高模型的性能和泛化能力.
]
