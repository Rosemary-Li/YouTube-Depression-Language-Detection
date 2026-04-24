# YouTube 抑郁信号检测

![Python](https://img.shields.io/badge/Python-3.11-blue)
![VADER](https://img.shields.io/badge/NLP-VADER-orange)
![TextBlob](https://img.shields.io/badge/NLP-TextBlob-orange)
![AFINN](https://img.shields.io/badge/NLP-AFINN-orange)
![LIWC](https://img.shields.io/badge/Lexicon-LIWC-purple)
![GMM](https://img.shields.io/badge/Clustering-GMM-teal)
![HDBSCAN](https://img.shields.io/badge/Clustering-HDBSCAN-teal)
![tSNE](https://img.shields.io/badge/Visualization-PCA--tSNE-9cf)
![sklearn](https://img.shields.io/badge/Model-sklearn--Pipeline-red)
![YouTube](https://img.shields.io/badge/Data-YouTube%20API%20v3-ff0000)
![Status](https://img.shields.io/badge/Status-Active-success)

> 衡量抑郁指示性语言在多类高流量 YouTube 视频评论区中的自然出现频率与语言模式。

本项目从五个内容类别的主流高流量 YouTube 视频中采集评论，使用 NLP 技术检测在日常评论区中自然出现的抑郁相关语言信号。我们并不主动搜索抑郁内容，而是衡量这类信号在普通公开话语中出现的频率与形式。

本项目**不**试图诊断抑郁症。我们检测的是_抑郁指示性语言信号_，将其作为在野外场景中自发、自我披露的抑郁相关表达的代理指标。

---

## 研究问题

1. 在不同 YouTube 内容类别中，抑郁指示性语言的自然出现频率有何差异？
2. 视频层面的内容框架在多大程度上影响评论中抑郁相关自我披露的语言模式？

---

## 设计理念

大多数已有研究从抑郁专属社区（如 r/depression）采集数据，这会引入**自我选择偏差**——在那里找到的用户已经主动将自己认同为抑郁社区的一员。本项目采取不同路径：从五个类别的高流量普通内容视频中采样，再逐条评论检测抑郁指示性语言。这更准确地反映了抑郁相关表达在日常网络话语中的真实**出现率**，同时允许我们比较这种表达在不同内容语境下的差异。

分析单位是**评论**，而非视频。一个音乐视频本身与抑郁无关，但其下方出现"_我最近一直很挣扎，这是唯一让我撑下去的东西_"这样的评论，就具有研究意义。我们在评论层面、跨视频语境地测量信号。

---

## 项目结构

```
.
├── data/
│   ├── raw/                  # 原始 API 响应（视频 + 评论）
│   ├── processed/            # 清洗后的特征工程数据
│   └── outputs/              # 模型输出、聚类标签、图表
│
├── notebooks/                # 探索性分析与可视化
│
├── src/
│   ├── collect.py            # Step 0：YouTube Data API 爬取
│   ├── video_analysis.py     # Step 1：视频级 NLP 特征提取
│   ├── comment_analysis.py   # Step 2：三层抑郁信号检测
│   ├── linking.py            # Step 3：聚类、可视化、跨类别对比
│   └── modeling.py           # Step 4：带交叉验证的监督分类
│
├── requirements.txt
└── README.md
```

---

## Pipeline 总览

| 步骤 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- |
| 0 — 数据采集 | `collect.py` | API 密钥、类别列表 | `data/raw/videos.json`、`data/raw/comments.json` |
| 1 — 视频分析 | `video_analysis.py` | `videos.json` | `data/processed/video_features.csv` |
| 2 — 评论分析 | `comment_analysis.py` | `comments.json` | `data/processed/comment_features.csv` |
| 3 — 跨类别关联 | `linking.py` | 两份特征 CSV | `data/outputs/heatmap.png`、`data/outputs/tsne.png` |
| 4 — 建模 | `modeling.py` | 两份特征 CSV | `data/outputs/model_results.json` |

---

## 环境配置

```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

需要 YouTube Data API v3 密钥。在项目根目录创建 `.env` 文件：

```
YOUTUBE_API_KEY=your_key_here
```

---

## 运行方式

按顺序执行各步骤：

```bash
# Step 0：采集数据
python src/collect.py

# Step 1：视频级特征提取
python src/video_analysis.py

# Step 2：评论级抑郁信号检测
python src/comment_analysis.py

# Step 3：跨类别对比与可视化
python src/linking.py

# Step 4：带交叉验证的监督分类
python src/modeling.py
```

---

## 数据采集策略

视频从五个预定义内容类别中采样，每个类别代表一种典型的日常 YouTube 内容类型。在每个类别内，按观看量排序并选取前 N 个视频。采集和排序视频时不使用任何抑郁相关关键词——抑郁相关性完全在 Step 2 的**评论层面**事后判定。

| 类别 | 在研究中的角色 |
| --- | --- |
| 心理健康 | 基准——预期抑郁信号率较高 |
| 健身 / 健康 | 与心理健康相邻；具有激励性框架 |
| 音乐 / 娱乐 | 高流量主流内容；受众广泛 |
| 新闻 / 时事 | 情绪带入感强但非个人化框架 |
| 个人 Vlog | 第一人称叙事；自我披露潜力较高 |

**每个视频采集：** 按点赞数排序的前 50 条评论，以 `video_id` 作为外键存储。目标：每类别约 40–50 个视频，共约 200–250 个视频，约 10,000–12,500 条评论。

---

## Step 1 — 视频级特征提取

由于视频类别在 Step 0 采集时已确定，Step 1 专注于从视频元数据（标题和描述）中提取 NLP 特征，用于刻画每个视频的语气与框架风格。这些特征在 Step 3 中与评论级数据通过 `video_id` 关联，用于检验视频框架是否影响评论中的抑郁语言模式。

**每个视频提取的特征：**

| 特征 | 方法 | 用途 |
| --- | --- | --- |
| `title_sentiment` | VADER compound 分 | 视频标题的情感倾向 |
| `desc_sentiment_vader` | VADER compound 分 | 视频描述的情感倾向 |
| `desc_sentiment_textblob` | TextBlob 极性分 | 辅助情感信号 |
| `desc_subjectivity` | TextBlob 主观性分 | 描述的个人化 / 主观化程度 |
| `desc_word_count` | 词元计数 | 描述长度 |
| `desc_lexical_diversity` | 唯一词元数 / 总词元数 | 描述的词汇丰富度 |

输出：`data/processed/video_features.csv`，每行对应一个视频。

---

## Step 2 — 抑郁信号检测（三层）

抑郁信号检测在评论层面进行。仅凭情感分析不足以判断——负面情感并不意味着抑郁（例如"这个视频太烂了"是负面的，但与抑郁无关）。我们使用三层递进的检测方法：

**第一层 — 多工具情感评分**（宽泛信号）

使用三种独立工具对每条评论进行情感评分，得到更稳健的情感信号，而非依赖单一模型：

| 工具 | 分数范围 | 说明 |
| --- | --- | --- |
| VADER | compound ∈ [−1, 1] | 专为短社交媒体文本优化 |
| TextBlob | polarity ∈ [−1, 1] | 基于词典，稳定的基准信号 |
| TextBlob | subjectivity ∈ [0, 1] | 高主观性与个人自我披露高度相关 |
| AFINN | 整数加和 | 词级别的情感效价评分 |

**第二层 — TF-IDF 加权抑郁词典匹配**（领域专项）

将评论与心理健康词典（LIWC 类别：负面情绪、悲伤、焦虑、健康）以及抑郁专项词表进行匹配。不同于二元的"词在/不在"匹配，TF-IDF 加权以抑郁词典为词汇表（锁定 vocabulary），衡量抑郁相关词汇在评论中的相对突出程度，并扩展至二元词组（如"feel hopeless"、"can't sleep"）以提升覆盖率。

**第三层 — 自我披露模式检测**（高精度）

使用词性标注（spaCy）识别第一人称代词 + 抑郁指示性词汇的组合（如"I feel hopeless"、"I've been struggling"、"I can't do this anymore"）。这类评论代表真实的个人披露，而非对视频内容的反应，具有最强的研究信号。

**同步提取的辅助文本特征：**

| 特征 | 计算公式 | 依据 |
| --- | --- | --- |
| `lexical_diversity` | 唯一词元数 / 总词元数 | 抑郁语言通常词汇更为单调重复 |
| `word_count` | 词元计数 | 极短评论（<5 词）不太可能是真实的自我披露 |

每条评论综合三层信号得到一个**抑郁信号复合分**，跨类别对比在此分数上进行。

---

## Step 3 — 跨类别对比与可视化

**聚类**

评论嵌入向量（`sentence-transformers/all-MiniLM-L6-v2`）使用两种互补方法进行聚类：

- **HDBSCAN** — 基于密度，无需预设簇数，天然处理噪声点。
- **高斯混合模型（GMM）** — 软聚类；每条评论以概率归属于各个簇，而非强制分入单一类别。对情绪模糊的评论更为合适。最优组件数通过 AIC/BIC 选择。

两种聚类结果均报告并对比可解释性。

**降维与可视化**

采用两步流水线：

1. **PCA** 将嵌入向量从高维降至 50 个主成分，去除噪声同时保留全局结构。
2. **t-SNE** 进一步降至 2D，保留局部邻域关系用于可视化。

该流水线比直接 2D 投影产生更清晰的簇分离效果，符合高维文本嵌入的主流处理实践。

**统计检验**

使用 ANOVA 和 KS 检验对五个类别的抑郁信号率差异进行显著性检验。

---

## Step 4 — 监督分类（可选）

若有标注样本，可训练监督分类器预测某条评论是否含有抑郁信号。使用标准化 Pipeline（TF-IDF 特征提取 + 分类器）和 5 折交叉验证，对比三个模型：

| 模型 | 说明 |
| --- | --- |
| Logistic Regression | 文本分类的强基准模型 |
| Naive Bayes | 计算高效；适合稀疏 TF-IDF 特征 |
| SVM（线性核） | 在高维文本上表现稳健 |

全程使用 5 折交叉验证，确保评估结果不依赖单次训练/测试划分。采用 F1-macro 指标，以应对抑郁信号与非信号评论间潜在的类别不平衡问题。

---

## 模型与工具

| 任务 | 模型 / 工具 |
| --- | --- |
| 情感分析（第一层） | VADER、TextBlob、AFINN |
| 抑郁词典匹配（第二层） | LIWC + TF-IDF（二元词组，锁定词汇表） |
| 自我披露检测（第三层） | spaCy 词性标注 + 第一人称代词模式 |
| 评论嵌入 | `sentence-transformers/all-MiniLM-L6-v2` |
| 聚类 | HDBSCAN + GMM（AIC/BIC 选簇数） |
| 降维 | PCA（50D）→ t-SNE（2D） |
| 监督分类 | Logistic Regression、Naive Bayes、SVM（sklearn Pipeline） |
| 交叉验证 | 5 折，F1-macro，sklearn `cross_val_score` |
| 可视化 | `seaborn`、`matplotlib` |

---

## 预期核心结果

主要输出为五个类别的抑郁信号出现率跨类别对比：

| 类别 | 抑郁信号率 | 平均情感分 | 自我披露率 |
| --- | --- | --- | --- |
| 心理健康 | （待测量） | （待测量） | （待测量） |
| 个人 Vlog | （待测量） | （待测量） | （待测量） |
| 健身 / 健康 | （待测量） | （待测量） | （待测量） |
| 新闻 / 时事 | （待测量） | （待测量） | （待测量） |
| 音乐 / 娱乐 | （待测量） | （待测量） | （待测量） |

各类别差异通过 ANOVA / KS 检验进行显著性检验。

---

## 局限性

- **评论 ≠ 评论者**：评论中出现抑郁指示性语言，并不意味着评论者患有抑郁症。
- **情感 ≠ 抑郁**：负面情感是必要但不充分的信号——三层检测方法旨在解决这一问题，但误报仍有可能发生。
- **高赞评论偏差**：按点赞数排序会偏向情绪共鸣性语言；较安静的痛苦表达可能代表性不足。
- **类别采样局限**：五个类别由研究团队定义，并不能穷举所有 YouTube 内容类型。
- **无临床基准真值**：所有标签均基于语言和行为信号的代理指标，而非临床评估。

结论应解读为衡量_公开评论区中抑郁指示性语言模式的自然出现率_，而非临床或诊断性声明。

---

## 团队

<!-- 在此添加成员姓名与分工 -->

---

## 许可证

仅供学术用途。通过 YouTube Data API 采集的数据须遵守 [YouTube 服务条款](https://www.youtube.com/t/terms)。
