# 搜索系统

## 学习资料

- [搜索系统核心技术概述【1.5w字长文】](https://juejin.cn/post/7033227795996246052) 了解基础概念



## 核心模块

### 爬虫
抓取网页的覆盖率、时效性及重要性

- [heritrix3](https://github.com/internetarchive/heritrix3)
- [crawler4j](https://github.com/yasserg/crawler4j)

### 解析
过滤清洗、信息提取以及权重排序


### 索引

- 正排索引
- 倒排索引
    - [lucene](https://lucene.apache.org/)
    - [solr](https://solr.apache.org/) 多模态搜索平台
    - [elasticsearch](https://www.elastic.co/elasticsearch) 搜索和分析引擎
- 向量索引
    - [faiss](https://github.com/facebookresearch/faiss)
    - [hnsw](https://github.com/nmslib/hnswlib)
    - [milvus](https://github.com/milvus-io/milvus)

### Query理解

Query理解主要包含Query预处理、Query纠错、Query扩展、Query归一、联想词、Query分词、意图识别、term重要性分析、敏感Query识别、时效性识别等，实际应用中会构建一个可插拔的pipeline完成Query理解流程。

- Query分词
    - [jieba](https://github.com/fxsjy/jieba)
    - [analysis-ik](https://github.com/infinilabs/analysis-ik)
- term重要性分析
    - [LDA](https://scikit-learn.org/stable/modules/lda_qda.html)
    - [TextRank](https://github.com/davidadamojr/TextRank)
- 意图识别
- 敏感Query识别
- 时效性识别

### 召回
基于词的传统召回和基于向量的语义召回。


### 排序

#### 经典论文

- [2009] [Learning to Rank for Information Retrieval](https://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/ir/ir13/1_-_learning_to_rank.pdf)
- [2009] [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
    - [BM25算法, Best Matching](https://zhuanlan.zhihu.com/p/79202151)
- [1999.09] [Authoritative sources in a hyperlinked environment](https://dl.acm.org/doi/10.1145/324133.324140) 基于用户query得到topk的网页，然后扩展出一个候选集，然后对候选集进行排序。
    - [[算法系列02] 浅谈HITS算法](https://zhuanlan.zhihu.com/p/206965478) 
- [1998.01] [The PageRank Citation Ranking: Bringing Order to the Web.](http://ilpubs.stanford.edu:8090/422/)
    - [PageRank算法详解](https://zhuanlan.zhihu.com/p/137561088)

#### 排序模型

- 粗排
    - tf-idf
    - bm25
    - embedding-similarity
- 精排
    - pointwise、pairwise、listwise
    - gbdt
    - DNN。dssm、transform

### 评估

评估指标

- 准确率和召回率
- F1值
- AP(Average Precision)  对不同召回率点上的准确率进行平均
- NDCG 
- MAP(Mean Average Precision) 每个query的ap的平均
- MRR(Mean Reciprocal Rank) 每个query第一个正例的位置导数的平均值
