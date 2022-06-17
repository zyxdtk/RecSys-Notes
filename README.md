# 1. RecSys-Notes
记录推荐系统相关的优化经验、学习笔记、面试题。

- [1. RecSys-Notes](#1-recsys-notes)
- [2. 推荐系统适用场景](#2-推荐系统适用场景)
- [3. 推荐系统从0到1](#3-推荐系统从0到1)
  - [3.1. 冷启动](#31-冷启动)
    - [3.1.1. 系统冷启动](#311-系统冷启动)
    - [3.1.2. 新用户冷启动](#312-新用户冷启动)
    - [3.1.3. 新内容冷启动](#313-新内容冷启动)
  - [3.2. 搭建推荐系统](#32-搭建推荐系统)
    - [3.2.1. 数据采集](#321-数据采集)
    - [3.2.2. 数据处理](#322-数据处理)
    - [3.2.3. 推荐算法](#323-推荐算法)
      - [3.2.3.1. 召回](#3231-召回)
      - [3.2.3.2. 排序](#3232-排序)
      - [3.2.3.3. 策略](#3233-策略)
    - [3.2.4. 评估体系](#324-评估体系)
- [4. 推荐系统优化](#4-推荐系统优化)
  - [4.1. 召回优化](#41-召回优化)
    - [4.1.1. 召回的评估](#411-召回的评估)
    - [4.1.2. 召回的迭代路径](#412-召回的迭代路径)
    - [4.1.3. 召回负样本处理](#413-召回负样本处理)
    - [4.1.4. 推荐历史去重](#414-推荐历史去重)
  - [4.2. 排序优化](#42-排序优化)
    - [4.2.1. 排序的评估](#421-排序的评估)
    - [4.2.2. 排序的迭代路径](#422-排序的迭代路径)
    - [4.2.3. label](#423-label)
    - [4.2.4. 多目标融合](#424-多目标融合)
    - [4.2.6. 特征工程](#426-特征工程)
    - [4.2.5. 非结构化特征的处理](#425-非结构化特征的处理)
    - [4.2.7. 样本](#427-样本)
    - [4.2.8. 模型结构](#428-模型结构)
    - [4.2.9. 偏置处理](#429-偏置处理)
    - [4.2.10. 多任务模型](#4210-多任务模型)
    - [4.2.11. 调参](#4211-调参)
  - [4.3. 策略优化](#43-策略优化)
    - [4.3.1. 打造生态：消费者、生产者、平台三方利益兼顾](#431-打造生态消费者生产者平台三方利益兼顾)
    - [4.3.2. 流量扶持：新内容、新生产者、新品类](#432-流量扶持新内容新生产者新品类)
    - [4.3.3. 探索与利用](#433-探索与利用)
    - [4.3.4. 如何缓解头部效应](#434-如何缓解头部效应)
    - [4.3.5. 重排模型](#435-重排模型)
- [5. 推荐系统的未来](#5-推荐系统的未来)
  - [5.1. 对长期收益建模](#51-对长期收益建模)
  - [5.2. 对item组合建模](#52-对item组合建模)
  - [5.3. 极致的时效性](#53-极致的时效性)
  - [5.4. 更丰富的交互信息](#54-更丰富的交互信息)
  - [5.5. 与其他模块的协同](#55-与其他模块的协同)
  - [5.6. 自动化](#56-自动化)

# 2. 推荐系统适用场景
信息过载+无明确意图

# 3. 推荐系统从0到1
在开始搭建推荐系统前，建议可以看看[Google的机器学习最佳实践(共43条)](https://developers.google.com/machine-learning/guides/rules-of-ml)[【汉】](https://www.jiqizhixin.com/articles/2018-05-22-12)。里面讲到进行机器学习的基本方法是：

1. 确保机器学习流程从头到尾都稳固可靠。
2. 从制定合理的目标开始。
3. 以简单的方式添加常识性特征。
4. 确保机器学习流程始终稳固可靠。
  
上述方法将在长时间内取得很好的效果。只要您仍然可以通过某种简单的技巧取得进展，就不应该偏离上述方法。增加复杂性会减缓未来版本的发布。


从0到1就是要解决冷启动问题，冷启动问题可以用产品的办法解决，也可以在推荐系统内解决。

## 3.1. 冷启动
### 3.1.1. 系统冷启动
### 3.1.2. 新用户冷启动
### 3.1.3. 新内容冷启动

## 3.2. 搭建推荐系统



一个完整的推荐系统包括：数据采集、数据处理、推荐算法、评估体系。
### 3.2.1. 数据采集
数据采集包括了：用户信息采集(人群属性、兴趣问卷)，用户行为数据采集(埋点日志)，推荐日志，内容打标。

### 3.2.2. 数据处理
数据处理包括：样本生成、特征工程、报表

### 3.2.3. 推荐算法
经典推荐架构：召回、排序、策略。从0到1的过程中需要特别关注冷启动问题：系统冷启动、用户冷启动、内容冷启动。
#### 3.2.3.1. 召回
冷启动阶段没有太多用户行为数据。可以采集用户信息、多利用item标签、捕捉实时信息。热门召回、人群热门召回、用户采集兴趣召回、用户实时兴趣召回。
另外需要做召回去重。
#### 3.2.3.2. 排序
冷启动阶段最好是用单目标简单模型，把整个流程跑通。
#### 3.2.3.3. 策略
黑白名单、调权、频控、打散、保量


### 3.2.4. 评估体系
评估体系包括：在线评估（ABtest、报表）、离线评估。

# 4. 推荐系统优化

## 4.1. 召回优化

### 4.1.1. 召回的评估
召回率、准确率、hit率、内容覆盖度、基尼指数

### 4.1.2. 召回的迭代路径
多路召回。热门召回、关键词召回、i2i、u2i、模型召回。
模型召回：DSSM、TDM

### 4.1.3. 召回负样本处理

### 4.1.4. 推荐历史去重

## 4.2. 排序优化

### 4.2.1. 排序的评估
- 线上评估
  - ABtest。留存、时长、ctr、刷帖、点赞、评论、转发等
  - 模型评估指标。在线auc、gauc 
- 离线评估
  - 分类。auc、guac
  - 回归。rmse、mae、mape
- debug工具
  - 推荐线上服务debug
    - 在推荐的整个留存打印debug信息，然后把debug信息放到debug工具中展示。
  - 模型debug 
    - TensorBoard 

### 4.2.2. 排序的迭代路径
由一轮排序，到包含粗排、精排的两轮排序。甚至随着候选池的增加，可以增加更多轮排序。
由单一目标排序，到多目标融合排序。
有简单模型到复杂模型。LR、FM、Wide&Deep、DIN、MMOE、SNR

### 4.2.3. label 
不同业务线的业务目标，用户关键行为都不一样，需要针对性的建模。另外构建一套样本是成本非常大的事情，包括推动前端增改卖点，大数据做好样本关联，数据校验，累计一段时间样本用于训练，整个周期会非常长。所以一般推荐在一开始就把所有能想到的user-item的行为数据都做好埋点。
- 电商
  - GMV。GMV=DAU*CTR*CVR*下单次数*单均价。以上指标一般是uv维度，按照天、周级别统计。无法作为直接排序label
  - item级别行为。曝光、击、加购、下单、成单、评分。电商的行为在时间维度上可能不较长，如成单、评分等动作的延迟甚至可能是几天。
- 娱乐
  - 用户总停留时长。TotalDur=DAU*刷帖数*贴均时长=(DNU+DAU*留存)*刷帖数*贴均时长。
  - item级别行为。
    - 隐式反馈：曝光、点击、时长、完播
    - 显式反馈：点赞、评论、转发
- 社交
  - 总互动。总互动=DAU*匹配率*互动数。
  - item级别行为。曝光、点击、关注、互动 


### 4.2.4. 多目标融合
引入多目标的几种方式：
- 样本调权
  - 见：[街首页推荐多目标优化之reweight实践：一把双刃剑？](https://zhuanlan.zhihu.com/p/271858727)
- 线性加权。Score = a*scoreA + b * scoreB + c * ScoreC + … + m * ScoreM
- 乘法加权。一般用在电商场景 Score = pow(ctr+ai, bi) * pow(price+aj, bj)
  
[超参数搜索的方式](https://zhuanlan.zhihu.com/p/304373868)：<span id="parameter-tuning"></span>
- 人工调参(babysitting)
- 网格搜索(Grid Search)
- 随机搜索(Random Search)
- 贝叶斯优化
  - [SMBO](https://zhuanlan.zhihu.com/p/53826787)(Sequential model-based optimization)
- 进化算法
  - 基础理论：[帕累托最优](https://zhuanlan.zhihu.com/p/54691447) 
  - [CEM](https://blog.csdn.net/ppp8300885/article/details/80567682)(Cross Entropy Method)
  - [PSO](https://cloud.tencent.com/developer/article/1424756)(Particle Swarm Optimization, 粒子群算法)
  - [NES](https://mofanpy.com/tutorials/machine-learning/evolutionary-algorithm/evolution-strategy-natural-evolution-strategy/)(Natural Evolution Strategy)

业界的一些多目标融合的实践：
- [BIGO | 内容流多目标排序优化](https://toutiao.io/posts/ds6kdac/preview)
- [爱奇艺：多目标排序在爱奇艺短视频推荐中的应用](https://juejin.cn/post/6977633076390133796)
- [快手：多目标排序在快手短视频推荐中的实践](http://www.360doc.com/content/21/0225/09/7673502_963854442.shtml)


### 4.2.6. 特征工程
- 主体维度
  - 用户
    - 人群属性
    - 行为特征
    - 统计特征
  - item
    - 标签
    - 统计特征
  - context
    - 地理位置
    - 时间
    - 推荐tab
- 时效性维度
- 批量特征
- 实时特征

### 4.2.5. 非结构化特征的处理
NLP和CV的引入
视频、图片的自动打标。场景识别、人脸识别、OCR
文本的关键词提取。

### 4.2.7. 样本
样本=label+特征。
label一般来自客户端的埋点，也有用到服务端数据的(比如，点赞、成单等数据服务端也有记录)。一般都会有多个label日志(曝光、点击、时长、点赞等)，需要把这些日志关联起来。特征来源于推荐埋点日志或离线批处理特征，最好是把特征都埋在推荐replay日志中，这样可以避免离线在线特征不一致问题。label埋点和推荐replay日志的关联可以通过约定的唯一性id来确定，如：user_id、item_id对，或者唯一性的session_id。

label埋点日志关联，可以在客户端关联，也可以在大数据这里关联。为了减少客户端的复杂性，现在一般都是无代码埋点，只埋事件日志，然后由大数据这边关联。中间存在一些问题：
- 日志丢失
  - 客户端要做日志的持久化, 会导致日志延迟比较大，不过总比丢了好
  - 日志回传要做好丢包重传等机制
  - 服务端接收后一般是直接到kafka
- user-itme粒度的日志回传事件跨度大
  - 天级别批处理，要做好跨天的处理 
  - 实时关联，一般设定cache时间窗口
    - 负样本cache,[skip-above](https://tech.meituan.com/2016/04/21/online-learning.html)
    - [负样本不cache](https://www.infoq.cn/article/lTHcDaZelZgC639P1P5q)，会有False Negative问题
      - 样本重要性采样(importance sampling)
      - FN矫正
      - PU loss (Positive-unlabeled loss) 问题是来多个正样本怎么办
      - 延迟反馈 Loss




### 4.2.8. 模型结构
正则：L1、L2、Dropout、BatchNorm、Relu
序列建模：Attention、Transformer、GRU、LSTM
参数共享：gate、routing
优化器：FTRL、Adagrad、Adam

### 4.2.9. 偏置处理
偏置的类型：
- 点击位置偏置
- 视频时长偏置

处理偏置的方法：
- 模型训练学习bias，serv的时候去掉bias
  - 如: [Youtube推荐算法中的多任务模型和消偏](https://zhuanlan.zhihu.com/p/261170030)


### 4.2.10. 多任务模型
多任务模型结构：  
- [ESMM](https://zhuanlan.zhihu.com/p/57481330)[【code】](https://github.com/busesese/ESMM)
- 2018年 [MMOE](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
- 2019年 [MKR](https://arxiv.org/pdf/1901.08907.pdf)[【code】](https://github.com/hwwang55/MKR)[【汉】](https://blog.csdn.net/qq_40006058/article/details/89977929)
- 2019年 [Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf)[【汉】](https://zhuanlan.zhihu.com/p/82584437)
- 2019年 [SNR:Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning](https://ojs.aaai.org//index.php/AAAI/article/view/3788)

多任务模型的loss设计：
- [多个label的loss如何平衡](https://blog.csdn.net/qq_34527082/article/details/100048864)  [【知乎上的讨论】](https://www.zhihu.com/question/359962155)
  - 2017年 [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://arxiv.org/abs/1711.02257) 对更新快的任务，使用小一点的学习率，对更新慢的任务，使用大一点的学习率。
  - 2018年 [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) [【code】](https://github.com/ranandalon/mtl) 基于不确定性，最终效果比我人工调参的结果好一丢丢
  - 2019年 [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/pdf/1810.04650.pdf)

业界实践 
- [多目标排序各大公司落地](https://zhuanlan.zhihu.com/p/341345727)

### 4.2.11. 调参

[超参数搜索方法](#parameter-tuning)

模型的自动调参工具(AutoML)：
- [Ray Tune](https://zhuanlan.zhihu.com/p/364613087)

## 4.3. 策略优化

### 4.3.1. 打造生态：消费者、生产者、平台三方利益兼顾

### 4.3.2. 流量扶持：新内容、新生产者、新品类

### 4.3.3. 探索与利用
缓解回音壁、保持多样性、提供惊喜和新鲜感

### 4.3.4. 如何缓解头部效应

### 4.3.5. 重排模型

- [微信「看一看」 推荐排序技术揭秘](http://blog.itpub.net/31559354/viewspace-2704029/) 重排用到了DQN
  


# 5. 推荐系统的未来

## 5.1. 对长期收益建模
强化学习在推荐的应用：
- 2018年 [Top-K Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/abs/1812.02353)[【汉】](https://zhuanlan.zhihu.com/p/71601897) 据说获得了Youtube近两年单次上线的最高收益，看起来是召回阶段，召回的优化效果都这么牛逼！
  - [Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
  - 2015年 [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- 2019年 [Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology](https://arxiv.org/abs/1905.12767)[【汉】](https://zhuanlan.zhihu.com/p/83387560) 这个是用来排序的ctr*Q值
  - [sarsa 算法](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-sarsa/) 
- [Values of User Exploration in Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3460231.3474236) 评估探索对长期收益的影响。对比了两种探索的做法：Entropy Regularization 和 Intrinsic Motivation(其实就是对没见过的item加一个提权)。多样性和新颖性并不一定带来用户体验上升，并且过犹不及。惊喜度才是长期收益的关键。
  
强化学习的资料：
- [openai的强化学习课程](https://spinningup.openai.com/en/latest/)
- [Facebook-朱哲清Bill 关于RL的研究进展的描述](https://www.zhihu.com/question/404471029/answer/2485186947)

## 5.2. 对item组合建模

## 5.3. 极致的时效性
实时特征、实时模型、端上重排
- 实时特征。这个最容易，时效性可以比模型高。
- 实时模型
  - [半实时](https://zhuanlan.zhihu.com/p/75597761)
    - GBDT+LR
    - Wide&Deep 
  - 全实时。现在都有ps了，模型本身是可以实时训练的。这里有几个问题
    - 模型的batch大小。为了凑齐batch最长等多久。
    - 更新serving模型的实践间隔。一个是工程问题，大模型参数同步也要实践。另外一个是更新太快不一定有效果。对广告这种特别依赖id，且素材id更新非常频繁的，可能会比较有用。
- 端上重排
  - 阿里 [EdgeRec：边缘计算在推荐系统中的应用](https://mp.weixin.qq.com/s/O806chMT_BFzkA-Tuv94Hw) 
  - 快手 [渠江涛：重排序在快手短视频推荐系统中的演进](https://mp.weixin.qq.com/s/OTyEbPCBh1NHogPM7bBtvA)


实时优化器。本质上就是要加好正则，避免被少数样本带偏了
  - FTRL(Follow The Regularized Leader)
    - [Online Learning算法理论与实践](https://tech.meituan.com/online_learning.html) 主要介绍Online Learning的基本原理和两种常用的Online Learning算法：FTRL（Follow The Regularized Leader）和BPR（Bayesian Probit Regression）。基本原理理解，具体公式推导有点晕。
    - [RDA， FTRL 在线学习算法最初的优化公式是依据什么得到的？](https://www.zhihu.com/question/266462198/answer/309780073)
    - [在线学习（Online Learning）导读](https://zhuanlan.zhihu.com/p/36410780)
    - [Online Learning and Online Convex Optimization](http://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf) 其中的2.3节讲到FTRL
  - MIRA(Margin-infused relaxed algorithm)
    - [浅谈在线机器学习算法](http://yjliu.net/blog/2012/07/14/a-brief-talk-about-online-learning.html) 提到了Perceptron算法用于二分类问题，MIRA算法用于多类问题。
  - Online gradient descent: Logarithmic Regret Algorithms for Online Convex Optimization
  - Dual averaging: Dual Averaging Methods for Regularized Stochastic Learning and Online Optimization
  - Adagrad: Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
  - PA(Online Passive-Aggressive Algorithms)
    - [Online Passive-Aggressive Algorithms](https://link.zhihu.com/?target=http%3A//www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf)  Shai Shalev-Shwartz于2006年发表，提出了一种基于边界的在线学习算法簇，可以支持多种预测任务。具体来说可以用于二分类、回归、多分类、序列预测等任务，使用hingeloss损失函数。
    - [Online Learning：Theory, Algorithms, and Applications](http://ttic.uchicago.edu/~shai/papers/ShalevThesis07.pdf) Shai Shalev-Shwartz的博士论文，2007年发表，旨在建立一个支持多种预测任务的在线学习簇。
  - 综述性文章
    - [Online Learning and Stochastic Approximations](http://leon.bottou.org/publications/pdf/online-1998.pdf) L´eon Bottou，AT&T实验室，2018年修订版，在线学习和随机优化的解释文章。
    - [Online Learning and Online Convex Optimization](http://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf) Shai Shalev-Shwartz于2011年写的一篇综述性论文

## 5.4. 更丰富的交互信息

## 5.5. 与其他模块的协同
迁移学习。引入其他人口(如搜索)，其他app的用户信息。

## 5.6. 自动化
减少人工干预



