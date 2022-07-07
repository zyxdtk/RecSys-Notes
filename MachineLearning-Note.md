# 1. 机器学习笔记

- [1. 机器学习笔记](#1-机器学习笔记)
- [2. 基础问题](#2-基础问题)
  - [2.1. 过拟合与欠拟合](#21-过拟合与欠拟合)
  - [2.2. 正则化](#22-正则化)
  - [2.3. AUC](#23-auc)
  - [超参搜索](#超参搜索)
- [3. 树模型](#3-树模型)
  - [3.1. xgbost和gbdt的区别](#31-xgbost和gbdt的区别)
  - [3.2. xgboost和lgb的区别](#32-xgboost和lgb的区别)
- [4. DNN](#4-dnn)
  - [4.1. 参数是否可以初始化为0](#41-参数是否可以初始化为0)
  - [4.2. relu](#42-relu)
  - [4.3. 梯度消失和梯度膨胀](#43-梯度消失和梯度膨胀)
- [5. NLP](#5-nlp)
- [6. CV](#6-cv)
- [7. 强化学习](#7-强化学习)
- [8. 参考资料](#8-参考资料)

# 2. 基础问题

## 2.1. 过拟合与欠拟合
过拟合：训练集效果好，测试机效果差。

- 欠拟合
  - 增加特征
  - 增加模型复杂度
  - 减少正则项稀疏
- 过拟合
  - 提高样本量
  - 简化模型
  - 加入正则项或提高惩罚稀疏
  - 是用集成学习
  - dropout
  - early stopping
  - [label smoothing(标签平滑)](https://blog.csdn.net/qq_40176087/article/details/121519888) 避免模型过于自信,让标签不绝对

## 2.2. 正则化
- 数据增强
- L2正则
- L1正则
- dropout，神经元被丢弃的概率为 1 − p，减少神经元之间的共适应。一种廉价的 Bagging 集成近似方法。减少训练节点提高了学习速度。
- drop connect， 在 Drop Connect 的过程中需要将网络架构权重的一个随机选择子集设置为零，取代了在 Dropout 中对每个层随机选择激活函数的子集设置为零的做法。
- 随机pooling
- early stopping

## 2.3. AUC
- ROC曲线下的面积。ROC是用FPR和TPR作为x，y绘制的曲线。
- AUC取值一般在0.5到1之间
- AUC更关注序，对于样本不均衡情况，也能给出合理的评价
- 有几种计算方式
  - 计算ROC曲线下的面积。只能用近似方法去算。
  - 统计逆序对个数。按照pred升序排列。得到正样本的rank累计值R。 (R-M(M-1)/2)/(M*N)。这里M是正样本个数，N是负样本个数。分母是正负样本对的个数，分子是逆序对的个数。

## 超参搜索

[超参数搜索的方式](https://zhuanlan.zhihu.com/p/304373868)
- 人工调参(babysitting)
- 网格搜索(Grid Search)，先用较大步长在较大范围搜索，确定可能的位置，然后逐渐缩小搜索范围和步长。简单有效，但是耗时久，目标函数非凸时容易miss全局最优。
- 随机搜索(Random Search)，在搜索范围内随机选取样本点，样本集足够大也能找到全局最优或者近似解。优点是快，但是也可能miss全局最优。
- 贝叶斯优化，对目标函数的形状进行学习，找到使目标函数向全局最优值提升的参数。优点是充分利用之前的信息。缺点是容易陷入局部最优。
  - [SMBO](https://zhuanlan.zhihu.com/p/53826787)(Sequential model-based optimization)
- 进化算法
  - 基础理论：[帕累托最优](https://zhuanlan.zhihu.com/p/54691447) 
  - [CEM](https://blog.csdn.net/ppp8300885/article/details/80567682)(Cross Entropy Method)
  - [PSO](https://cloud.tencent.com/developer/article/1424756)(Particle Swarm Optimization, 粒子群算法)
  - [NES](https://mofanpy.com/tutorials/machine-learning/evolutionary-algorithm/evolution-strategy-natural-evolution-strategy/)(Natural Evolution Strategy)

# 3. 树模型
## 3.1. xgbost和gbdt的区别
- GBDT是机器学习算法，xgboost是该算法的工程实现
- 传统GBDT是用CART作为基分类器，XGBOOST还支持线性分类器，比如LR或者线性回归
- GBDT只用了一阶导数。xgboost用了二阶泰勒展开。支持自定义损失函数，但是要求一阶、二阶可导
- GBDT每次迭代用全量数据，xgboost有行采样和列采样
- xgboost的目标函数里面有正则项，相当于预剪枝
- xgboost支持对缺失值处理
- xgboost支持并行。并行是在特征粒度上
- 支持统计直方图做近似计算

## 3.2. xgboost和lgb的区别

不同：
- xgboost是level-wise，lightgbm使用leaf-wise，这个可以提升训练速度。但是其实xgboost已经支持leaf-wise
  - leaf-wise的问题是可能忽略未来有潜力的节点
- xgboost单机默认是exact greedy，搜索所有的可能分割点。分布式是dynamic histogram，每一轮迭代重新estimate 潜在split candidate。LightGBM和最近的FastBDT都采取了提前histogram binning再在bin好的数据上面进行搜索。在限定好candidate splits。lightgbm在pre-bin之后的histogram的求和用了一个非常巧妙的减法trick，省了一半的时间。
  - 提前限定分割点然后快速求histogram的方法，实际影响不确定。理论上树越深，需要的潜在分割点越多，可能需要动态训练来更新潜在分割点
- xgboost主要是特征并行，lightgbm是有数据并行、特征并行、投票并行。当时其实xgboost也支持数据并行了。
- lightgbm支持分类特征的many vs many，用G/H排序，然后再分桶
参考：
- [如何看待微软新开源的LightGBM?-陈天奇和柯国霖都有回答](https://www.zhihu.com/question/51644470)


# 4. DNN

## 4.1. 参数是否可以初始化为0
不可以。对于全连接多层神经网络，初始化为0，反向传播过程如下：
- 1）第1次反向传播，仅最后一层权重值更新，前面的权重值不更新（仍为0）
- 2）第2次反向传播，最后一层和倒数第二层权重值更新，前面的权重值不更新（仍为0）
- 3）以此类推，若干次反向传播之后，达到如下状态：
- a）输入层和隐层之间，链接在同一个输入节点的权重值相同，链接在不同输入节点的权重值不同；
- b）两个隐层之间，所有权重值相同
- 输出层和隐层之间，链接在同一个输出节点的权重值相同，链接在不同输出节点的权重值不同
以上分析针对sigmoid激活函数，tanh函数下权重值无论怎么训练都保持为0.relu估计也是都为0
  
参考：[关于神经网络参数初始化为全0的思考](https://zhuanlan.zhihu.com/p/32063750)

## 4.2. relu
- y=max(0,x)
- relu在0处是不可导的，TensorFlow实现默认0点的导数为0
- relu是非饱和激活函数。sigmoid和tanh是饱和激活函数。
  - relu的优势：
    - 非饱和激活函数可以解决梯度消失问题
    - 能加快收敛速度
    - 稀疏表达，防止过拟合
  - 缺点
    - 训练很脆弱，很容易权重就为0了，也就是神经元死亡
- relu变种
  - Leaky Relu 给所有负值一个非零斜率
  - [PRelu](https://blog.csdn.net/shuzfan/article/details/51345832) 是Leaky Relu的一个变体，负值部分的斜率是根据数据来定的。斜率a是用带动量的更新方式。论文中初始值0.25，动量= 动量系数*动量+学习率*偏导
  - RRelu 也是Leaky Relu的一个变体，负值斜率在训练中是随机的，在测试中变成固定值。权重a是一个均匀分布U(l,u) l,u∈[0,1) 

参考：[激活函数ReLU、Leaky ReLU、PReLU和RReLU](https://blog.csdn.net/qq_23304241/article/details/80300149)

## 4.3. 梯度消失和梯度膨胀
原因：反向传播是根据链式法则，连乘每一层的偏导。每一层的偏导是激活函数的导数乘以当前节点的权重。如果每一层的偏导都同时偏向一个方向，都大于1会导致最终梯度爆炸，都小于1会导致梯度消失(变为0)

解决办法：relu可以缓解。但是用了relu的网络也还是存在梯度消失的问题，Leaky ReLU就此应运而生。
参考：
- [梯度消失与梯度爆炸的原因](https://zhuanlan.zhihu.com/p/25631496)
- [在使用relu的网络中，是否还存在梯度消失的问题？]（https://www.zhihu.com/question/49230360）

# 5. NLP

# 6. CV


# 7. 强化学习


# 8. 参考资料

- [算法工程师-机器学习面试题总结](https://github.com/zhengjingwei/machine-learning-interview) 总结得很全