# 贝叶斯理论(Bayes Theorem)

$P(A\mid B) = \frac{P(B \mid A) P(A)}{P(B)}$

# 朴素贝叶斯法

## 基本方法

数据集：$T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$

输入：$X \subseteq R^n ,x\in X$

输出 ：$Y=\{c_1,c_2,...,c_i\}$

朴素贝叶斯通过训练数据集学习联合概率分布$P(X,Y)$ 这是通过对其先验概率分布$P(Y=c_k),k = 1,2,...,K$ 以及条件概率分布

$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}|Y=c_k),k=1,2,...,K$ 完成的

> 朴素贝叶斯法对条件概率分布作了条件独立性假设,使其丢失一定的准确性

所以

$P(X=x|Y=c_k)=\prod_{j=1}^{n}P(X^{(j)}=x^{(j)}|Y=c_k)$ 

朴素贝叶斯分类器可表示为

$y=f(x)=arg\ max\frac{P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_kP(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}$ 

实际只关注分子

## 后验概率最大化

～概念等同于期望风险最小化

## 极大似然估计

～用于估计先验概率与条件概率,其中

$P(Y=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)}{N},k=1,2,...K$ $P=(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^N(y_i=c_k)}$ ,a_jl是第j个特征可能取的l个值,$I$ 为指示函数。

$j=1,2,...,n\ \ l=1,2,...,S_j$ 

## 算法

1. 计算先验概率及条件概率（通过数据集）

2. 计算实例x $P(Y=c_k)\prod_{j=1}^nP(X^{(j)}=x^{(j)}|Y=c_k),k=1,2,...,K$ 

3. 找出后验概率最大的分类

## 贝叶斯估计

为了避免极大似然估计导致的概率值为0的情况。在随机变量各个取值的频数上赋予一个正数$\lambda$ 其中$\lambda \ge 1$ 等于1时为拉普拉斯平滑。

$P_\lambda(Y=c_k)=\frac{\sum_{i=1}^{N}I(y_i=c_k)+\lambda}{N+k\lambda}$ 

$P_\lambda(X^{(j)}=a_{jl} |Y=c_k)=\frac{\sum_{i=1}^{N}I(x_i^{(j)}=a_{jl,y_i=c_k)+\lambda}}{\sum_{i=1}^NI(y_i=c_k)+S_j\lambda}$

上式为改变后的先验概率与条件概率。

## 高斯朴素贝叶斯

当数据为连续的时候在计算条件概率时需要求出均值与标准差，根据每一个feature的均值与标准差构建高斯模型，使用高斯模型得出条件概率

> 当数值很小时为了防止underflow可以借助log函数。