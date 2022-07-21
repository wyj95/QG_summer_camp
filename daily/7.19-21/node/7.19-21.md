# MAS

[toc]

### 概念

****

##### 拉普拉斯矩阵

L= D - A ， 性质：

****

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719203156.png)

****

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719203415.png)

****

##### 归一化

​	通常，我们可以通过下面的方法对矩阵进行归一化

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719202152.png)

****

##### 矩阵不可约

若矩阵A无法通过线性变换转化成（一般为右）上三角形式，则其不可约，同时**其充要条件是对应矩阵为强连通图**

****

##### 非负矩阵相关

全部元素非负， 若行（列）sum==1 -> 行（列）随机矩阵， 双随机矩阵

****

##### 随机矩阵的周期性

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719203748.png).

****

##### 矩阵的谱半径

矩阵的最大特征值![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719204021.png)

****

##### 本原矩阵和特征向量记法

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719204116.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719204243.png)

****

##### 系统趋于一致

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719204604.png)

****

##### 平均一致性算法

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719215551.png)

****

##### 连续分布式算法



![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719215801.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719215817.png)

****

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719215846.png)

****

##### 网络的代数连通度

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719220442.png)

****

##### 平衡图

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719220551.png)

人话就是入度等于出度

****

##### 连续分布式算法2

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719221431.png)

****

 ##### 离散时间算法

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719221558.png)

****

##### Perron 矩阵与对应引理

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719221617.png)

****

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719221908.png)

****

##### 离散分布式算法一致性收敛

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220719222435.png)

****

##### 离散一致拓扑的周期性

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720084039.png)

****

##### 离散周期性收敛

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720083747.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720083826.png)

****

##### 离散最终连通及其收敛性

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720083747.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720084120.png)

![image-20220720084134517](C:\Users\wyj\AppData\Roaming\Typora\typora-user-images\image-20220720084134517.png)

****

##### 代数连通度和收敛速度

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720105955.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720110145.png)

部分符号可参考**拉普拉斯矩阵**

****

##### 一致均衡状态

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720111906.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220720112643.png)

****

##### 不明觉厉的玩意儿1 X 2

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721152120.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721152033.png)



****

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721152308.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721152329.png)

****

##### 时滞一致性算法

对称算法（智能体本身检测信息和接收到的信息都有时滞）：

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721154114.png)

****

不对称性算法（智能体本身检测信息没有时滞，仅接收到的信息有时滞）：

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721154733.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721154744.png)

****

##### 不明觉厉的玩意儿2之时滞线性系统

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721155128.png)

Metzler矩阵为飞对角线元素非负的矩阵

****

##### 时变时滞算法

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721155754.png)

##### 不明觉厉的玩意儿3之时变时滞固定网络拓扑

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721155840.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721155906.png)

****

##### 不明觉厉的玩意儿4之时变时滞切换网络拓扑

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220721160214.png)

























