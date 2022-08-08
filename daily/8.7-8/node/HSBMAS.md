# HSBMAS

[toc]

## 提出问题

1. 普通算法只考虑到边到边的信息交流，不能考虑到不同阶层之间的信息交流
2. 普通算法确定智能体范围后切换的拓扑就唯一确定了

## 相关概念

### 结点优先级的计算

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807143812.png)

### clusterhead CH

若结点i比其所有的一阶和二阶邻居权重高，则该结点为CH

### doorway DW

满足以下条件的结点i不能是DW，橙色都是簇头

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807142018.png)

### gateway GW

满足以下条件的结点i不能是GW，其中c1，c2可以是CH或DW但至少有一个是CH

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807142412.png)

### 结构裁减



![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807150531.png)

### CS

结点i的可活动范围

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807161405.png)

### CWP

通过邻居计算得到的点，m为1， 2....等维度

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807161503.png)

### 更新方法

在可活动区域CS中尽量靠近CWP（获取更多的邻居的信息）

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807161546.png)

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220807161754.png)

### 算法步骤

1. 根据位置和$r_c$确定初始拓扑结构
2. 根据上述相关概念依次确定CH，DW，GW和NB
3. 裁剪无关拓扑
4. 计算出活动区域和目标点
5. 移动
6. 没收敛就爬回去1继续循环

![](https://wyj-bck.oss-cn-guangzhou.aliyuncs.com/pic/20220808113854.png)

## 实验

pass

## 创新点

pass

