---
title: PoseFormer论文阅读笔记-ICCV2021
toc: true
mathjax: true
tags:
  - 论文
  - ICCV
  - 3D人体姿态估计
categories: 文献阅读笔记
abbrlink: 31545
date: 2023-02-24 17:51:56
---

论文题目为：3D Human Pose Estimation with Spatial and Temporal Transformers。本文提出的模型PoseFormer是第一个用于3D人体姿态估计问题的纯Transformer模型。

<!-- more -->

# 大纲

<img src="https://user-images.githubusercontent.com/56388518/221148541-13482dd7-7af7-4520-b95d-ba2e7675f506.png" alt="" style="zoom:50%;" />

# 引言

3D人体姿态估计的工作通常可分为两类：直接估计方法和2D-to-3D提升方法。直接估计方法从2D图像或视频帧直接推断3D人体姿态。2D-to-3D提升方法从中间估计的2D姿态推断出3D人体姿态。得益于最先进的2D姿态检测器的卓越性能，2D-to-3D提升方法通常优于直接估计方法。

然而，这些2D姿态到3D的映射是非常重要的；由于深度歧义和遮挡，可以从相同的2D姿态生成各种潜在的3D姿态。为了缓解其中的一些问题并保持自然连贯性，最近的许多工作中利用了视频中的时间信息。

由于Transformer的自注意机制，可以清楚地捕获长序列之间的全局相关性。这使得它成为一种特别适合序列数据问题的架构，因此自然可以扩展到3D人体姿态估计。凭借其全面的连接性和表达，Transformer可以跨帧学习到更强的时间表示。

研究表明，Transformer需要特定的设计，才能在视觉任务中实现与CNN相当的性能。它们通常需要超大规模的训练数据集，如果应用于较小的数据集，则需要增强的数据扩充和正则化。此外，现有的视觉Transformer主要局限于图像分类、对象检测和分割，但如何利用Transformer进行3D人体姿态估计仍不清楚。

作者首先将Transformer直接应用于2D-to-3D提升的人体姿态估计。在这种情况下，将给定序列中每个帧的整个2D姿态作为token（如图1a）。虽然这种基线方法在一定程度上是可行的，但它忽略了空间关系（关节到关节）的自然区别，留下了潜在的改进。该基线的一个自然扩展是将每个2D关节坐标视为token，并从序列的所有帧中提供由这些关节形成的输入（如图1b）。然而，在这种情况下，当使用长的帧序列时，token的数量会越来越大（在3D HPE中，可多达243个帧，每帧17个关节，token数量将为243×17=4131）。由于Transformer是计算每个token对另一个token的直接关注，因此模型的内存需求接近不合理的水平。

<img src="https://user-images.githubusercontent.com/56388518/221148757-f7af474a-4e50-483a-9718-0eca0b5a48c2.png" alt="" style="zoom: 33%;" />

<center>图1. 时间Transformer基线及其扩展</center>

基于上述问题，作者提出了PoseFormer，这是首个用于视频中2D-to-3D提升HPE的纯Transformer网络。PoseFormer使用两个Transformer模块直接对空间和时间两个维度进行建模。PoseFormer不仅在时空交叉上产生了强大的表示，而且不会对较长的输入序列产生巨量的token。在更高的层次上，PoseFormer只需从现成的2D姿态估计器中获取一系列检测到的2D姿态，并输出中心帧的3D姿态。

具体地说，作者构建了一个空间Transformer模块来编码每个帧中2D关节之间的局部关系。空间自关注层考虑2D关节的位置信息，并返回该帧的潜在特征表示。而时间Transformer模块分析每个空间特征表示之间的全局相关性，并生成准确的3D姿态估计。

本文的贡献有三个方面：

- 提出了第一个基于Transformer用于2D-to-3D提升3D HPE的模型PoseFormer；
- 设计了一个有效的时空Transformer模型，其中空间Transformer模块编码人体关节之间的局部关系，而时间Transformer模块捕获整个序列中帧之间的全局相关性；
- PoseFormer在Human3.6M和MPI-INF-3DHP数据集上都取得了当时最好的结果。

# 相关工作

### 2D-to-3D提升HPE

2D-to-3D提升方法利用从输入图像或视频帧估计的2D姿态。OpenPose、CPN、AlphaPose和HRNet已被广泛用作2D姿态检测器。基于该中间表示，可以使用多种方法生成3D姿态。

Martinez等人提出了一种简单有效的全连接残差网络，以基于仅来自单个帧的2D关节位置来回归3D关节位置。然而，相较于从单目图像中估计3D人体姿态，视频可以提供时间信息以提高准确性和鲁棒性。Hossain和Little提出了一种使用长短期记忆（LSTM）单元来利用输入序列中的时间信息的循环神经网络。此外还有几项工作利用时空关系和约束（如骨骼长度和左右对称性）来提高性能。Pavlo等人引入了时间卷积网络，以从连续2D序列中估计2D关键点上的3D姿态。而Chen等人添加了骨骼方向模块和骨骼长度模块，以确保视频帧之间的时间一致性。Liu等人利用注意力机制来识别重要帧。

然而，先前最先进的方法依赖于扩展的时间卷积来捕获全局依赖性，这在时间连接性方面固有地受到限制。此外，大多数工作使用简单的操作将关节坐标投影到潜在空间，而不考虑人体关节的运动学相关性。

### 3D HPE中的图神经网络

自然地，人体姿态可以表示为一个图形，其中关节是结点，骨骼是边。图形神经网络（GNN）也已应用于2D-to-3D姿态提升问题，并提供了较好的性能。

Ci等人提出了一个名为本地连接网络（Locally Connected Networks, LCN）的框架，该框架利用全连接网络和GNN操作来编码本地相邻关节之间的关系。Zhao等人解决了图卷积网络（GCN）操作的局限性，特别是如何在结点之间共享权重矩阵，引入语义图卷积运算来学习边的通道权重。

对于本文的PoseFormer，Transformer可以被视为一种具有独特的且通常是有利的图形操作的图形神经网络。具体而言，Transformer的编码器模块基本上形成了一个全连通图，其中使用基于输入条件的、多头自注意力机制来计算边的权重。该操作还包括结点特征的归一化，跨注意力头输出的前馈聚合器，以及使其能够与堆叠层有效缩放的残差连接。与其他图形操作相比，这种操作可能是有利的。

### 视觉Transformer

Carion等人提出了一种用于目标检测和全景分割的Detection Transformer（DETR）。Dosovitskiy等人提出了一种纯Transformer架构，即视觉Transformer（Vision Transformer, ViT），它在图像分类方面达到了最先进的性能。然而，ViT是在需要大量计算资源的大规模数据集ImageNet-21k和JFT300M上训练的。然后，提出了一种数据高效的图像Transformer（Data-efficient image Transformer, DeiT），它基于知识蒸馏后的ViT。对于HPE等回归问题，Yang等人提出了一种Transformer网络Transpose，它仅从图像中估计2D姿态。Lin等人在其方法METRO（Mesh Transformer）中将神经网络与Transformer网络相结合，从单个图像重建3D姿态和网格顶点。与我们的方法不同，METRO属于直接估算的范畴。此外，METRO中忽略了时间一致性，这限制了其鲁棒性。我们的时空Transformer架构利用每个帧中的关键点相关性，并保持视频中的自然时间一致性。

# 本文方法

每个帧的2D姿态由现成的2D姿态检测器获得，然后连续帧的2D姿态序列被用作估计中心帧的3D姿态的输入。与以前基于神经网络的最先进模型相比，本文生产了一个极具竞争力的无卷积Transformer网络。

## 时间Transformer基线

作为Transformer在2D-to-3D提升中的基线应用，作者将每个2D姿态视为输入token，并使用Transformer捕获输入之间的全局相关性，如图2a所示。

![](https://user-images.githubusercontent.com/56388518/221148850-fd5c5458-0a1b-47b9-ae77-7ce82ecb40a3.png)

<center>图2. 时间和空间Transformer架构</center>

作者将每个输入token称为patch，在术语上类似于ViT。对于输入序列${\bf X}\in\mathbb{R}^{f\times (J\cdot2)}$，$f$是输入序列的帧数，$J$是每个2D姿态的关节数，2表示关节坐标在2D空间中。$\{ {\bf x}^i\in\mathbb{R}^{1\times(J\cdot2)}\,|\,i=1,2,...,f\}$表示每帧的输入向量。Patch Embedding是一个可训练的线性投影层，用来将每个patch嵌入到一个高维的特征中。Transformer网络利用位置嵌入来保留序列的位置信息，该过程用公式表示如下：
$$
Z_0=[{\bf x}^1E;,{\bf x}^2E;...;{\bf x}^fE]+E_{pos}
$$
通过线性投影矩阵$E\in\mathbb{R}^{(J\cdot2)\times C}$嵌入，再与位置嵌入$E_{pos}\in\mathbb{R}^{f\times C}$求和之后，输入序列$X\in\mathbb{R}^{f\times(J\cdot2)}$变为了$Z_0\in\mathbb{R}^{f\times C}$，其中$C$是嵌入维度。$Z_0$将被送入时间Transformer编码器。

作为Transformer的核心功能，自注意力机制将输入序列的不同位置与嵌入特征关联起来。该Transformer的编码器由多头自注意力块和多层感知机（MLP）块组成。每个块之前都应用了层归一化（Layer Normalization, LN），每个块之后应用了残差连接。

### 缩放点积注意力

缩放点积注意力（Scaled Dot-Product Attention）可以描述为将查询矩阵$Q$，键矩阵$K$和值矩阵$V$映射到输出注意力矩阵的一个映射函数。$Q,K,V\in\mathbb{R}^{N\times d}$，其中$N$是序列中向量的数量，$d$是维数。在该注意力操作中使用比例因子$\frac{1}{\sqrt{d}}$进行适当的归一化，以防止较大的$d$导致的点积巨大增长时出现的极其小的梯度。缩放点积注意力的输出可以表示为：
$$
\text{Attention}(Q,K,V)=\text{Softmax}(\frac{QK^{\top}}{\sqrt{d}})V
$$
在本文的时间Transformer中，$d=C,N=f$。矩阵$Q,K,V$由嵌入特征矩阵$Z\in\mathbb{R}^{f\times C}$经过线性变换$W_Q,W_K,W_V\in\mathbb{R}^{C\times C}$得到：
$$
Q=ZW_Q,\quad K=ZW_K,\quad V=ZW_V
$$

### 多头自注意力层

多头自注意力层（Multi-head Self Attention Layer, MSA）利用多个头部从不同位置的不同表示子空间对信息进行联合建模，每个头部并行的使用缩放点积注意力操作。该MSA层的输出由$h$个头部的输出进行连结操作得出：
$$
\text{MSA}(Q,K,V)=\text{Concat}(H_1,H_2,...,H_h)W_{out}\quad \text{where}\quad H_i=\text{Attention}(Q_i,K_i,V_i),i\in[1,2,...,h]
$$
回到对时间Transformer编码器的讨论：

对于给定的嵌入特征矩阵$Z_0\in\mathbb{R}^{f\times C}$，本文的L个时间Transformer编码器层的结构可表示为：
$$
\begin{array}{l}
Z'_l=\text{MSA}(\text{LN}(Z_{l-1}))+Z_{l-1},\quad l=1,2,...,L\\
Z_l=\text{MLP}(\text{LN}(Z'_l))+Z'_l,\quad l=1,2,...,L\\
Y=\text{LN}(Z_L)
\end{array}
$$
其中，$\text{LN}(\cdot)$表示层归一化操作（与ViT中的相同），时间Transformer编码器由L个相同的层组成，编码器的输出$Y\in\mathbb{R}^{f\times C}$与输入$Z_0\in\mathbb{R}^{f\times C}$保持一致。

为了预测中心帧的3D姿态，通过在帧维度上取平均值，将编码器输出$Y\in\mathbb{R}^{f\times C}$收缩为向量$Y\in\mathbb{R}^{1\times C}$。最后，MLP块将输出回归到$Y\in\mathbb{R}^{1\times(J\cdot3)}$，即中心帧的3D姿态。

## PoseFormer：时空Transformer

可以观察到，时间Transformer基线主要关注序列中帧之间的全局相关性。patch嵌入是一种线性变换，用于将关节坐标投影到隐藏维度。然而简单的线性投影层不能学习注意力信息，因此局部关节坐标之间的运动信息在时间Transformer基线中没有强力的表示。一种可能的解决方案是将每个关节坐标视为一个单独的patch，并将所有帧中的关节作为Transformer的输入（图1b）。

<img src="https://user-images.githubusercontent.com/56388518/221148757-f7af474a-4e50-483a-9718-0eca0b5a48c2.png" alt="" style="zoom: 33%;" />

<center>图1. 时间Transformer基线及其扩展</center>

然而patch的数量会迅速增加（$f\times J$），导致模型计算复杂度为$O((f\cdot J)^2)$。为了有效地学习局部关节相关性，作者分别对空间和时间信息使用了两个独立的Transformer。如图2b所示，PoseFormer由三个模块组成：空间Transformer、时间Transformer和回归头模块。

![](https://user-images.githubusercontent.com/56388518/221148850-fd5c5458-0a1b-47b9-ae77-7ce82ecb40a3.png)

### 空间Transformer模块

空间Transformer模块用于从单帧中提取高维特征嵌入。给定具有$J$个关节的2D姿态，将每个关节视作一个patch，并利用通用视觉Transformer流水线在所有patch上执行特征提取。

首先，使用可训练的线性投影将每个关节坐标映射到高维，该过程称为空间patch嵌入。然后将空间patch嵌入与可学习的空间位置嵌入$E_{SPos}\in\mathbb{R}^{J\times c}$相加，因此第i帧的输入${\bf x}_i\in\mathbb{R}^{1\times(J\cdot2)}$变为$z_0^i\in\mathbb{R}^{J\times c}$，其中$c$为空间嵌入维度。得到的特征$z_0^i$的关节序列被送入空间Transformer编码器，该编码器利用自注意力机制来整合所有关节上的信息。对于第$i$帧，具有L层的空间Transformer编码器的输出应为$z_L^i\in\mathbb{R}^{J\times c}$。

### 时间Transformer模块

由于空间Transformer模块对每个单帧的高维特征进行编码，所以时间Transformer模块的目标是对帧序列中的相关性进行建模。对于第$i$帧，空间Transformer的输出$z_L^i\in\mathbb{R}^{J\times c}$被展平为一个向量${\bf z}_i\in\mathbb{R}^{1\times(J\cdot c)}$。然后将来自$f$个输入帧的向量$\{ {\bf z}_1,{\bf z}_2,...,{\bf z}^f\}$连结为$Z_0\in\mathbb{R}^{f\times(J\cdot c)}$。在时间Transformer模块之前添加了可学习的时间位置嵌入$E_{TPos}\in\mathbb{R}^{f\times(J\cdot c)}$来保留帧位置信息。时间Transformer编码器使用了与空间Transformer编码器相同的架构，由多头自注意力块和MLP块组成。时间Transformer模块的输出为$Y\in\mathbb{R}^{f\times(J\cdot c)}$。

### 回归头

由于使用了一个帧序列来预测中心帧的3D姿态，时间Transformer模块的输出$Y\in\mathbb{R}^{f\times(J\cdot c)}$需要减少到${\bf y}\in\mathbb{R}^{1\times(J\cdot 2)}$，在帧维度上使用加权平均（习得的权重）运算来实现。最后，由具有层归一化和一个线性层的MLP返回输出${\bf y}\in\mathbb{R}^{1\times(J\cdot 3)}$，即中心帧的3D姿态。

### 损失函数

变为使用标准MPJPE损失来最小化预测姿态和真实姿态之间的误差：
$$
\mathcal{L}=\frac{1}{J}\sum_{k=1}^J||p_k-\hat{p}_k||_2
$$
其中，$p_k$和$\hat{p}_k$分别为第$k$个关节的真实位置和预测的位置。

# 实验

## 数据集和评价指标

### Human3.6M

本文在S1，S5，S6，S7，S8上进行训练，在S9，S11上进行测试。

使用MPJPE（协议1）和PMPJPE（协议2）来评估模型。

### MPI-INF-3DHP

在训练集中所有的8个演员上进行训练，使用独立的MPI-INF-3DHP测试集上进行测试。

使用MPJPE、阈值为150mm的PCK、曲线下面积（AUC）来评估模型。

## 实现细节

作者基于Pytorch实现本文的方法。用两个NVIDIA RTX 3090 GPU进行训练和测试。实验过程中选择了3中不同的帧序列长度，即$f=9,f=27,f=81$。在训练和测试中采用姿态水平翻转作为数据增强。采用Adam优化器对模型进行了130个epoch的训练，权重衰减为0.1。学习率采用指数衰减策略，初始学习率为2e-4，每个epoch的衰减因子为0.98。批量大小为1024，并采用随即深度的Transformer编码器层，丢弃率为0.1。

> 随机深度（Stochastic depth）
>
> 在训练过程中随机去掉某些层，并不影响算法的收敛性。这种方法可以解决深度网络的训练时间难题，并能够改善模型的精度。

对于2D姿态，本文在Human3.6M上使用级联金字塔网络（CPN）作为检测器，在MPI-INF-3DHP上使用真实2D姿态。

## 与SOTA对比

### Human3.6M

![](https://user-images.githubusercontent.com/56388518/221149078-ebbdf393-8cd6-40cd-a087-053dccef58ef.png)

<center>表1. Human3.6M上的定量结果</center>

表1展示了测试集（S9，S11）所有的15个动作上的结果。表中协议1和协议2分别采用MPJPE和PMPJPE作为评价指标。$f$表示方法中输入序列的帧数；$*$表示2D姿态是由CPN检测得到的；$\dagger$表示该方法基于Transformer。红色数字表示最优结果，蓝色表示次优结果。

在协议1和协议2下，PoseFormer分别以大幅度（6.1%和6.4%）优于本文之前提到的的基线，清楚地证明了使用空间Transformer对每个帧中关节之间的相关性进行表达性建模的优点。PoseFormer可以在拍照、坐下、遛狗和抽烟等困难动作上实现更准确的姿态预测。与其他简单动作不同，这些动作中的姿态变化更快，一些长距离帧具有很强的相关性。在这种情况下，全局依赖性起着重要的作用，Transformer的注意力机制尤其有利。

为了研究本文方法的下界，作者直接使用真实2D姿态作为输入，以减轻由带噪声的2D姿态数据引起的误差，结果如表2所示。

![](https://user-images.githubusercontent.com/56388518/221149171-e4280045-55aa-479f-a3ee-3231ebc675dd.png)

<center>表2. 使用真实2D姿态为输入的结果</center>

<img src="https://user-images.githubusercontent.com/56388518/221149268-d541e5a9-581c-4ca0-a3c4-e4e77b283ec7.png" alt="" style="zoom: 33%;" />

<center>图3. 单个关节的MPJPE对比</center>

在图3中比较了一些单个关节的MPJPE，这些关节在Human3.6M测试集S11的拍照动作上具有最大的误差。PoseFormer在这些困难关节上的性能优于其他两个模型。

### MPI-INF-3DHP

表3展示了在MPI-INF-3DHP数据集上的定量结果。由于该数据集的序列长度通常较短，故本文使用9帧的2D姿态作为模型输入。

<img src="https://user-images.githubusercontent.com/56388518/221149360-3fd81f7a-d6a8-4dee-8884-182d9a665aae.png" alt="" style="zoom:50%;" />

<center>表3. MPI-INF-3DHP上的定量结果</center>

### 定性结果

图4展示了Human3.6M的测试集S11中的拍照动作，这是最具挑战性的动作之一。

![](https://user-images.githubusercontent.com/56388518/221149502-724667e2-155c-416d-94dd-d113b39d6901.png)

<center>图4. 定性结果</center>

## 消融实验

消融实验基于Human3.6M的协议1。

### PoseFormer的设计

本文研究了空间Transformer以及空间和时间Transformer中位置嵌入的影响。

以9帧CPN检测到的2D姿态（$J=17$）作为输入来预测3D姿态。所有架构参数都是固定的，以便公平地比较每个模块的影响。空间Transformer的嵌入维度为$17\times 32=544$，空间Transformer编码器层数为4；时间Transformer的嵌入维度与空间Transformer一致（544），使用了4个时间Transformer层。

表4展示了与本文提到的基线模型的比较，即时空设计的影响。

<img src="https://user-images.githubusercontent.com/56388518/221149583-92e4797b-177c-4494-a7da-79433ec6af40.png" alt="" style="zoom:50%;" />

<center>表4. 消融实验</center>

表4结果表明，本文的时空Transformer产生了显著的影响，因为对关节之间的相关性进行了强有力的建模。当$f=81$时与表1的结果一致。

此外还评估位置嵌入的影响。本文探讨了四种可能的组合：无位置嵌入、仅空间位置嵌入、仅时间位置嵌入、空间和时间位置嵌入。比较这些组合的结果，很明显，位置嵌入提高了性能。通过将这些应用于空间和时间模块，可以获得最佳的总体结果。

### 架构参数分析

本文探索了各种参数组合，以寻求表5中的最佳网络架构。

<img src="https://user-images.githubusercontent.com/56388518/221149654-1342a75a-380b-4fc6-bbbc-dee621a9ca00.png" alt="" style="zoom:50%;" />

<center>表5. 架构参数分析</center>

$c$表示空间Transformer中的嵌入特征维度，$L$表示Transformer编码器中使用了多少层。在PoseFormer中，空间Transformer的输出被平坦化，并与时间位置嵌入相加，以形成时间Transformer编码器的输入。因此，时间Transformer编码器中的嵌入特征维数为$c\times J$。本文模型的最佳参数为$c＝32,L_S＝4,L_T＝4$。

### 计算复杂度分析

表6中展示了模型性能、参数总数、每帧的估计浮点运算（FLOPs）以及具有不同输入序列长度（$f$）的每秒输出帧数（FPS）。

<img src="https://user-images.githubusercontent.com/56388518/221149729-24806910-89a6-4200-a114-31c091c89b6f.png" alt="" style="zoom:50%;" />

<center>表6. 计算复杂度分析</center>

当序列长度增加时，本文的模型获得了更好的精度，并且参数总数不会增加太多。因为帧的数量只影响时间位置嵌入层，而该层不需要很多参数。尽管本文模型的推理速度不是最快的，但该速度对于实时推理来说仍然是可接受的。普通2D姿态检测器的FPS通常低于80，故本模型的推理速度不会成为瓶颈。

### 注意力可视化

在Human3.6M测试集S11中的坐下动作上可视化了空间和时间Transformer的自注意力热图，如图5、6所示。

<img src="https://user-images.githubusercontent.com/56388518/221149807-5df05ee6-09a0-4903-97a6-e4edb8cfc650.png" alt="" style="zoom:50%;" />

<center>图5. 空间Transformer自注意力</center>

对于空间Transformer自注意力，x轴对应17个关节的查询，y轴表示注意力输出。注意力头返回不同的注意力强度，这表示在输入关节之间学习到的各种局部关系。例如Head 3主要集中于关节15和16，即右手肘和右手腕；而Head 5建立了关节4、5、6与11、12、13的关系，即左腿和左臂的连接。

<img src="https://user-images.githubusercontent.com/56388518/221149823-bccfdff2-6c93-4867-adc7-a4400c762bc4.png" alt="" style="zoom:50%;" />

<center>图6. 时间Transformer自注意力</center>

而对于时间Transformer自注意力，x轴对应81帧的查询，y轴表示注意力输出。长期的全局依赖性由不同的注意力头学习。例如Head 3的注意力与中心帧右侧的某些帧高度相关，即帧58、62、69等。

空间和时间注意力热图表明，PoseFormer成功地建模了关节之间的局部关系，并捕获了整个输入序列的长期全局相关性。

### 针对小数据集的泛化性

作者用本文的模型进行了一项实验，以研究小数据集HumanEva上Transformer的学习能力。表7展示了从头开始训练的结果，以及微调Human3.6M上的预训练模型的结果。

<img src="https://user-images.githubusercontent.com/56388518/221149927-b50829cd-294d-4e24-a01d-6e96bac7c2a3.png" alt="" style="zoom:50%;" />

<center>表7. 小数据集泛化性</center>

微调时，性能可以大幅提高，Transformer在大规模数据集上预训练时可以表现良好。

