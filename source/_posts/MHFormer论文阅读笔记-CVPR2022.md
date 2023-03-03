---
title: MHFormer论文阅读笔记-CVPR2022
toc: true
mathjax: true
tags:
  - 论文
  - CVPR
  - 3D人体姿态估计
categories: 文献阅读笔记
abbrlink: 16725
date: 2023-03-03 10:43:40
---

文章题目为：MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation。文章提出了一个基于Transformer的多假设模型，不同于之前单纯的一对多映射，MHFormer在自假设之间进行通信以细化估计，并最终生成一个可信的3D姿态。

<!-- more -->

# 大纲

<img src="https://user-images.githubusercontent.com/56388518/222618685-5c6d0c86-9113-4136-84d8-148650fd4322.png" alt="" style="zoom:50%;" />

# 引言

从单目视频中进行3D人体姿态估计是一项具有广泛应用的基本视觉任务。该任务通常通过将其划分为两个分离的子任务来解决，即利用2D姿态检测以定位图像上的关键点，然后利用2D-to-3D提升从2D关键点推断3D空间中的关节位置。由于2D表示中的自遮挡和深度歧义，它仍然是一个固有的不适定问题。

> 不适定问题就是不满足适定性条件的问题，适定性条件为：
>
> - 有且只有一个解（存在性、唯一性）；
> - 解连续地依赖于数据，即$x_n\to x,Kx_n\to Kx$（稳定性）。

为了缓解这些问题，大多数方法侧重于探索空间和时间关系。要么使用图卷积网络来估计具有人类骨架的时空图表示的3D姿态，要么应用纯基于Transformer的模型来从2D姿态序列中捕获空间和时间信息。然而，从单目视频中2D-to-3D是一个反问题，其中存在多个可行的解决方案（即假设），因为其在给定缺失深度的情况下具有不适定性。这些方法忽略了这个问题，只估计了一个单一的解决方案，这通常会导致不满意的结果，尤其是当目标产生严重自遮挡的情况下，如图1所示。

<img src="https://user-images.githubusercontent.com/56388518/222618746-74f05205-008f-4dde-b789-c5409aee5fe2.png" alt="" style="zoom: 50%;" />

<center>图1. 多假设概念</center>

最近，针对该反问题提出了两种生成多个假设的方法。这些方法通常依赖于一对多映射，通过向具有共享特征提取器的现有架构中添加多个输出头来实现，但无法建立不同假设特征之间的关系。这是一个重大的缺点，因为这种能力对于提高模型的表现和性能至关重要。鉴于3D HPE是具有歧义的反问题，作者认为，首先进行一对多映射，然后使用各种中间假设进行多对一映射更合理，这种方法可以丰富特征的多样性，并生成更好的最终3D姿态。

![](https://user-images.githubusercontent.com/56388518/222618792-76e58979-a7b0-49b1-8f39-02c0ab46f5ac.png)

<center>图2. 三阶段模型</center>

本文提出了多假设Transformer（MHFormer），模型的关键是使模型从不同的姿态假设中学习时空表示。为此，本文构建了一个三阶段框架，首先生成多个初始表示，然后以独立和相互的方式在多假设之间进行通信，以合成更精确的估计，如图2所示。

接下来，本文提出了两个新的模块来建模时间一致性，并增强时间域中的粗糙表示。在第二阶段，提出了自我假设细化（Self-Hypothesis Refinement, SHR）模块来细化每个假设特征。SHR由两个块组成。第一个模块是多假设自我关注（Multi-Hypothesis Self-Attention, MH-SA），它独立地对单个假设依赖性进行建模，以构建自假设通信，使信息能够在每个假设内传递以增强特征。第二块是一个假设混合多层感知器（Hypothesis-Mixing MLP），它跨假设交换信息。将多个假设合并为单个收敛表示，然后将该表示划分为几个发散假设。

尽管SHR对这些假设进行了细化，但由于SHR中的MH-SA仅传递假设内信息，因此不同假设之间的联系不够强。为了解决这个问题，在最后阶段，交叉假设交互（Cross-Hypothesis Interaction, CHI）模块对多假设特征之间的交互进行建模。它的关键组成部分是多假设交叉关注（Multi-Hypothesis Cross-Attention, MH-CA），它捕获相互的多假设相关性，以建立跨假设通信，从而实现假设之间的信息传递，从而更好地进行交互建模。随后，使用假设混合多层感知机（Hypothesis-Mixing MLP）来聚合多个假设以合成最终预测。

利用所提出的MHFormer，多假设时空特征层次被明确地纳入Transformer模型，其中身体关节的多假设信息能够以端到端的方式被独立和交叉的处理。潜在地增强了表示能力，并且合成的姿态更加准确。

本文的贡献如下：

- 本文提出了一种新的基于Transformer的用于单目视频3D HPE的方法，称为多假设Transformer（MHFormer）。MHFormer可以以端到端的方式有效地学习多个姿态假设的时空表示。
- 提出了在多假设特征之间独立和相互的交流，提供强大的自假设和跨假设信息传递，以加强假设之间的关系。
- MHFormer在3D HPE的两个具有挑战性的数据集上实现了最先进的性能，超过PoseFormer约3%。

# 相关工作

### 3D人体姿态估计

现有的单视图3D姿态估计方法可分为两种主流类型：一阶段方法和两阶段方法。一阶段方法直接从输入图像中推断3D姿态，而不需要中间的2D姿态表示，而两阶段方法首先从预训练的2D姿态检测中获得2D关键点，然后将其送入到2D-to-3D网络中以估计3D姿态。受益于2D人体姿态估计的优异性能，这种2D-to-3D的方法可以使用检测到的2D关键点来高效且准确地回归3D姿态。尽管通过使用全卷积或基于图的架构获取时间相关性有良好的结果，但这些方法在跨帧捕获全局上下文信息方面效率较低。

### 视觉Transformer

对于基本图像分类任务，ViT提出将标准Transformer架构直接应用于图像patch序列。对于姿势估计任务，PoseFormer应用纯Transformer来捕获人类关节相关性和时间相关性。跨步Transformer（Strided Transformer）引入了一种基于Transformer的架构，该架构具有跨步卷积，以将长2D姿态序列提升为单个3D姿态。

本文的工作受到了它们的启发，同样使用Transformer作为基本架构。但不只是利用具有单一表示的简单架构；相反，多假设和多层次特征层次结构的开创性思想在Transformer中被联系起来，这使得模型不仅具有表现力，而且强大。此外，为有效的多假设学习引入了交叉注意机制。

### 多假设方法

单视图3D HPE是不适定的，因此只有一个假设的解决方案可能是次优的。一些工作为该反问题生成了不同的假设，并实现了实质性的性能增益。与这些专注于一对多映射的工作不同，本文方法首先学习一对多的映射，然后学习多对一的映射，这允许对与各种假设相对应的不同特征进行有效建模，以提高表示能力。

# 多假设Transformer

MHFormer的概述如图3a所示。给定由现成的2D姿态检测器从视频中估计的连续2D姿态序列，本文的方法旨在通过充分利用多假设特征层次中的空间和时间信息来重建中心帧的3D姿态。

![](https://user-images.githubusercontent.com/56388518/222618859-e933088c-b5cb-4c75-bd16-93b25f22a822.png)

<center>图3. MHFormer架构</center>

为了实现本文提出的三阶段框架，MHFormer基于

- 三个主要模块：多假设生成（MHG）、自假设细化（SHR）和交叉假设交互（CHI）
- 两个辅助模块：时间嵌入和回归头。

## 预备工作

在这项工作中，因为Transformer在长期依赖性建模中表现良好，故作者采用了基于Transformer的架构。首先简要描述Transformer中的基本组件，包括多头自注意力（MSA）和多层感知器（MLP）。

### 多头自注意力（MSA）

在多头自注意力中，输入${\bf x}\in\mathbb{R}^{n\times d}$被线性映射到查询矩阵${\bf Q}\in\mathbb{R}^{n\times d}$、键矩阵${\bf K}\in\mathbb{R}^{n\times d}$和值矩阵${\bf V}\in\mathbb{R}^{n\times d}$，其中$n$是序列长度，$d$是维度，缩放点积注意力由下式计算：
$$
\text{Attention}(Q,K,V)=\text{Softmax}(\frac{QK^{\top}}{\sqrt{d}})V
$$
MSA将查询、键、值矩阵拆分$h$次，然后并行的执行关注，最后对$h$个注意力头的输出进行连结。

### 多层感知机（MLP）

MLP由两个线性层组成，用于非线性和特征变换。公式描述为：
$$
\text{MLP}({\bf x})=\sigma({\bf xW}_1+{\bf b}_1){\bf W}_2+{\bf b}_2
$$
其中，$\sigma$表示GELU激活函数，${\bf W}_1\in\mathbb{R}^{d\times d_m}$和${\bf W}_2\in\mathbb{R}^{d_m\times d}$分别为两个线性层的权重，${\bf b}_1\in\mathbb{R}^{d_m}$和${\bf b}_2\in\mathbb{R}^d$为偏置项。

## 多假设生成（MHG）

在空域中，通过设计一个基于Transformer的级联架构来解决该反问题，以在潜在空间的不同深度中生成多个特征。为此，引入多假设生成（MHG）来建模人类关节关系并初始化多假设表示，如图3b所示。假设MHG中有$M$个不同的假设和$L_1$个层，它以$N$个视频帧和$J$个身体关节的2D姿态序列${\bf X}\in\mathbb{R}^{N\times J\times 2}$为输入，并输出多个假设$[{\bf X}_{L_1}^1,{\bf X}_{L_1}^2,...,{\bf X}_{L_1}^M]$，其中${\bf X}_{L_1}^m\in\mathbb{R}^{(J\cdot 2)\times N}$是第$m$个假设。

更具体地说，该模块将每个帧的关节坐标$(x,y)$连结为${\bf X}\in\mathbb{R}^{(J\cdot 2)\times N}$，通过一个可学习的空间位置嵌入$E_{SPos}^m\in\mathbb{R}^{(J\cdot 2)\times N}$来保留关节的空间信息，并将嵌入的特征送入MHG的编码器中。为了鼓励梯度传播，在编码器的原始输入和输出特征之间使用跳跃残差连接（skip residual connection）。该过程可形式化为：
$$
\begin{array}{l}
{\bf X}_0^m=\text{LN}({\bf X}^m)+E_{SPos}^m,\\
{\bf X}_l^{'m}={\bf X}_{l-1}^m+\text{MSA}^m(\text{LN}(X_{l-1}^m)),\\
{\bf X}_l^{''m}={\bf X}_l^{'m}+\text{MLP}^m(\text{LN}({\bf X}_l^{'m})),\\
{\bf X}_{L_1}^m={\bf X}^m+\text{LN}({\bf X}_{L_1}^{''m})
\end{array}
$$
其中，$\text{LN}(\cdot)$是层归一化，$l\in[1,...,L_1]$是MHG层的索引，${\bf X}^1=\bar{\bf X}$，并且${\bf X}^m={\bf X}_{L_1}^{m-1}\quad(m>1)$。MHG的输出（例如${\bf X}_{L_1}^{m}$）是包含不同语义信息的多级特征。因此，这些特征可以被视为不同状态假设的初始表示，这些初始表示需要进一步增强。

## 时间嵌入

MHG有助于在空间域中生成初始多假设特征，而这些特征的能力不够强。考虑到这一限制，本文使用两个模块来构建跨假设特征的关系，并捕获时域中的时间相关性，即自假设细化（SHR）模块和交叉假设交互（CHI）模块，如图3c和3d所示。

为了利用时间信息，首先将空间域转换为时间域。为此，使用换位操作和线性嵌入将每个帧编码后的假设特征${\bf X}_{L_1}^m$嵌入到高维特征$\widetilde{\bf Z}^m\in\mathbb{R}^{N\times C}$中。其中，$C$是嵌入维度。然后利用一个可学习的时间位置嵌入$E_{TPos}^m\in\mathbb{R}^{N\times C}$来保留帧的位置信息。用公式表示为：
$$
\widetilde{\bf Z}_0^m=\widetilde{\bf Z}^m+E_{TPos}^m
$$

## 自假设细化（SHR）

在时域中，首先构建SHR以细化单个假设特征。每个SHR层由多假设自注意力（MH-SA）块和假设混合MLP块组成。

### 多假设自注意力（MH-SA）

Transformer模型的核心是MSA，通过MSA，任意两个元素都可以相互作用，从而建模长期依赖关系。相反，MH-SA旨在独立地捕获每个假设中的单假设依赖性，以进行自假设沟通。具体而言，不同假设的嵌入特征$\widetilde{\bf Z}_0^m\in\mathbb{R}^{N\times C}$被送入几个并行的MSA块中，可以表示为：
$$
\widetilde{\bf Z}_l^{'m}=\widetilde{\bf Z}_{l-1}^m+\text{MSA}^m(\text{LN}(\widetilde{\bf Z}_{l-1}^m))
$$
其中，$l\in[1,2,...,L_2]$是SHR层的索引，因此不同假设特征的信息可以以自假设的方式传递，用于特征增强。

### 假设混合多层感知机（Hypothesis-Mixing MLP）

多个假设在MH-SA中独立处理，但假设之间没有信息交换。为了处理该问题，在MH-SA后添加了一个混合假设MLP。将多个假设的特征连结起来并输入到假设混合MLP中合并（即收敛过程）。然后，将收敛的特征沿通道维度均匀地划分为不重叠的块（即发散过程），以形成精炼的假设表示。用公式描述为：
$$
\begin{array}{r}
\widetilde{\bf Z}_l'=\text{Concat}(\widetilde{\bf Z}_l^{'1},...,\widetilde{\bf Z}_l^{'M})\in\mathbb{R}^{N\times(C\cdot M)}\\
\text{Concat}(\widetilde{\bf Z}_l^{'1},...,\widetilde{\bf Z}_l^{'M})=\widetilde{\bf Z}_l'+\text{HM-MLP}(\text{LN}(\widetilde{\bf Z}_l'))
\end{array}
$$
其中，$\text{Concat}(\cdot)$是连结运算，$\text{HM-MLP}(\cdot)$为假设混合MLP函数，该函数的与Transformer中的MLP的公式化表达一致，即$\text{MLP}({\bf x})=\sigma({\bf xW}_1+{\bf b}_1){\bf W}_2+{\bf b}_2$。该过程探索了不同假设通道之间的关系。

## 交叉假设交互（CHI）

通过CHI对多假设特征之间的相互作用进行建模，CHI包含两个模块：多假设交叉注意力（MH-CA）和假设混合MLP。

### 多假设交叉注意力（MH-CA）

MH-SA缺乏跨假设的联系，限制了其交互建模的能力。为了捕获多假设之间的关系以进行跨假设通信，提出了由多个多头交叉注意力（MCA）元素并行组成的MH-CA。

MCA度量交叉假设特征之间的相关性，具有与MSA相似的结构。MCA的常见配置是在键和值之间使用相同的输入。然而，这种配置将导致需要更多的块（例如，3个假设需要6个MCA块）。而本文采用了一种更有效的策略，通过使用不同的输入（只需要3个MCA块）来减少参数量，如图4右半部分所示。

<img src="https://user-images.githubusercontent.com/56388518/222618928-4a199dac-442b-4efb-a277-6225b76823c3.png" alt="" style="zoom:50%;" />

<center>图4. MSA与MCA对比</center>

多个假设${\bf Z}^m$被交替地视为查询、键和值，并被送入MH-CA：
$$
{\bf Z}_l^{'m}={\bf Z}_{l-1}^m+\text{MCA}^m(\text{LN}({\bf Z}_{l-1}^{m_1}),\text{LN}({\bf Z}_{l-1}^{m_2}),\text{LN}({\bf Z}_{l-1}^m))
$$
其中，$l\in[1,2,...,L_3]$是CHI层的索引，${\bf Z}_0^m=\widetilde{\bf Z}_{L_2}^m$，而$m_1$和$m_2$是其他两个相应的假设，$\text{MCA}(Q,K,V)$表示MCA函数。由于MH-CA，可以以交叉的方式执行消息传递，从而显著提高了模型的建模能力。

### 假设混合多层感知机（Hypothesis-Mixing MLP）

CHI中的HM-MLP与SHR中的具有相同的功能。MH-CA的输出被送入该HM-MLP中：
$$
\begin{array}{r}
{\bf Z}_l'=\text{Concat}({\bf Z}_l^{'1},...,{\bf Z}_l^{'M})\in\mathbb{R}^{N\times(C\cdot M)}\\
\text{Concat}({\bf Z}_l^{'1},...,{\bf Z}_l^{'M})={\bf Z}_l'+\text{HM-MLP}(\text{LN}({\bf Z}_l'))
\end{array}
$$
在最后一个CHI层的HM-MLP中，不进行分离操作，从而最终聚合所有假设的特征以合成单个假设表示${\bf Z}_{L_3}\in\mathbb{R}^{N\times(C\cdot M)}$。

## 回归头

在回归头中，在输出${\bf Z}_{L_3}$上使用线性变换层来执行回归以产生3D姿态序列$\widetilde{\bf X}\in\mathbb{R}^{N\times J\times 3}$。最终，从$\widetilde{\bf X}$中选择中心帧的3D姿态作为最终预测$\hat{\bf X}\in\mathbb{R}^{J\times 3}$。

## 损失函数

整个模型以端到端的方式进行训练，采用均方误差（MSE）损失来最小化估计和真实姿态之间的误差：
$$
\mathcal{L}=\sum_{n=1}^N\sum_{i=1}^J||{\bf Y}_i^n-\widetilde{\bf X}_i^n ||_2
$$
其中，$\widetilde{\bf X}_i^n$和${\bf Y}_i^n$分别表示第$n$帧关节$i$的预测值和真实值。

# 实验

## 数据集和评价指标

### Human3.6M

与之前工作相同，在五个受试者（S1，S5，S6，S7，S8）上训练单个模型，并在两个受试对象（S9和S11）上进行测试。采用协议1的MPJPE和协议2的P-MPJPE。

### MPI-INF-3DHP

在训练集中所有的8个演员上进行训练，使用独立的MPI-INF-3DHP测试集上进行测试。

使用MPJPE、阈值为150mm的PCK、曲线下面积（AUC）来评估模型。

## 实现细节

本文对MHFormer的实现中，包含$L_1=4$个MHG层，$L_2=2$个SHR层，以及$L_3=1$个CHI层。模型基于Pytorch实现，在RTX 3090上训练。使用Amsgrad优化器以端到端的方式从头训练模型。初始学习率为0.001，每个周期的衰减因子为0.95。使用了水平翻转姿态作为数据增强。使用级联金字塔网络（CPN）对Human3.6M进行2D姿态检测，在MPI-INF-3DHP中使用真实2D姿态。

## 与SOTA比较

### Human3.6M上的结果

表1展示了MHFormer的结果，该模型具有351帧的感受野。

![](https://user-images.githubusercontent.com/56388518/222619016-154e6c86-e1a8-4842-bf84-614a3f361929.png)

<center>表1. Human3.6M的结果</center>

其中，表中上半部分使用2D姿态检测器的输出作为输入，下半部分使用2D真实姿态作为输入，以探索模型的上界。$(\dagger)$表示使用了时间信息，粗体表示最优结果，下划线表示次优结果。

图5展示了在一些具有挑战性的姿势上与PoseFormer和基线模型（与ViT相同的架构）的定性比较。

![](https://user-images.githubusercontent.com/56388518/222619063-9517e58d-6f48-442c-82ad-4c5c65b8514c.png)

<center>图5. 与PoseFormer对比</center>

此外，表2中展示了与之前采用多假设的方法比较的结果。由于其他方法采用一对多映射，故这些方法都报告了最佳假设的评价指标，而本文的方法通过学习确定映射来得出评价指标，这在现实中更加使用。且本文的假设数更少。

<img src="https://user-images.githubusercontent.com/56388518/222619105-6bcf98cc-7708-4396-898e-b0516df20349.png" alt="" style="zoom:50%;" />

<center>表2. 与其他多假设方法比较</center>

其中，$M$为方法使用的假设数量，粗体表示最优结果，下划线表示次优结果。

### MPI-INF-3DHP上的结果

使用9帧的2D姿势序列作为该模型输入，因为与Human3.6M相比，该数据集的样本更少，序列长度更短。表3中的结果表明，本文的方法在所有指标（PCK、AUC和MPJPE）上都达到了最佳性能。它强调了MHFormer在户外场景中提高性能的有效性。

<img src="https://user-images.githubusercontent.com/56388518/222619105-6bcf98cc-7708-4396-898e-b0516df20349.png" alt="" style="zoom:50%;" />

<center>表3. MPI-INF-3DHP的结果</center>

## 消融实验

消融实验基于Human3.6M数据集的协议1，以MPJPE为评价指标。

### 感受野的影响

对于基于视频的3D HPE任务，大的感受野对于估计精度至关重要。表4展示了本文的方法在不同输入帧下的结果。

<img src="https://user-images.githubusercontent.com/56388518/222619240-7a591215-4ca2-4337-96f5-a2587b898011.png" alt="" style="zoom:50%;" />

<center>表4. 感受野消融实验</center>

其中，CPN和GT分别表示使用CPN的输出和真实2D姿态作为输入的情况。

接下来的消融实验使用27帧的感受野进行，以平衡计算效率和性能。

### MHG中参数的影响

表5上半部分展示了不同数量MHG层的影响。实验表明，在MHG中堆叠更多的层可以稍微提高性能，但参数很少增加，但当层数大于4时，增益消失。

表5的下半部分展示了在MHG中使用不同假设数量的影响。增加假设的数量可以改善结果，但当使用3个假设表示时，性能会饱和。本文的模型与单一假设模型相比有着显著的进步，这表明利用多个姿态假设的不同表示有助于提高模型的性能，从而验证了本文的动机。

<img src="https://user-images.githubusercontent.com/56388518/222619283-a67c2f6f-4fda-49d4-9659-ce4180691031.png" alt="" style="zoom:50%;" />

<center>表5. MHG参数消融实验</center>

其中，$M$代表假设数量，$L_1$表示MHG的层数。

### SHR和CHI中参数的影响

表6展示了SHR和CHI的不同参数对该模型的性能和计算复杂性的影响。结果表明，将嵌入维数从256扩展到512可以提高性能，但使用大于512的维数不能带来进一步的改进。此外，通过堆叠更多SHR或CHI层而没有更多的增益。因此，该模型的最佳参数为$L_2=2$，$L_3=1$，$C=512$。

<img src="https://user-images.githubusercontent.com/56388518/222619324-8aa6ee17-2b20-4ee1-a528-a77a38331b39.png" alt="" style="zoom:50%;" />

<center>表6. SHR和CHI参数消融实验</center>

其中，$L_2$和$L_3$分别为SHR和CHI的层数，$C$为嵌入维度。

### 模型组件的影响

表7展示了该模型的各个组件对性能的影响。

<img src="https://user-images.githubusercontent.com/56388518/222619367-44e87fa3-f1b4-497b-bd8c-243f432c44f3.png" alt="" style="zoom:50%;" />

<center>表7. 模型组件消融实验</center>

其中，$*$表示在MHG中使用了多个并行的Transformer编码器以利用多级特征。

# 定性结果

虽然本文的方法不旨在产生多个3D姿态预测，但为了更好地观察，添加了额外的回归层，并对模型进行了微调，以可视化中间假设。几个定性结果如图6所示。可以看出，本文的方法能够生成不同的可信3D姿态解决方案，特别是对于具有深度歧义、自遮挡和2D检测器不确定性的模糊身体部位。此外，通过聚合多假设信息合成的最终3D姿态更加合理和准确。

![](https://user-images.githubusercontent.com/56388518/222619430-f427c8b0-e4b7-4d2f-9b3e-19ffe283981d.png)

<center>图6. 定性结果</center>

本文方法的一个局限性是相对较大的计算复杂性。Transformer的卓越性能是以高计算成本为代价的。
