---
title: Tensor的创建
tags:
  - 机器学习
  - Pytorch
categories: 机器学习
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F1657136d7d631b3e8d70f7816776b470bdcf244a.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1665912697&t=8ae6eb8b9c0873c0b3c426da22bf9432
abbrlink: 44567
date: 2022-09-16 17:29:03
---

# Tensor的创建

首先引入torch包

```python
import torch
```

## Tensor的一般定义方式

### 定义一个Tensor并直接初始化

```python
a = torch.Tensor([[1, 2], [3, 4]])
```

参数给出Tensor的每个元素的值，打印效果如下

```
tensor([[1., 2.],
        [3., 4.]])
```

类型默认为`torch.FloatTensor`

### 定义一个给定形状的Tensor

```python
a = torch.Tensor(2, 3)
```

参数为Tensor的形状，为一个2 × 3的Tensor

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### 定义一个给定尺寸或形状的全为1的Tensor

```python
a = torch.ones(2, 2)
```

参数为Tensor的形状

```
tensor([[1., 1.],
        [1., 1.]])
```

### 定义一个给定形状的对角线为1的Tensor

```python
a = torch.eye(2, 2)
```

参数为Tensor的形状

```
tensor([[1., 0.],
        [0., 1.]])
```

### 定义一个给定尺寸的全为0的Tensor

```python
a = torch.zeros(2, 2)
```

参数为Tensor的形状

```
tensor([[0., 0.],
        [0., 0.]])
```

### 通过已有的Tensor定义一个形状相同的Tensor

定义一个与给定Tensor大小相同的全0/1的Tensor

```python
template = torch.Tensor(2, 3)
a = torch.zeros_like(template)
b = torch.ones_like(template)
```

```
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

### 定义一个元素为0~1之间随机值的Tensor

```python
a = torch.rand(2, 2)
```

参数为Tensor的形状

```
tensor([[0.1935, 0.1548],
        [0.9900, 0.6763]])
```

## 其他定义方式

### `normal()`函数

`torch.normal()`函数返回一个Tensor，Tensor的元素是从单独的正态分布中提取的随机数

```python
a = torch.normal(mean=0.0, std=torch.rand(5))
b = torch.normal(mean=torch.rand(5), std=torch.rand(5))
```

参数mean为正态分布的均值，std为标准差；均值与标准差均可以是Tensor

```
tensor([ 0.1767,  1.5656,  0.2805, -0.0696, -0.6908])
tensor([1.0059, 1.1477, 0.1194, 1.3413, 0.9816])
```

### `uniform_()`函数

`uniform_()`函数需要提前定义好一个Tensor，通过Tensor对象去调用该函数；该函数使该Tensor对象在给定范围的均匀分布中采样

```python
a = torch.Tensor(2, 2).uniform_(-1, 1)
```

参数分别为from和to，指定均匀分布的范围

```
tensor([[ 0.7738, -0.6772],
        [-0.9346, -0.2675]])
```

> 类似`uniform_()`函数这样带_后缀的函数成为in-place函数，会直接改变调用它的变量

### 通过序列定义Tensor

#### `arange()`函数

```python
a = torch.arange(0, 11, 1)
b = torch.arange(0, 11, 2)
```

前两个参数构成序列范围的左闭右开区间，第三个参数为步长

```
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
tensor([ 0,  2,  4,  6,  8, 10])
```

#### `linspace()`函数

```python
a = torch.linspace(2, 10, 3)
```

前两个参数构成序列范围的左闭右闭区间，第三个参数指定Tensor的元素个数n，将返回一个元素为区间内的n个等间隔数的Tensor

```
tensor([ 2.,  6., 10.])
```

#### `randperm()`函数

```python
a = torch.randperm(10)
```

参数为n，将返回`[0,n-1]`(包括n-1)随机打乱后的数字序列为元素的Tensor

```
tensor([0, 4, 3, 1, 8, 7, 5, 6, 9, 2])
```

