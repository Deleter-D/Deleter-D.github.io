---
title: Pytorch常用数学与统计学函数
tags:
  - 机器学习
  - Pytorch
categories: 机器学习
cover: https://user-images.githubusercontent.com/56388518/193995325-c9392df5-b759-41c5-8772-7a69700d148e.png
abbrlink: 47260
mathjax: true
date: 2022-10-04 17:10:37
---



# Pytorch中的数学函数

## 三角函数

```python
# 余弦
torch.cos(input)
# 反余弦
torch.acos(input)
# 双曲余弦
torch.cosh(input)
# 正弦
torch.sin(input)
# 反正弦
torch.asin(input)
# 双曲正弦
torch.sinh(input)
# 正切
torch.tan(input)
# 反正切
torch.atan(input)
# input中元素代表坐标y，other中元素代表坐标x
# 返回向量(x,y)与向量(1,0)的夹角
torch.atan2(input, other)
# 双曲正切
torch.tanh(input)
```

## 其他数学函数

### `abs()`函数与`neg()`函数

`abs()`函数取张量中每个元素的绝对值

```python
torch.abs(input)
```

`neg()`函数取张量中每个元素的相反数

```python
torch.neg(input)
```

### `sign()`函数

```python
torch.sign(input)
```

- 当元素小于0返回-1；
- 当元素等于0返回0；
- 当元素大于0返回1；

即：
$$
f(x)=\cases{1, x>0\\
0,x=0\\
-1,x<0}
$$

函数图像如下：

![](https://user-images.githubusercontent.com/56388518/193826314-a1866ffc-5162-4937-a5a5-4b38ab0c78b2.jpg)

### `sigmoid()`函数

```python
torch.sigmoid(input)
```

通过sigmoid函数，可以将元素映射到0~1之间

sigmoid函数本质上是连续化的sign函数，可以解决sign函数的结果不连续而导致无法求导的问题。

- 当元素趋于负无穷时，函数\值无限趋近于0；
- 当元素趋于正无穷时，函数值无限趋近于1；

即：
$$
S(x)=\frac{1}{1+e^{-x}}
$$
函数图像如下：

![](https://user-images.githubusercontent.com/56388518/193827200-c3fd9d2f-3511-4db1-a1c3-f2efb9ece469.jpg)

### `erf()`与`erfc()`函数

```python
torch.erf(input)
```

$erf$函数是误差函数，$erfc$是互补误差函数
$$
erf(x)=\frac{2}{\sqrt{\pi}}\int_0^xe^{-t^2}dt\\
erfc(x)=1-erf(x)=\frac{2}{\sqrt{\pi}}\int_x^\infty e^{-t^2}dt
$$
函数图像如下：

红色曲线为误差函数$erf$，蓝色曲线为互补误差函数$erfc$

![](https://user-images.githubusercontent.com/56388518/193827832-a410358a-aab9-46f3-a189-1b7f5c5ec85b.jpg)

### `erfinv()`函数

```python
torch.erfinv(input)
```

$erfinv$是逆误差函数

函数图像参考$erf$函数的图像，若$erf$函数是给定$x$值求得$erf(x)$，则$erfinv$函数为给定$erf(x)$求得$x$

### `lerp()`函数

```python
torch.lerp(start, end, weight)
```

`lerp()`函数对两个张量以start、end做线性插值
$$
out_i=start_i+weight*(end_i-start_i)
$$

### `addcdiv()`函数与`addcmul()`函数

`addcdiv()`函数

```python
torch.addcdiv(input, tensor1, tensor2, value=1)
```

`addcdiv()`函数将`tensor1`与`tensor2`的商乘以`value`再加上`input`
$$
out_i=input_i+value\times\frac{tensor1_i}{tensor2_i}
$$
`addcmul()`函数

```python
torch.addcmul(input, tensor1, tensor2, value=1)
```

`addcmul()`函数将`tensor1`与`tensor2`的积乘以`value`再加上`input`
$$
out_i=input_i+value\times tensor1_i\times tensor2_i
$$

### `cumprod()`函数与`cumsum()`函数

`cumprod()`函数

```python
torch.cumprod(input, dim=0)
```

`cumprod()`函数是向量维度上的计算，`dim`参数指定参与计算的维度
$$
y_i=x_1\times x_2\times\cdots\times x_i
$$
例如：定义一个形状为(2,3)的张量，指定`cumprod()`在3这个维度上运算

```python
input = torch.Tensor([[0, 1, 2], [3, 4, 5]])
print(torch.cumprod(input, dim=1))
```

返回结果为

```
tensor([[ 0.,  0.,  0.],
        [ 3., 12., 60.]])
```

`cumsum()`函数

```python
torch.cumsum(input, dim=1)
```

与`cumprod()`函数类似，`cumsum()`函数也是向量维度上的计算，`dim`参数指定参与计算的维度
$$
y_i=x_1+x_2+\cdots+x_i
$$
同上例，指定`cumsum()`函数在3这个维度上运算

```python
input = torch.Tensor([[0, 1, 2], [3, 4, 5]])
print(torch.cumsum(input, dim=1))
```

返回结果为

```
tensor([[ 0.,  1.,  3.],
        [ 3.,  7., 12.]])
```

### `reciprocal()`函数

```python
torch.reciprocal(input)
```

`reciprocal()`函数取张量中每个元素的倒数

### `sqrt()`函数与`rsqrt()`函数

```python
torch.sqrt(input)
torch.rsqrt(input)
```

`sqrt()`函数取张量中每个元素的平方根

`rsqrt()`函数取张量中每个元素的平方根的倒数

# Pytorch中的统计学相关函数

此处以求平均值函数`mean()`为例

```python
torch.mean(input)
torch.mean(input, dim=0)
torch.mean(input, dim=0, keepdim=True)
```

若只传入一个张量，则计算该张量中所有元素的平均值；

若指定维度，则计算出该维度的每个向量中元素的平均值；

若指定了`keepdim`为`Ture`，则返回的结果张量保持与输入张量相同的维度；

## 常用函数

| 函数名     | 功能               |
| ---------- | ------------------ |
| `mean()`   | 返回平均值         |
| `sum()`    | 返回元素之和       |
| `prod()`   | 返回元素之积       |
| `max()`    | 返回最大值         |
| `min()`    | 返回最小值         |
| `argmax()` | 返回最大值的索引值 |
| `argmin()` | 返回最小值的索引值 |
| `median()` | 返回中位数         |
| `mode()`   | 返回众数           |

以上函数的参数均与上例`mean()`函数的参数大同小异

| 函数名  | 功能       |
| ------- | ---------- |
| `std()` | 返回标准差 |
| `var()` | 返回方差   |

上述两个函数除与`mean()`函数相同的参数外，还有一个`unbiased(bool)`参数，指定是否使用贝叶斯矫正

> 在统计学中，贝塞尔矫正是在样本方差和样本标准差的公式中使用 n - 1 代替 n，其中 n 是样本中的观察数。该方法纠正了总体方差估计中的偏差。它还部分纠正了总体标准偏差估计中的偏差。然而，校正通常会增加这些估计中的均方误差。

## `histc()`函数

计算输入张量的直方图

```python
torch.histc(input, bins=10, min=0, max=0)
```

- 参数`bins`指定直方图的统计区间个数；
- `min`和`max`分别指定直方图中的最小值和最大值，若这两个参数为0，则选取输入张量中的最小/大值作为直方图的最小/大值；

## `bincount()`函数

统计一维非负整型数组中每个值出现的频次，他的`bins`即区间个数比数组中的最大值大1，空数组除外

```python
torch.bincount(input)
torch.bincount(input, weight)
torch.bincount(input, weight, minlength=10)
```

`bincount()`函数只能处理一维的非负整型数组

- 参数`weight`指定数组中每个元素的权重，`weight`应当是一个与`input`形状相同的数组，指定`weight`后返回的结果将是每个元素出现的频次与权重的乘积；
- 参数`minlength`指定统计结果的`bins`的最小长度，即区间个数的最小值，没有出现数的区间将用0补齐；

## `distributions`模块

在pytorch中有`torch.distributions`这样一个模块，里面包含了很多的分布函数，例如伯努利分布、正态分布、均匀分布等，具体参考官方手册。:link:[torch.distributions](https://pytorch.org/docs/stable/distributions.html)

在随机抽样过程中，可以通过`manual_seed(seed)`函数来定义随机种子，以正太分布为例：

```python
# manual_seed的参数要求是int型
torch.manual_seed(1)
mean = torch.rand(1, 2)
std = torch.rand(1, 2)
print(torch.normal(mean, std))
```

定义随机种子后，无论执行多少次，随机产生的结果都是一样的
