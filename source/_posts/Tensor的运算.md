---
title: Tensor的运算
tags:
  - 机器学习
  - Pytorch
categories: 机器学习
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fmod.3dmgame.com%2Fstatic%2Fupload%2Fmod%2F202010%2FMOD5f83a16d23609.jpeg&refer=http%3A%2F%2Fmod.3dmgame.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1667106682&t=843feab65e68dd2aeef7250e5c6c8e65
abbrlink: 7822
date: 2022-09-30 13:09:30
---

# Tensor的算术运算

## 基本运算

四种基本运算都要求参与运算的两个Tensor形状相同

```python
a = torch.Tensor([[1, 2, 3], [1, 2, 3]])
b = torch.Tensor([[3, 2, 1], [3, 2, 1]])
```

### 加法

```python
print(a + b)
print(torch.add(a, b))
print(a.add(b))
print(a.add_(b))
```

> 最后一种in-place函数的方式同之前提到的`uniform_()`函数类似，会将a的值改变为做完加法后的结果 

### 减法

```python
print(a - b)
print(torch.sub(a, b))
print(a.sub(b))
print(a.sub_(b))
```

### 乘法

这里的乘法结果是哈达玛积，即对应元素相乘

```python
print(a * b)
print(torch.mul(a, b))
print(a.mul(b))
print(a.mul_(b))
```

### 除法

```python
print(a / b)
print(torch.div(a, b))
print(a.div(b))
print(a.div_(b))
```

## 矩阵运算

### 矩阵乘法

```python
a = torch.eye(2, 3)
b = torch.eye(3, 2)
```

```python
print(a @ b)
print(torch.matmul(a, b))
print(torch.mm(a, b))
print(a.matmul(b))
print(a.mm(b))
```

对于高维的Tensor，只要求最后两个维度满足矩阵乘法的维度要求

```python
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
```

```python
print(a @ b)
```

### 幂运算

这里并不是矩阵的幂运算，而是对每个元素进行幂运算

```python
a = torch.tensor([1, 2])
```

```python
print(a ** 3)
print(torch.pow(a, 3))
print(a.pow(3))
print(a.pow_(3))
```

对于指数为e的幂运算，有单独的函数可以实现

```python
a = torch.tensor([1, 2], dtype=torch.float32)
```

类型定义为float是因为`exp_()`函数不支持long类型的Tensor

```python
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())
```

### 对数运算

对每个元素进行对数运算

 ```python
a = torch.tensor([10, 2], dtype=torch.float32)
 ```

`log_()`、`log2_()`、`log10_()`均需要float类型的Tensor

```python
# 以e为底的对数
print(torch.log(a))
print(torch.log_(a))
# 以2为底的对数
print(torch.log2(a))
print(torch.log2_(a))
# 以10为底的对数
print(torch.log10(a))
print(torch.log10_(a))
```

### 开方运算

对每个元素进行开方

```py
a = torch.tensor([1, 2], dtype=torch.float32)
```

```python
print(torch.sqrt(a))
print(torch.sqrt_(a))
print(a.sqrt())
print(a.sqrt_())
```

# in-place与广播机制

## in-place操作

在之前的讨论中已经见过很多in-place函数了，in-place操作——“就地”操作，即不允许使用临时变量，也称原位操作。

即满足`x = x + y`的操作，如上述的`add_()`、`sub_()`等。

## 广播机制

广播机制：张量参数可以自动扩展为相同的大小

广播机制需要满足两个条件

- 每个张量至少有一个维度
- 满足右对齐

> 判断是否满足右对齐需要从右往左看两个张量的维度
>
> 若两个维度的值相等或其中有一个为1，则认为两个张量满足右对齐；若遇到维数不相同的情况，则会在维数较少的张量之前补1。
>
> 例如两个维度分别为(2,1,1)的张量a和(3)的张量b：
>
> - 张量b的维度会被补齐为(1,1,3)；
> - 从右往左看，1和3对应，其中有一个为1，故第三个维度是对齐的；又因为张量b的前两个维度是补齐为1的，必然会对齐，故a和b满足右对齐

```python
a = torch.rand(2, 1, 1)
b = torch.rand(3)
c = a + b
print(c.shape)
```

```
torch.Size([2, 1, 3])
```

# Tensor的其他运算

## 取整、取余运算

```python
a = torch.rand(2, 2) * 10
```

向下取整

```python
print(torch.floor(a))
print(torch.floor_(a))
```

向上取整

```python
print(torch.ceil(a))
print(torch.ceil_(a))
```

四舍五入取整

```python
print(torch.round(a))
print(torch.round_(a))
```

裁剪，只取整数部分

```python
print(torch.trunc(a))
print(torch.trunc_(a))
```

裁剪，只取小数部分

```python
print(torch.frac(a))
print(torch.frac_(a))
```

取余

```python
print(a % 2)
```

## Tensor的比较运算

```python
# 比较运算
a = torch.ones(2, 2)
b = torch.eye(2, 2)
# 对应位置的元素进行比较，返回一个由布尔类型构成的Tensor
print(torch.eq(a, b))
# 比较整个Tensor，若所有对应元素相同则返回Ture，否则返回False
print(torch.equal(a, b))
# 以下函数均返回一个由布尔类型构成的Tensor
# a > b
print(torch.gt(a, b))
# a >= b
print(torch.ge(a, b))
# a < b
print(torch.lt(a, b))
# a <= b
print(torch.le(a, b))
# a != b
print(torch.ne(a, b))
```

## Tensor的排序

```python
a = torch.tensor([[3, 9, 7, 4, 5],
                  [1, 4, 8, 5, 2]])
print(torch.sort(a))
```

默认状态下是升序排列，且会将每个维度都排序

```
torch.return_types.sort(
values=tensor([[3, 4, 5, 7, 9],
        [1, 2, 4, 5, 8]]),
indices=tensor([[0, 3, 4, 2, 1],
        [0, 4, 1, 3, 2]]))
```

`values`为排序后的Tensor，`indices`为排序后的元素在原Tensor中对应的下标

还可以指定`descending`参数为True实现降序排列

```python
print(torch.sort(a, descending=True))
```

```
torch.return_types.sort(
values=tensor([[9, 7, 5, 4, 3],
        [8, 5, 4, 2, 1]]),
indices=tensor([[1, 2, 4, 3, 0],
        [2, 3, 1, 4, 0]]))
```

还可以指定参与排序的维度

```python
print(torch.sort(a, dim=0))
```

由于我们定义的例子Tensor是`(2,5)`的，当`dim=0`时会在2这个维度上排序

```
torch.return_types.sort(
values=tensor([[1, 4, 7, 4, 2],
        [3, 9, 8, 5, 5]]),
indices=tensor([[1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]]))
```

## 返回最大的k个元素

```python
a = torch.tensor([[2, 4, 3, 1, 5],
                  [2, 3, 5, 1, 4]])
print(torch.topk(a, k=2, dim=1))
```

参数`k`指定返回降序排列的前k个元素，`dim`指定操作的维度

```
torch.return_types.topk(
values=tensor([[5, 4],
        [5, 4]]),
indices=tensor([[4, 1],
        [2, 4]]))
```

## 返回指定维度的升序排列的第k个元素

```python
print(torch.kthvalue(a, k=2, dim=1))
```

```
torch.return_types.kthvalue(
values=tensor([2, 2]),
indices=tensor([0, 0]))
```

# Tensor数据的合法性校验

```python
a = torch.rand(2, 3)
# 以下函数均返回一个由布尔类型构成的Tensor
# 元素为有界的则返回True，否则返回False
print(torch.isfinite(a / 0))
# 元素为无界的则返回True，否则返回False
print(torch.isinf(a / 0))
# 元素为非数值则返回True，否则返回False
print(torch.isnan(a))
```

```
tensor([[False, False, False],
        [False, False, False]])
tensor([[True, True, True],
        [True, True, True]])
tensor([[False, False, False],
        [False, False, False]])
```

`nan`并不能通过pytorch定义出来，这里借助numpy定义一个含有`nan`元素的Tensor

```python
import numpy as np

a = torch.tensor([1, 2, np.nan])
print(torch.isnan(a))
```

```
tensor([False, False,  True])
```

