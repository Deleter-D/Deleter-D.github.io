---
title: Tensor的属性及稀疏张量
tags:
  - 机器学习
  - Pytorch
categories: 机器学习
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F0f2b3ad5a078ab2946498a5c0b6ddf9c56051576.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1665912800&t=bfac89c27eafd3bea8e1826b5cac4851
abbrlink: 42093
date: 2022-09-16 17:32:22
---

# Tensor的属性

每个Tensor有三个属性

- `torch.dtype`：表示Tensor的类型
- `torch.device`：表示Tensor对象在创建后所存储在的设备名称
- `torch.layout`：表示Tensor对象的内存布局（稠密或稀疏）

```python
# dev = torch.device("cpu")
dev = torch.device("cuda")
# 通过参数指定Tensor的类型和设备
a = torch.tensor([2, 2], dtype=torch.float32, device=dev)
```

第一个参数为Tensor的元素值

```
tensor([2., 2.], device='cuda:0')
```

# 稀疏张量

定义稀疏张量需要三个参数

- 非零元素的坐标
- 非零元素的值
- Tensor的形状

其中，非零元素的坐标和值需要定义为Tensor

```python
# i为非零元素的坐标
i = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
# v为非零元素的值
v = torch.tensor([1, 2, 3, 4])
```

利用`sparse_coo_tensor()`函数定义稀疏张量

```python
a = torch.sparse_coo_tensor(i, v, (4, 4))
```

最后一个参数为Tensor的形状

```
tensor(indices=tensor([[0, 1, 2, 3],
                       [0, 1, 2, 3]]),
       values=tensor([1, 2, 3, 4]),
       size=(4, 4), nnz=4, layout=torch.sparse_coo)
```

可以转换为稠密张量

```python
a = a.to_dense()
```

```
tensor([[1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]])
```

> 注意非零元素的坐标和值的定义，采用的是COO风格矩阵的表示方法
>
> COO风格矩阵采用三元组表示元素的位置和值，即`(row, col, value)`
>
> `sparse_coo_tensor()`函数中
>
> - `indices`表示非零元素的位置，该参数定义为了含有两个向量的Tensor，第一个向量表示行，第二个向量表示列
> - `values`表示非零元素的值，该参数定义为了含有一个向量的Tensor，向量的值即为非零元素的值
>
> 两个参数共同组成了三元组，即COO风格矩阵

当然也可以同时指定Tensor的属性

```python
a = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32, device=dev).to_dense()
```

```
tensor([[1., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 3., 0.],
        [0., 0., 0., 4.]], device='cuda:0')
```

