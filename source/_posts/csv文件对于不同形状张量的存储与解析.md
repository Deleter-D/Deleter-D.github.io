---
title: csv文件对于不同形状张量的存储与解析
toc: true
mathjax: true
tags:
  - python
  - numpy
  - csv
categories: 项目
abbrlink: 18632
date: 2023-11-22 20:58:58
---

关于同一个csv文件存储和解析不同形状张量的问题，看似简单，暗坑很多。

<!-- more -->

# csv文件对于不同形状张量的存储与解析

## 起因

最近写项目测试的时候，发小有一个小的需求，需要将不同形状的张量存入同一个文件中，同时需要记录其形状信息，从而方便算子的测试。但由于`numpy`中的`savetxt`等方法都要求张量形状一致，所以没办法直接使用。

一开始想的比较简单，使用`csv`文件来存储，分四列数据，前三列记录三个维度的形状信息，最后一列将整个张量存储下来。但实际开始写之后才发现，其中有一些问题处理起来比较麻烦，特此记录一下。

## 初步尝试

生成数据的代码比较简单，使用`numpy`随机生成一个张量，不仅张量中的数据是随机的，张量本身的形状也是在一定范围内随机生成的。

```python
def generateData3D(shape_limit, dtype):
    dim0 = randint(1, shape_limit[0] + 1)
    dim1 = randint(1, shape_limit[1] + 1)
    dim2 = randint(1, shape_limit[2] + 1)
    data = randn(dim0, dim1, dim2).astype(dtype)
    return (dim0, dim1, dim2, data)
```

这个函数返回一个元组，前三个元素记录形状，最后一个元素是张量本身。然后写一个简单的批量数据生成代码。

```python
def generateDataset(filepath, count, shape_limit, dtype):
    with open(filepath, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["dim0", "dim1", "dim2", "data"])
        for i in trange(count):
            data = generateData3D(shape_limit, dtype)
            writer.writerow(data)
```

## 问题出现

看起来没有任何问题，调用它来生成文件。

```python
generateDataset("test.csv", 10, (8, 8, 320), "float32")
```

这里的意思是生成10个张量，每个张量的形状最大为`(8, 8, 320)`，每一个维度都是0到该数之间的一个随机整数，数据类型是`float32`。我们来看看生成的`csv`文件的内容，这里只取有问题的一部分展示。

```
5,3,110,"[[[-8.6134362e-01  3.6351961e-01 -1.4064503e-01 ... -8.0155706e-01
    7.4856186e-01  3.2745436e-01]
  [-1.4208823e+00  4.2760447e-01 -7.0980674e-01 ...  3.3898675e-01
    1.9081663e-01  1.2164949e-01]
  [ 2.0867598e+00 -2.5110641e-01 -7.9451543e-01 ... -9.6020055e-01
    9.4596267e-01  1.8399478e-01]]

 [[-7.6276101e-02  1.0084456e+00 -5.4734468e-01 ...  7.9609489e-01
   -2.9747225e-02  3.6186981e-01]
  [ 1.6717568e-01 -2.7845892e-01 -6.9172156e-01 ... -7.8677136e-01
    4.0880820e-01  1.1563424e-01]
  [ 1.2279550e+00  2.7903655e+00  3.3148596e-01 ... -1.0443866e+00
   -1.7026719e-01 -7.7582508e-01]]
```

会发现生成的`csv`文件中出现了省略号，这是`numpy`在输出张量的时候为了美观做的处理，但当他被解析为字符串后，数据信息就丢失了。这一点比较好处理，只需要令`numpy`完整输出整个张量即可，我们修改`generateDataset`函数。

```python
def generateDataset(filepath, count, shape_limit, dtype):
    np.set_printoptions(threshold=np.inf) # 注意添加这一句
    with open(filepath, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["dim0", "dim1", "dim2", "data"])
        for i in trange(count):
            data = generateData3D(shape_limit, dtype)
            writer.writerow(data)
```

上述问题就得到了解决。

## 文件解析

到此我以为这个需求已经实现了，只需要再写一个解析`csv`文件的函数就好了，于是我写了如下函数。

```python
def loadFromFile(filepath, shape_limit, dtype):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            data.append(row)
    return data
```

我们调用它来读取刚才生产的文件。

```python
data = loadFromFile("test.csv", (8, 8, 320), "float32")
print(data[0])
```

这里只输出第一条记录，下面展示一部分。

```
['4', '3', '216', '[[[ 3.38217378e-01  8.90219882e-02  2.17747045e+00  1.05902016e+00\n   -1.54809088e-01 -1.73685062e+00  2.45146394e-01 -8.57644677e-01\n    2.61454940e+00 -5.65550804e-01 -6.17945969e-01 -2.49281359e+00\n   -1.82697034e+00 -2.62623811e+00  6.89034387e-02  1.78881836e+00\n   -9.63348448e-02 -2.35723400e+00  9.03523326e-01 -1.08545446e+00\n ......
```

这里发现输出的张量中有一些换行符，这是由于`numpy`格式化输出张量带来的后果，一开始我并没有觉得这是个问题，以`numpy`的强大能力，应该可以就这样将这个张量解析出来。

## 问题又出现

这里使用`type`函数查看从文件中读出来的张量的真实类型，即`type(data[0][3])`，发现是`<class 'str'>`，然后我做了如下尝试。

```python
np.fromstring(data[0][3], dtype=np.float32)
```

但是报如下错误。

```
ValueError: string size must be a multiple of element size
```

经过一通查询，原来`np.fromstring`是根据`dtype`来解析字符串的，它要求字符串的大小必须是元素个数的整数倍。但在我们存文件的时候会发现，有些数字是科学计数法存入的，有些数是普通的浮点数形式。对于字符串来说，每个数字映射后的字符串长度显然不一定是`float32`类型的4字节，所以解析的时候肯定会出问题。（PS. 又一个因为`numpy`格式化输出带来的问题。）

到此遇到的两个问题都是因为`numpy`的格式化输出引起的，所以我就尝试令`numpy`不要格式化输出，但在搜索了很多资料后我放弃了这个想法。（可能是我粗心没找到解决方案。）

后来换了一种思路来解决这个问题，我尝试先将这个读取出来的字符串解析成`python`的`list`，然后再用这个`list`来初始化一个`numpy`张量，从而供其他地方使用，于是进行了如下尝试。

```python
ast.literal_eval(data[0][3])
```

但执行会报如下错误。

```
    [[[ 3.38217378e-01  8.90219882e-02  2.17747045e+00  1.05902016e+00
                        ^
SyntaxError: invalid syntax
```

查询了很多资料，都没有说明这个问题是什么引起的。但在查询资料的过程中发现，别人在调用这个函数的时候，字符串都是以逗号隔开的一系列数字。这里由于`numpy`的格式化输出，是用空格隔开的，中间还有很多换行符。（again！）

于是我尝试令`numpy`输出的数据以逗号隔开，找了很多`numpy`中的写文件函数，均由于各种限制无法实现我的需求。所以在存文件的时候尝试将`numpy`数组先转换为字符串后，在写入文件。因为`numpy`的`array2string`函数是支持指定元素分隔符的，故我们改进`generateData3D`函数。

```python
def generateData3D(shape_limit, dtype):
    dim0 = randint(1, shape_limit[0] + 1)
    dim1 = randint(1, shape_limit[1] + 1)
    dim2 = randint(1, shape_limit[2] + 1)
    # 注意下面这句的修改
    data = np.array2string(randn(dim0, dim1, dim2).astype(dtype), separator=",")
    return (dim0, dim1, dim2, data)
```

重新生成文件后，再次尝试将该字符串解析为`list`，这次成功解析了！

```
[[[0.0282006636, -0.705630183, 0.205503568, 0.10408926, -0.130971402, -0.0346300565, 1.86623621, 1.35530257, -0.83048594, 1.27699852, -0.725055277, -0.514897704, 0.423814148, 1.65991676, -0.527909875, -0.678127706, -0.269491076, -1.05497122, 0.670092762, 1.45376074, 1.53001487, -0.844848216, 0.337865025, -0.144725695, -0.4941248, 0.819156349 ......
```

我们再进一步尝试将该`list`转为`numpy`张量。

```python
np.array(ast.literal_eval(data[0][3])).astype(np.float32)
```

打印该数组。

```
[[[ 0.02820066 -0.7056302   0.20550357 ...  0.01451482 -2.1041124
    1.8852443 ]]
 [[-0.05282624  0.5842251   0.70602226 ... -0.14502124  2.547757
    0.28345433]]
 [[-0.01983055 -0.5919099  -0.9039266  ... -1.7636172  -1.9037529
   -1.0482264 ]]
```

发现已经变成了`numpy`张量的格式化输出，再查看其类型和形状进一步确认有没有问题。

```
<class 'numpy.ndarray'>
(6, 1, 264)
```

类型与生成的数据中记录下来的一致，问题应该是解决了。

## 进一步优化

写到这里，回头看看可以发现，记录下来的形状信息其实并没有用到。因为在将字符串解析为`list`的时候，形状的信息自然的保留了下来。所以我们将存储形状信息的代码去掉，让这个`csv`文件只存储张量本身。

```python
def generateData3D(shape_limit, dtype):
    dim0 = randint(1, shape_limit[0] + 1)
    dim1 = randint(1, shape_limit[1] + 1)
    dim2 = randint(1, shape_limit[2] + 1)
    data = np.array2string(randn(dim0, dim1, dim2).astype(dtype), separator=",")
    return (data,) # 注意这里返回的依然是一个元组
```

```python
def generateDataset(filepath, count, shape_limit, dtype):
    np.set_printoptions(threshold=np.inf)
    with open(filepath, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        # 这里就直接写入数据了，没有写表头
        for i in trange(count):
            data = generateData3D(shape_limit, dtype)
            writer.writerow(data)
```

在读取数据的时候做一些相应处理即可。

```python
def loadFromFile(filepath, shape_limit, dtype):
    data = []
    csv.field_size_limit(sys.maxsize)
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.append(row)
    result = []
    for item in data:
        result.append(np.array(ast.literal_eval(item[0]), dtype=dtype))
    return result
```

这样读取出来的数据就是一个`list`，其中的每个元素都是一个`numpy.ndarray`。最后再写一个生成算子真值的函数。

```python
def generateGolden(output_file, input_file, dtype):
    np.set_printoptions(threshold=np.inf)
    input = tqdm(loadFromFile(input_file, dtype))
    softmax = Softmax()
    with open(output_file, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        for item in input:
            # 获取到的张量转换为mindspore的张量进行运算
            golden = softmax(ms.Tensor(item, dtype=ms.float32))
            # 转回numpy张量
            golden = golden.asnumpy()
            # numpy数组转字符串
            golden = np.array2string(golden.astype(dtype), separator=",")
            writer.writerow((golden,))
```

其中加了一些`tqdm`的内容，输出信息丰富一些。

## 完整代码

```python
import csv
import ast
import sys
import argparse
from tqdm import trange, tqdm
import numpy as np
from numpy.random import randint, randn
import mindspore as ms
from mindspore.nn import Softmax


def generateData3D(shape_limit, dtype):
    dim0 = randint(1, shape_limit[0] + 1)
    dim1 = randint(1, shape_limit[1] + 1)
    dim2 = randint(1, shape_limit[2] + 1)
    data = np.array2string(randn(dim0, dim1, dim2).astype(dtype), separator=",")
    return (data,)


def generateDataset(filepath, count, shape_limit, dtype):
    np.set_printoptions(threshold=np.inf)
    with open(filepath, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        for i in trange(count):
            data = generateData3D(shape_limit, dtype)
            writer.writerow(data)


def loadFromFile(filepath, dtype):
    data = []
    csv.field_size_limit(sys.maxsize)
    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.append(row)
    result = []
    for item in data:
        result.append(np.array(ast.literal_eval(item[0]), dtype=dtype))
    return result


def generateGolden(output_file, input_file, dtype):
    np.set_printoptions(threshold=np.inf)
    input = tqdm(loadFromFile(input_file, dtype))
    softmax = Softmax()
    with open(output_file, "w", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=",")
        for item in input:
            # 获取到的张量转换为mindspore的张量进行运算
            golden = softmax(ms.Tensor(item, dtype=ms.float32))
            # 转回numpy张量
            golden = golden.asnumpy()
            # numpy数组转字符串
            golden = np.array2string(golden.astype(dtype), separator=",")
            writer.writerow((golden,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data generation.")

    parser.add_argument(
        "--input-filename",
        type=str,
        default="input_data.csv",
        help="set the filename of input data.",
    )
    parser.add_argument(
        "--shape-limit",
        nargs="+",
        type=int,
        default=(32, 32, 320),
        help="the largest amount of each dimension.",
    )
    parser.add_argument(
        "--data-amount",
        type=int,
        default=100,
        help="the amount of input data.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="the data type of input data.",
    )
    parser.add_argument(
        "--golden-filename",
        type=str,
        default="golden_data.csv",
        help="set the filename of golden data.",
    )

    args = parser.parse_args()
    input_filename = args.input_filename
    golden_filename = args.golden_filename
    data_amount = args.data_amount
    dtype = args.dtype
    shape_limit = tuple(args.shape_limit)

    ms.set_context(device_target="Ascend", device_id=4)

    print(f"input filename: {input_filename}")
    print(
        f"\twill generating {data_amount} {dtype} input data with random shape less than {shape_limit}"
    )
    print("generating...")
    generateDataset(
        input_filename,
        data_amount,
        shape_limit,
        dtype,
    )

    print(f"golden filename: {golden_filename}")
    print("generating...")
    generateGolden(golden_filename, input_filename, dtype)

    print("all done.")
```

