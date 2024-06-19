---
title: 以张量角度考虑softmax算子
toc: true
mathjax: true
tags:
  - 异构编程
  - 毕昇编译器
categories: [高性能计算,项目]
abbrlink: 27559
date: 2023-12-05 16:01:50
---

在之前的文章中已经介绍了Softmax算子开发的整体思路，但笔者只从向量的角度进行了说明，本篇文章就以处理张量的角度来进一步阐述。

<!--more-->

# 张量的内存排布

要使Softmax算子处理张量，首先要了解张量在内存上的排布，我们用`numpy`的`ndarray`来说明。

首先利用`python`来创建一个`ndarray`。

```python
import numpy as np
data = np.random.randn(32, 32).astype(np.float32)
```

然后我们写一个简单的`C++`函数（`tensor.cpp`），该函数接受一个对应数据类型的指针作为参数，功能就简单的打印输出。

```c++
#include <iostream>

extern "C" void printTensor(float *tensor, int *shape){
    for(int i = 0; i < shape[0]; i++) {
        for(int j = 0; j < shape[1]; j++) {
            std::cout << tensor[i * shape[0] + j] << " ";
        }
        std::cout << "\n";
    }
}
```

其中的`extern "C"`是必要的，否则调用时会出现`undefined symbol`问题。

然后将其编译成一个动态链接库，以供`python`调用，命令如下。

```sh
g++ -shared -fPIC -o tensor.so tensor.cpp
```

> 注意：如果代码中使用了`C++`的标准库，则编译器要使用`g++`而不是`gcc`，否则会出现标准库符号找不到的问题，`clang`同理。

编译好后在`python`中调用该库，数据类型的转换是利用`ctypes`来实现的。

```python
import numpy as np
from ctypes import CDLL, POINTER, c_float, c_int

data = np.random.randn(32, 32).astype(np.float32)
c_int_arr = c_int * len(data.shape)

lib = CDLL("./tensor.so") # 获取动态链接库
printTensor = lib.printTensor # 获取库中的函数符号
printTensor.argtypes = [ # 定义函数的参数类型
    POINTER(c_float),
    POINTER(c_int),
]

# 函数调用
printTensor(
    data.ctypes.data_as(POINTER(c_float)), # 将numpy的ndarray转换为C语言的float职责
    c_int_arr(data.shape[0], data.shape[1]), # 将形状信息也传递给C++
)
```

部分执行结果如下所示。

```
-1.48456 0.336725 1.69846 -0.0248114 0.39322 0.614784 0.326595 0.575949 -0.708058 -1.39587 -1.83477 0.349339 -0.610898 -0.423076 -0.136989 0.442269 0.446412 -0.486558 -0.292987 1.29332 0.187811 0.331237 0.63905 -1.46251 -0.536956 0.495119 -0.429213 -0.988436 -0.414105 -2.26553 1.23408 -0.544561 
0.369648 -1.12966 -0.154628 1.09682 0.676383 0.444374 -0.706796 -0.873308 -1.32488 -0.537758 -1.81611 -2.06588 0.721618 1.02888 -0.919128 -0.765203 -0.42332 0.0602946 1.16713 0.140398 -0.534829 -0.0961945 0.0153079 -0.261519 0.0927059 -0.868659 1.27008 -0.379786 0.382002 -1.76778 0.660476 1.06135
...
```

以上得出结论，`numpy`中的张量在内存上就是一个**一维数组**，可以用指针来操作，其他框架的张量同理。

# 算子逻辑

由于张量在内存上都是一维排布的，所以最内层维度在内存上是连续的。所以对于昇腾芯片来说，`Softmax`最适合加速的就是在张量的最后一个维度上进行计算，下面的讨论都基于最后一个维度。对于一个NHWC的张量来说，我们在最后一个维度上进行`Softmax`也是实际中最常用的情况。

算子涉及向量自然指数、向量归约求和、向量除法等运算，其中最需要关心的就是向量归约求和，因为它涉及到对齐的问题。

## 可能的数据情况

主要有以下几种情况：

### 情况一：数据`repeat`对齐

最好解决的就是`repeat`对齐的情况，不需要做尾块处理，直接利用`vec_cross_add()`归约求和即可。

![](https://github.com/Deleter-D/Images/assets/56388518/32e28548-e1d0-4781-b340-fa7631e1fba8)

### 情况二：数据部分`repeat`对齐，部分`block`对齐

对于部分`repeat`对齐，部分`block`对齐的情况，需要分开来处理。对于`repeat`对齐的部分同样简单处理，对于`block`对齐的部分，无法直接调用`vec_cross_add`接口进行归约求和，需要利用标量操作来累加进前面的结果中。

![](https://github.com/Deleter-D/Images/assets/56388518/9beafb91-6f29-4fbf-bc9b-5f3ba6996c4c)

### 情况三：数据部分`repeat`对齐，部分`block`对齐，剩余尾块`block`不对齐

最后一种情况是最需要注意的情况，而且实际使用中大部分是这种情况。着重关注非`block`对齐部分的数据，这部分数据要从搬移的时候就开始做单独处理。因为GM与UB之间的数据搬移最小粒度是一个`block`，无法真正做到元素级别的搬移。

<img src="https://github.com/Deleter-D/Images/assets/56388518/3f33eedc-f0df-47f2-992c-acde552b4b62" style="zoom: 67%;" />

对于这种情况的数据搬移，我们考虑一种简化情况，即数据长度大于一个`block`但不足两个`block`。对于这样的数据，GM与UB之间的搬移需要一个临时空间来辅助。

<img src="https://github.com/Deleter-D/Images/assets/56388518/dd30457f-da75-4fbc-8f4c-fab2a5dc42b6" style="zoom:50%;" />

具体方式是从末尾向前取一个整`block`进行搬移，这样不会影响到后续的数据，同时使得`group`之间的访存严格隔离开来。

这样处理的时候，由于会有被重复搬移的数据，所以要注意在累加的时候不要重复累加元素。

## 算子实现

为了避免张量过大，在UB上申请的空间超出限制，这里使`group`循环分批处理一个向量。即从GM搬移进UB，处理完后再搬回GM，再搬入下一批数据进行处理，直到所有数据被处理完成。

这样处理有一个好处就是，情况二和情况三的数据只会出现在最后一次迭代中。该算子的处理大体分为三个小模块，求$e^x$、归约求和以及向量除法。

在写核心逻辑之前，我们需要为算子准备一系列的常量，方便后面使用。

```cpp
// 一个向量的repeat数量
std::size_t total_repeat_count = vec_bytes / REPEAT_SIZE;
// 需要的核内迭代次数
std::size_t iteration_times = total_repeat_count / MAX_REPEAT_PER_ITERATION + 1;
// 最后一次迭代处理的字节数
std::size_t last_iter_bytes = vec_bytes - (iteration_times - 1) * MAX_BYTES_PER_ITERATION;
// 最后一次迭代数据中元素的个数
std::size_t elem_count = last_iter_bytes / sizeof(float);
// 最后一次迭代数据中对齐block的个数
std::size_t block_count = last_iter_bytes / BLOCK_SIZE;
// 最后一次迭代数据中对齐repeate的个数
std::size_t repeat_count = last_iter_bytes / REPEAT_SIZE;
// 最后一次迭代数据中block对齐的元素个数
std::size_t align_block_elem_count = block_count * BLOCK_SIZE / sizeof(float);
// 最后一次迭代数据中repeat对齐的元素个数
std::size_t align_repeat_elem_count = repeat_count * REPEAT_SIZE / sizeof(float);
// 最后一次迭代数据中非对齐元素个数
std::size_t tail_elem_count = elem_count - align_block_elem_count;
// 最后一次迭代数据中非对齐字节数
std::size_t tail_bytes = tail_elem_count * sizeof(float);
// 最后一次迭代数据中，向前取整block的元素个数
std::size_t tail_block_elem_count = BLOCK_SIZE / sizeof(float);
// 最后一次迭代数据中，向前取整block的起点索引
std::size_t tail_memcpy_index = dim2 - tail_block_elem_count;
```

核心逻辑如下，详细说明见注释。

```cpp
for (std::size_t i = 0; i < iteration_times; i++) {
  index = group_id * dim2 + i * MAX_BYTES_PER_ITERATION / sizeof(float);
  if (i == iteration_times - 1) {
    tail_index = group_id * dim2 + tail_memcpy_index;
    // 加载最后一次迭代中，block对齐的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    if (tail_bytes) {
      // 加载最后一次迭代中，block非对齐的数据，向前取整block
      temp.load(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
    bisheng::vec_exp(input_vec.to_view(elem_count), input_vec.to_view(elem_count));
    bisheng::vec_exp(temp, temp);
    input_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    if (tail_bytes) {
      temp.store(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
  } else {
    // 整块的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
    bisheng::vec_exp(input_vec, input_vec);
    input_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
  }
}

// 计算向量和
for (std::size_t i = 0; i < iteration_times; i++) {
  index = group_id * dim2 + i * MAX_BYTES_PER_ITERATION / sizeof(float);
  if (i == iteration_times - 1) {
    tail_index = group_id * dim2 + tail_memcpy_index;
    // 加载最后一次迭代中，block对齐的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    if (tail_bytes) {
      // 加载最后一次迭代中，block非对齐的数据，向前取整block
      temp.load(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
    if (align_repeat_elem_count) {
      // 将最后一次迭代中repeat对齐的数据求和
      bisheng::vec_cross_add(sum_vec.to_view(repeat_count, 0, 1), input_vec.to_view(align_repeat_elem_count));
      for (std::size_t j = 0; j < repeat_count; j++) {
        sum += sum_vec[j];
      }
    }
    // 计算repeat不对齐，但block部分对齐的数据
    for (std::size_t j = align_repeat_elem_count; j < align_block_elem_count; j++) {
      sum += input_vec[j];
    }
    // 计算block不对齐的数据
    for (std::size_t j = tail_block_elem_count - tail_elem_count; j < tail_block_elem_count; j++) {
      sum += temp[j];
    }
  } else {
    // 整块的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
    bisheng::vec_cross_add(sum_vec.to_view(MAX_REPEAT_PER_ITERATION, 0, 1), input_vec.to_view(MAX_BYTES_PER_ITERATION / sizeof(float)));
    for (std::size_t j = 0; j < MAX_BYTES_PER_ITERATION / sizeof(float); j++) {
      sum += sum_vec[j];
    }
  }
}

// 利用向量和初始化分母向量
bisheng::vector<float, BLOCK_SIZE / sizeof(float)> temp_res;
bisheng::vector<float, MAX_BYTES_PER_ITERATION / sizeof(float)> divisor(sum);
bisheng::vector<float, BLOCK_SIZE / sizeof(float)> temp_divisor(sum);

for (std::size_t i = 0; i < iteration_times; i++) {
  index = group_id * dim2 + i * MAX_BYTES_PER_ITERATION / sizeof(float);
  if (i == iteration_times - 1) {
    tail_index = group_id * dim2 + tail_memcpy_index;
    // 加载最后一次迭代中，block对齐的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    if (tail_bytes) {
      // 加载最后一次迭代中，block非对齐的数据，向前取整block
      temp.load(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
    bisheng::vec_div(res_vec.to_view(elem_count), input_vec.to_view(elem_count), divisor.to_view(elem_count));
    bisheng::vec_div(temp_res, temp, temp_divisor);
    res_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    if (tail_bytes) {
      temp_res.store(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
  } else {
    // 整块的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
    bisheng::vec_div(res_vec, input_vec, divisor);
    res_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
  }
}
```

# 算子优化

## 算子逻辑优化

观察上述的核心逻辑，可以观察到几个比较明显的优化点：

- 计算$e^x$和归约求和的过程可以合并，不需要先计算$e^x$后搬出，再搬入计算归约和；
- 最后计算除法的过程，可以用倒数乘法来代替；
- 可以开启`double buffering`。

优化后的核心逻辑如下所示。

```cpp
// 计算e^x并归约求和
for (std::size_t i = 0; i < iteration_times; i++) {
  index = group_id * dim2 + i * MAX_BYTES_PER_ITERATION / sizeof(float);
  // 判断当前缓冲区
  auto &input_vec = i % 2 ? input_vec_0 : input_vec_1;
  auto &sum_vec = i % 2 ? sum_vec_0 : sum_vec_1;
  auto &sum_temp = i % 2 ? sum_temp_0 : sum_temp_1;
  auto &sum = i % 2 ? sum_0 : sum_1;

  if (i == iteration_times - 1) {
    tail_index = group_id * dim2 + tail_memcpy_index;
    // 加载最后一次迭代中，block对齐的数据
    if (align_block_elem_count) {
      input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
      bisheng::vec_exp(input_vec.to_view(elem_count), input_vec.to_view(elem_count));
      // 将最后一次迭代中repeat对齐的数据求和
      if (align_repeat_elem_count) {
        bisheng::vec_cross_add(sum_vec.to_view(repeat_count, 0, 1), input_vec.to_view());
        for (int j = 0; j < repeat_count; j++) {
          sum += sum_vec[j];
        }
      }
      // 计算repeat不对齐，但block部分对齐的数据
      for (std::size_t j = align_repeat_elem_count; j < align_block_elem_count; j++) {
        sum += input_vec[j];
      }
      input_vec.store(sycl::global_ptr<float>(d_exp_tensor + index).get(), align_block_elem_count);
    }
    if (tail_bytes) {
      // 加载最后一次迭代中，block非对齐的数据，向前取整block
      temp.load(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
      bisheng::vec_exp(temp, temp);
      // 计算block不对齐的数据
      for (std::size_t j = tail_block_elem_count - tail_elem_count; j < tail_block_elem_count; j++) {
        sum += temp[j];
      }
      temp.store(sycl::global_ptr<float>(d_exp_tensor + tail_index).get(), tail_block_elem_count);
    }

  } else { // 整块的数据
    input_vec.load(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
    bisheng::vec_exp(input_vec, input_vec);
    bisheng::vec_cross_add(sum_vec.to_view(MAX_REPEAT_PER_ITERATION, 0, 1), input_vec.to_view(MAX_BYTES_PER_ITERATION / sizeof(float)));
    bisheng::vec_cross_add(sum_temp.to_view(sum_vec_repeat_count, 0, 1), sum_vec.to_view(MAX_REPEAT_PER_ITERATION));
    for (int j = 0; j < sum_vec_repeat_count; j++) {
      sum += sum_temp[j];
    }
    input_vec.store(sycl::global_ptr<float>(d_exp_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
  }
}

// 两个缓冲区的和相加
sum_0 += sum_1;
auto &sum = sum_0;

bisheng::vector<float, BLOCK_SIZE / sizeof(float)> temp_res;
float divisor = 1 / sum;

// 向量除法
for (std::size_t i = 0; i < iteration_times; i++) {
  index = group_id * dim2 + i * MAX_BYTES_PER_ITERATION / sizeof(float);
  // 判断当前缓冲区
  auto &input_vec = i % 2 ? input_vec_0 : input_vec_1;
  auto &res_vec = i % 2 ? res_vec_0 : res_vec_1;
  if (i == iteration_times - 1) {
    tail_index = group_id * dim2 + tail_memcpy_index;
    // 加载最后一次迭代中，block对齐的数据
    if (align_block_elem_count) {
      input_vec.load(sycl::global_ptr<float>(d_exp_tensor + index).get(), align_block_elem_count);
      bisheng::vec_mul(res_vec.to_view(elem_count), input_vec.to_view(elem_count), divisor);
      res_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), align_block_elem_count);
    }
    if (tail_bytes) {
      // 加载最后一次迭代中，block非对齐的数据，向前取整block
      temp.load(sycl::global_ptr<float>(d_exp_tensor + tail_index).get(), tail_block_elem_count);
      bisheng::vec_mul(temp_res, temp, divisor);
      temp_res.store(sycl::global_ptr<float>(d_tensor + tail_index).get(), tail_block_elem_count);
    }
  } else { // 整块的数据
    input_vec.load(sycl::global_ptr<float>(d_exp_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
    bisheng::vec_mul(res_vec, input_vec, divisor);
    res_vec.store(sycl::global_ptr<float>(d_tensor + index).get(), MAX_BYTES_PER_ITERATION / sizeof(float));
  }
}
```

## 分核方案优化

由于毕昇异构算子中存在一个限制，即`group`的数量最大为65535。按照上述的分核方案，最多只能处理65535个向量，显然是不合理的。所以，当向量个数大于65535时，要令每个逻辑核处理多个向量。

```cpp
// 每个group处理的向量个数
std::size_t vec_count_per_group = (vec_count + MAX_KERNEL_COUNT - 1) / MAX_KERNEL_COUNT;
// 开启的group数量
std::size_t group_count = (vec_count + vec_count_per_group - 1) / vec_count_per_group;
```

```cpp
for (std::size_t i = 0; i < vec_count_per_group; i++) {
  iteration_begin = group_index + i * dim3;
  if (iteration_begin >= element_total_count) // 注意判断边界
    break;
  bisheng::vector<float, MAX_BYTES_PER_ITERATION / sizeof(float)> input_vec_0;
  bisheng::vector<float, MAX_BYTES_PER_ITERATION / sizeof(float)> input_vec_1;
  bisheng::vector<float, MAX_REPEAT_PER_ITERATION> sum_vec_0(0);
  bisheng::vector<float, MAX_REPEAT_PER_ITERATION> sum_vec_1(0);
  const std::size_t sum_vec_repeat_count = MAX_REPEAT_PER_ITERATION * sizeof(float) / REPEAT_SIZE;
  bisheng::vector<float, sum_vec_repeat_count> sum_temp_0(0);
  bisheng::vector<float, sum_vec_repeat_count> sum_temp_1(0);
  bisheng::vector<float, MAX_BYTES_PER_ITERATION / sizeof(float)> res_vec_0;
  bisheng::vector<float, MAX_BYTES_PER_ITERATION / sizeof(float)> res_vec_1;
  bisheng::vector<float, BLOCK_SIZE / sizeof(float)> temp;
  __local float sum_0 = 0.0f;
  __local float sum_1 = 0.0f;

  // 计算e^x并归约求和
  for (std::size_t j = 0; j < iteration_times; j++) {
    ...
  }

  // 两个缓冲区的和相加
  sum_0 += sum_1;
  auto &sum = sum_0;

  bisheng::vector<float, BLOCK_SIZE / sizeof(float)> temp_res;
  float divisor = 1 / sum;

  // 向量除法
  for (std::size_t j = 0; j < iteration_times; j++) {
    ...
  }
}
```

# 功能测试

功能测试采取将算子封装到`MindSpore`框架中进行测试，具体方案如下。先将算子代码编译为动态链接库。

```sh
clang++ -fsycl -fdevices=ascend_910 \
    -I ${ASCEND_TOOLKIT_HOME}/include \
    -L ${ASCEND_TOOLKIT_HOME}/lib64 -lascendcl \
    -shared -fPIC -o softmax.so \
    -mllvm -inline-threshold=9000 -mllvm -enable-explicit-vectorizer -Rpass=ascend-vec \
    ./softmax.cpp
```

然后按照要求封装为`MindSpore`可调用的状态。

```python
class SoftmaxBS(Cell):
    def __init__(self):
        super(SoftmaxBS, self).__init__()
        self.bisheng_softmax = ops.Custom(
            "softmax.so:softmax_npu",
            out_shape=lambda x: x,
            out_dtype=lambda x: x,
            func_type="aot",
        )
        self.bisheng_softmax.add_prim_attr("primitive_target", "Ascend")

    def construct(self, x0):
        output = self.bisheng_softmax(x0)
        return output
```

在结果正确性方面，采用了`numpy`中的`allClose()`函数来对比`MindSpore`算子与自定义算子的结果张量，若两者在一定精度范围内接近，则认为计算结果正确。具体判断逻辑如下。

```python
context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
softmax_bs = SoftmaxBS()
softmax = Softmax(axis=-1)
data = ms.Tensor(np.random.randn(dim0, dim1, dim2, dim3), ms.float16)
output_bs = softmax_bs(data)
output_ms = softmax(data)
if np.allclose(output_bs.asnumpy(), output_ms.asnumpy(), rtol=1e-3, atol=1e-3):
    print("correct!")
else:
    print("error!")
```

经过测试，算子逻辑没有问题，精度由于使用了`float16`来计算，所以只设置到了`1e-3`。

# 性能测试

性能测试采用单算子测试的方式。对于`MindSpore`中的算子，采用框架自带的`Profiler()`来分析算子性能，再通过`msprof.py`脚本工具导出算子性能数据的`summary`数据，通过读取`Task Duration`列来获取算子的执行时间。而对于自定义算子，则采用`msprof`命令行工具运行算子，同样通过`summary`数据来获取算子执行时间。

| ID   | Shape           | 数据类型 | MindSpore | BiSheng  | 加速比   |
| ---- | --------------- | -------- | --------- | -------- | -------- |
| 1    | 8x16x1024x1024  | half     | 2658.672  | 9720.826 | 0.273503 |
| 2    | 16x16x1024x1024 | half     | 5274.796  | 18483.56 | 0.285378 |
| 3    | 16x16x1024x2048 | half     | 10550.81  | 21184.74 | 0.498038 |
| 4    | 16x16x1024x4096 | half     | 22255.22  | 26612.28 | 0.836276 |
| 5    | 4x4x512x8192    | half     | 1221.49   | 1350.776 | 0.904288 |
| 6    | 4x4x512x16384   | half     | 2438.494  | 1337.628 | 1.822999 |
| 7    | 4x4x512x32768   | half     | 4869.044  | 2412.854 | 2.01796  |
| 8    | 4x4x512x65535   | half     | 6671.878  | 6200.692 | 1.075989 |
| 9    | 4x4x512x131072  | half     | 19467.07  | 8748.974 | 2.225069 |
| 10   | 4x4x512x8193    | half     | 4984.124  | 1564.236 | 3.186299 |
| 11   | 4x4x512x16385   | half     | 1354.314  | 1555.814 | 0.870486 |
| 12   | 4x4x512x32769   | half     | 4288.27   | 2667.928 | 1.607341 |
| 13   | 4x4x512x65536   | half     | 9732.024  | 4516.888 | 2.154586 |
| 14   | 4x4x512x131073  | half     | 79671.69  | 9390.606 | 8.484191 |

经过一系列的性能测试，发现在小数据量的情况下，性能始终无法与TBE算子相比。推测可能的原因是，TBE算子针对某些静态形状有优化，但本算子针对的是动态形状场景，所以性能较差。但当数据量变大，充分发挥设备并行能力的情况下，性能有所好转。在其最擅长的形状上，加速比可以达到2左右，在用例14这种情况下，加速比甚至达到了8以上。
