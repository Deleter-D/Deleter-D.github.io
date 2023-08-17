---
title: 关于vec_cross_add接口的详细测试
toc: true
mathjax: true
tags:
  - 异构编程
  - 毕昇编译器
  - 机器学习
  - 深度学习
categories: 项目
abbrlink: 1040
date: 2023-08-17 20:14:52
---

先说结论，问题出在我粗心了，磕头道歉！！本接口没有任何问题，为上篇中不严谨的言论道歉。

<!-- more -->

# 关于`vec_cross_add`接口的详细测试

之前在写Softmax算子的时候，碰到了该接口结果不稳定的情况，所以进行一次详细测试，看看问题出在哪里。

## 测试方案

先用标准C++实现一个求和算子，用作计算标准。

```c++
using data_t = float;

data_t summary(std::vector<data_t> input) {
  data_t sum = 0.0;
  for (auto item : input) {
    sum += item;
  }
  return sum;
}
```

测试数据从`(-1,1)`的均匀分布中采样，为了方便复习，设置随机种子。结果的相对误差在2%以内均认为结果正确。

```c++
#define INPUT 64

int main() {
  std::vector<data_t> input;

  std::mt19937 gen(1234);
  std::uniform_real_distribution<data_t> urdis(-1, 1);
  for (std::size_t i = 0; i < INPUT; i++)
    input.push_back(urdis(gen));

  data_t sum = summary(input);
  data_t ascend_sum = ascend_summary(input);
  std::cout.precision(12);
  std::cout << std::setw(12) << "host sum: " << std::setw(12) << sum << std::endl;
  std::cout << std::setw(12) << "ascend sum: " << std::setw(12) << ascend_sum << std::endl;
  if ((std::fabs(sum - ascend_sum) / sum) < 0.02)
    std::cout << "Result correct." << std::endl;
  else
    std::cout << "Result error." << std::endl;

  return 0;
}
```

## 用例设计

为了尽可能全面测试该接口，同时排除不必要的影响，可以设置一下几种测试用例，均以`float`数据为例。

1. 输入向量长度为64，字节数为256B，1个repeat，用1个group处理。
2. 输入向量长度为128，字节数为512B，2个repeat，用1个group处理。
3. 输入向量长度为128，字节数为512B，2个repeat，用2个group处理。

这样设计用例可以分析出，到底是接口本身有问题，还是访存过程中出现了问题。用例1可以判断接口本身计算单个repeat的结果的正确性，用例2可以判断计算多个repeat时结果的正确性，用例3可以判断分核对repeat计算的影响。

## 算子实现

```c++
#define GROUP_NUM 1
#define ELEM_PER_GROUP 64

data_t ascend_summary(std::vector<data_t> &input) {
  sycl::queue Q(sycl::ascend_selector{});

  auto input_buf = sycl::malloc_device<data_t>(INPUT, Q);
  // 结果buf申请的长度，可以将每个核访问的block严格分离
  // 避免了由于block踩踏发生的问题
  const std::size_t res_vec_num = GROUP_NUM * (32 / sizeof(data_t));
  auto res_buf = sycl::malloc_device<data_t>(res_vec_num, Q);

  const std::size_t repeat_num = ELEM_PER_GROUP * sizeof(data_t) / 256;

  // Host -> GM
  Q.memcpy(input_buf, input.data(), INPUT * sizeof(data_t));

  Q.launch<class Sum>(GROUP_NUM, [=](sycl::group<1> group) {
    const std::size_t group_id = group.get_group_id();

    bisheng::vector<data_t, ELEM_PER_GROUP> input_vec;
    // 和向量的长度取决于本group内处理的repeat数量
    bisheng::vector<data_t, repeat_num> res_vec;

    input_vec.load(
        sycl::global_ptr<data_t>(input_buf + group_id * ELEM_PER_GROUP).get(),
        ELEM_PER_GROUP);

    bisheng::vec_cross_add(res_vec.data(), input_vec);

    res_vec.store(
        sycl::global_ptr<data_t>(res_buf + group_id * (32 / sizeof(data_t)))
            .get(),
        repeat_num);
  });

  // Host端的结果数组也避免了block踩踏
  std::vector<data_t> sum_host_vec(res_vec_num, 0.0f);
  Q.memcpy(sum_host_vec.data(), res_buf, res_vec_num * sizeof(data_t));
  Q.wait();

  data_t sum;
  for (std::size_t i = 0; i < res_vec_num; i++)
    // 只累加有意义的数据
    if (i % (32 / sizeof(data_t)) < repeat_num)
      sum += sum_host_vec[i];

  return sum;
}
```

针对不同的测试用例，只需要更改两个`#define`宏定义的数据即可。

## 测试结果

### 用例一

<img src="https://github.com/Deleter-D/Images/assets/56388518/a1c99f22-c4bf-41c8-b2a4-c0c9c025fab9" style="zoom: 50%;" />

宏修改为：

```c++
#define INPUT 64
#define GROUP_NUM 1
#define ELEM_PER_GROUP 64
```

结果没有任何问题。

```
  host sum: 1.76275920868
ascend sum: 1.76275908947
Result correct.
```

### 用例二

<img src="https://github.com/Deleter-D/Images/assets/56388518/070147b9-4cdf-4ebc-aeb7-e04920a2155a" style="zoom:50%;" />

宏修改为：

```c++
#define INPUT 128
#define GROUP_NUM 1
#define ELEM_PER_GROUP 128
```

结果也没有任何问题。

```
  host sum: 1.49505186081
ascend sum: 1.49505162239
Result correct.
```

之前问题就出在类似用例二的情况。当一个group处理多个repeat时，结果向量中会存在多个有意义的值，需要在累加时将这些有意义的值全部都加起来，之前问题就出在，没妥善处理单个核处理多repeat的情况，导致有一部分有意义的数没有累加上去，最终导致结果出错。

### 用例三

<img src="https://github.com/Deleter-D/Images/assets/56388518/4faaa240-9221-4358-8771-d4a85f3dcb13" style="zoom:50%;" />

宏修改为：

```c++
#define INPUT 128
#define GROUP_NUM 2
#define ELEM_PER_GROUP 64
```

结果自然也没有问题。

```
  host sum: 1.49505186081
ascend sum: 1.49505162239
Result correct.
```

# 重新实现Softmax

经过修改，使用`vec_cross_add()`接口的Softmax算子实现如下。

```c++
using data_t = float;

std::vector<data_t> ascend_softmax(std::vector<data_t> input) {
  std::size_t input_sz = input.size();
  std::size_t byte_count = input_sz * sizeof(data_t);

  // call the host operator if input isn't enough a full block
  if (byte_count < 32) {
    return softmax(input);
  }

  // number of elements per group
  const std::size_t elem_per_group = 640;
  // number of repeats per group
  const std::size_t repeat_per_group = elem_per_group * sizeof(data_t) / 256;
  // number of elements in tail block
  const std::size_t tail_elem_count = input_sz % elem_per_group;
  // number of groups
  // if tail block is exist, apply for one more group
  const std::size_t group_num = (tail_elem_count > 0)
                                    ? ((input_sz / elem_per_group) + 1)
                                    : (input_sz / elem_per_group);

  sycl::queue Q(sycl::ascend_selector{});

  // GM memory allocation
  auto dev_buf = sycl::malloc_device<data_t>(group_num * elem_per_group, Q);
  auto sum_res_buf = sycl::malloc_device<data_t>(group_num * (32 / sizeof(data_t)), Q);

  // Host memory allocation
  std::vector<data_t> sum_res(group_num * (32 / sizeof(data_t)), 0.0f);
  std::vector<data_t> res(input_sz, 0.0f);

  // host -> GM
  Q.memcpy(dev_buf, input.data(), byte_count);

  Q.launch<class Summary>(group_num, [=](sycl::group<1> group) {
    bisheng::vector<data_t, elem_per_group> input_vec;
    bisheng::vector<data_t, repeat_per_group> sum_vec;
    std::size_t group_id = group.get_group_id();

    // GM -> UB
    input_vec.load(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(), 
        elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      // if tail block has element and this is the last group
      bisheng::vector_view<data_t> input_vec_v(input_vec.data(), tail_elem_count);

      bisheng::vec_exp(input_vec_v, input_vec_v);
      for (int i = 0; i < tail_elem_count; ++i)
        sum_res_buf[group_id * (32 / sizeof(data_t))] += input_vec_v[i];
    } else {
      // full block data
      bisheng::vec_exp(input_vec, input_vec);
      bisheng::vec_cross_add(sum_vec.data(), input_vec);
      for (int i = 0; i < repeat_per_group; ++i) {
        sum_res_buf[group_id * (32 / sizeof(data_t))] += sum_vec[i];
      }
    }

    // UB -> GM
    input_vec.store(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  // GM -> Host
  Q.memcpy(sum_res.data(), sum_res_buf, group_num * (32 / sizeof(data_t)) * sizeof(data_t));
  Q.wait();

  data_t sum;
  for (int i = 0; i < sum_res.size(); i += 32 / sizeof(data_t))
    sum += sum_res[i];

  Q.launch<class Softmax>(group_num, [=](sycl::group<1> group) {
    // UB memory of exponent result
    bisheng::vector<data_t, elem_per_group> exp_res_vec;
    // UB memory of divisor
    bisheng::vector<data_t, elem_per_group> divisor_vec(sum);
    // UB memory of final result
    bisheng::vector<data_t, elem_per_group> res_vec;
    std::size_t group_id = group.get_group_id();

    // GM -> UB
    exp_res_vec.load(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      // if tail block has element and this is the last group
      bisheng::vector_view<data_t> exp_res_vec_v(exp_res_vec.data(),
                                                 tail_elem_count);
      bisheng::vector_view<data_t> divisor_vec_v(divisor_vec.data(),
                                                 tail_elem_count);
      bisheng::vector_view<data_t> res_vec_v(res_vec.data(),
                                             tail_elem_count);

      bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
    } else {
      // full block data
      bisheng::vec_div(res_vec, exp_res_vec, divisor_vec);
    }

    // UB -> GM
    res_vec.store(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  // GM -> host
  Q.memcpy(res.data(), dev_buf, byte_count);
  Q.wait();

  sycl::free(dev_buf, Q);
  sycl::free(sum_res_buf, Q);

  return res;
}
```

## 功能测试

功能测试全部验证正确。

## 性能测试

与之前效果最好的方案三对比。

| 测试用例     | 640      | 6400     | 64000    | 640000   |
| :----------- | :------- | :------- | :------- | :------- |
| 方案三加速比 | 0.224613 | 1.512394 | 13.30433 | 88.70575 |
| 新方案加速比 | 0.225836 | 1.330532 | 12.58051 | 101.0137 |

可以看到在向量长度比较大的情况下，接口的优势还是远大于`for`循环求和的。

再次为我不严谨的言论道歉！
