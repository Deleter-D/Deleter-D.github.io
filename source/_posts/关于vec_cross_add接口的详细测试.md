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

先说结论，问题出在我粗心了，磕头道歉！！本接口没有计算逻辑性问题，为上篇中不严谨的言论道歉。但在特定情况下，会有不符合预期的返回值。

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

# 特定不符合预期的情况

## 情况描述

在此之前，我认为这个接口确实没问题了，但在偶然的情况下，还是测试出了不符合预期返回结果的情况。

在使用`vec_cross_add()`时，我们应该定义两个向量，一个作为输入向量，另一个作为结果向量。正常情况下，结果向量的长度应该为输入向量的repeat数量。例如：

```c++
bisheng::vector<float, 128> src; // src为128个float，也就是2个repeat
bisheng::vector<float, 2> dst; // 结果向量长度应该为2
bisheng::vec_cross_add(src.data(), dst); // 正常的调用方式
```

这种方式确实没有问题，结果返回也正常，将结果数组中的元素都相加后，确实是输入向量元素的和。

但当我们用变长向量`bisheng::vector_view`时，返回结果就不像预期这样了。

下面实现了一个利用变长向量操作数据的算子。大部分代码与上面的一致，只需关注有注释的部分。

```c++
data_t ascend_summary_view(std::vector<data_t> &input) {
  sycl::queue Q(sycl::ascend_selector{});

  auto input_buf = sycl::malloc_device<data_t>(INPUT, Q);
  const std::size_t res_vec_num = GROUP_NUM * (32 / sizeof(data_t));
  auto res_buf = sycl::malloc_device<data_t>(res_vec_num, Q);

  const std::size_t repeat_num = ELEM_PER_GROUP * sizeof(data_t) / 256;

  Q.memcpy(input_buf, input.data(), INPUT * sizeof(data_t));

  sycl::stream out(512 * GROUP_NUM, 512, Q.get_device());
  Q.launch<class SumView>(GROUP_NUM, [=](sycl::group<1> group) {
    const std::size_t group_id = group.get_group_id();

    // 申请较大的向量
    bisheng::vector<data_t, 30000> input_vec;
    bisheng::vector<data_t, 30000> res_vec;

    input_vec.load(
        sycl::global_ptr<data_t>(input_buf + group_id * ELEM_PER_GROUP).get(),
        ELEM_PER_GROUP);

    // 使用变长向量来操作正确范围内的数据
    bisheng::vector_view<data_t> input_vec_v(input_vec.data(), ELEM_PER_GROUP);
    bisheng::vector_view<data_t> res_vec_v(res_vec.data(), repeat_num);

    bisheng::vec_cross_add(res_vec_v, input_vec_v);

    // 输出变长结果向量前20个元素
    out << "group " << group_id << "\n";
    for (std::size_t i = 0; i < 20; ++i)
      out << res_vec_v[i] << " ";
    out << "\n";

    res_vec.store(
        sycl::global_ptr<data_t>(res_buf + group_id * (32 / sizeof(data_t)))
            .get(),
        repeat_num);
  });

  std::vector<data_t> sum_host_vec(res_vec_num, 0.0f);
  Q.memcpy(sum_host_vec.data(), res_buf, res_vec_num * sizeof(data_t));
  Q.wait();

  data_t sum;
  for (std::size_t i = 0; i < res_vec_num; i++)
    if (i % (32 / sizeof(data_t)) < repeat_num)
      sum += sum_host_vec[i];

  return sum;
}
```

为了更清楚的看到其中发生了什么，故定义一个`sycl::stream`来输出Kernel中的信息。

理论上，即使是变长向量，返回的结果向量长度也应该是输入向量的repeat数，在这里用上面的用例二进行说明。

此时的宏为：

```c++
#define INPUT 128
#define GROUP_NUM 1
#define ELEM_PER_GROUP 128
```

先来看计算结果的输出：

```
  host sum: 1.49505186081
ascend sum: 1.76275908947
Result error.
```

计算错误，这成功复现了另一种错误的情况。这个错误表面上看起来，和上面使用定长向量测试时描述的错误原因一样，原因没有将所以有意义的数据累加进去，最终导致了结果不正确。

先想一个问题，我们在Kernel中输出了结果向量的前20个元素，理论上讲，输出应该类似于下面这种形式：

```
1.7628 -0.2677 ......
```

前两个元素应该就是两个repeat的和，后面的元素应该都是未初始化的无意义的数据。

**但问题的关键来了，来看一看Kernel中输出的结果向量的前20个元素。**

```
1.7628 0.0000000013091844320297241211 -0.0000000027893838882446289062 0.0000000034829735755920410156 0.000000041852550506591796875 -0.000000004057476043701171875 -0.0000004261577606201171875 0.0000000014236330986022949219 -0.2677 -0.000000044695949554443359375 -0.000000012639684677124023437 0.00000000012021332979202270508 -0.000000060646677017211914062 -0.0000000027346568107604980469 -0.000000011391909122467041016 -0.000000012336615324020385742 0.00000018131090164184570312
```

这样看不够直观，我们将每个元素都换个行再输出，得到如下。

```
1.7628
0.0000000013091844320297241211
-0.0000000027893838882446289062
0.0000000034829735755920410156
0.000000041852550506591796875
-0.000000004057476043701171875
-0.0000004261577606201171875
0.0000000014236330986022949219
-0.2677
-0.000000044695949554443359375
-0.000000012639684677124023437
0.00000000012021332979202270508
-0.000000060646677017211914062
-0.0000000027346568107604980469
-0.000000011391909122467041016
-0.000000012336615324020385742
0.00000018131090164184570312
```

观察到的结果是有些惊喜的，`vec_cross_add`并没有按照预期的形式返回结果。这些很小的数都是未初始化的无意义数据，真正有意义的数据被放在的位置0和8上。

而为什么是这两个位置，一种可能的解释是，该接口在运算的过程中，是按照block来计算的，一个repeat是8个block，所以理论上会有8个求和结果，然后这个接口将8个block的和再累加起来，放到结果向量的第一个位置。按照正常来讲，最终返回结果之前，应该将所有repeat的求和结果做一个紧凑排布，就会得到像定长向量那样正常的结果向量。可能在实现变长向量重载的时候没有做这个紧凑？我乱猜的（狗头保命）。

## 解决方案

所以只要注意这一点，在使用变长向量的时候，将正确位置的元素累加即可得到正确的结果。

正确的代码实现如下，这里只展示核心部分。

```c++
Q.launch<class SumViewFixed>(GROUP_NUM, [=](sycl::group<1> group) {
    const std::size_t group_id = group.get_group_id();

    bisheng::vector<data_t, 30000> input_vec;
    bisheng::vector<data_t, 30000> res_vec;

    input_vec.load(
        sycl::global_ptr<data_t>(input_buf + group_id * ELEM_PER_GROUP).get(),
        ELEM_PER_GROUP);

    bisheng::vector_view<data_t> input_vec_v(input_vec.data(), ELEM_PER_GROUP);
    // 使用变长向量的时候就要考虑到正确数据的存放位置
    bisheng::vector_view<data_t> res_vec_v(res_vec.data(), repeat_num * 8);

    bisheng::vec_cross_add(res_vec_v, input_vec_v);

    // 结果就直接用标量操作写回GM
    for (std::size_t i = 0; i < repeat_num * 8; i += 8)
      res_buf[group_id * (32 / sizeof(data_t))] += res_vec_v[i];
  });
```

## 测试结果

```
  host sum: 1.49505186081
ascend sum: 1.49505162239
Result correct.
```

这是一个比较隐蔽的问题，在使用该接口的时候还是要多加注意。
