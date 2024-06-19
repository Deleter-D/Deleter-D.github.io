---
title: 基于毕昇编译器的softmax异构算子
toc: true
mathjax: true
tags:
  - 异构编程
  - 毕昇编译器
  - 机器学习
  - 深度学习
categories: [高性能计算,项目]
abbrlink: 15984
date: 2023-07-31 16:10:00
---

使用毕昇编译器异构开发Softmax算子，坑太多太多了。。。。

<!-- more -->

# 分析算子

Softmax是非常常见的激活函数，此处不过多赘述。观察其公式：
$$
\text{Softmax}(x) = \frac{e^{x_i}}{\sum_ie^{x_i}}
$$
首先应该注意到的是分母中的求和，因为向量内元素求和本身就是一个不太适合于向量化的操作，初步分析可知，此处的求和可能是性能的瓶颈。而分子除以分母的运算可以在得到分母（即向量元素之和）的情况下，自然而然的向量化运算。

# 初步方案（方案一）

- 向量化方案：因为向量除法是可以自然而然向量化的操作，故暂时先将分母的求和操作放在Host端进行。
- 分核方案：本算子是在昇腾910B上进行开发的，由于数据对其以及访存粒度的限制，暂时令一个核处理640个元素。

<img src="https://github.com/Deleter-D/Images/assets/56388518/8acb8eb7-a02d-4699-9601-49df2dbe3ab2" style="zoom: 50%;" />

## 初步方案实现

先用标准C++实现一个Softmax算子，作为功能验证和性能对比的基准。

```c++
using data_t = float;

std::vector<data_t> softmax(std::vector<data_t> input) {
  data_t sum = 0.0;
  for (auto x : input) {
    sum += expf(x);
  }
  std::vector<data_t> res;
  for (auto x : input) {
    res.push_back(expf(x) / sum);
  }
  return res;
}
```

接下来开始实现异构算子的逻辑。

```c++
std::vector<data_t> ascend_softmax(std::vector<data_t> input) {
  // 首先拿到输入向量的大小
  std::size_t input_sz = input.size();
  // 计算出总的数据量，单位是字节
  std::size_t byte_count = input_sz * sizeof(data_t);

  // 如果输入向量的大小不足一个block，则直接调用Host算子逻辑
  if (byte_count < 32)
    return softmax(input);

  sycl::queue Q(sycl::ascend_selector{});

  // 申请GM上的内存，分别存储输入和结果
  auto input_buf = sycl::malloc_device<data_t>(input_sz, Q);
  auto res_buf = sycl::malloc_device<data_t>(input_sz, Q);

  // host -> GM 的内存搬移
  Q.memcpy(input_buf, input.data(), byte_count);

  // 指定每个核处理的元素个数
  const std::size_t elem_per_group = 640;
  // 计算尾块中的元素个数
  const std::size_t tail_elem_count = input_sz % elem_per_group;
  // 逻辑核的数量，若尾块中存在元素，则多开一个逻辑核
  const std::size_t group_num = (tail_elem_count > 0)
                                    ? ((input_sz / elem_per_group) + 1)
                                    : (input_sz / elem_per_group);

  // 求和计算暂时由Host完成
  data_t sum = 0.0;
  for (auto x : input) {
    sum += expf(x);
  }

  Q.launch<class Softmax>(group_num, [=](sycl::group<1> group) {
    // UB内存申请，分别为输入向量、指数计算结果向量、分母向量、结果向量
    bisheng::vector<data_t, elem_per_group> input_vec;
    bisheng::vector<data_t, elem_per_group> exp_res_vec;
    bisheng::vector<data_t, elem_per_group> divisor_vec(sum);
    bisheng::vector<data_t, elem_per_group> res_vec;
    // 获取group id
    std::size_t group_id = group.get_group_id();

    // GM -> UB 的内存搬移
    input_vec.load(
        sycl::global_ptr<data_t>(input_buf + group_id * elem_per_group).get(), elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      // 本分支处理存在尾块，且当前是最后一个处理尾块的group
      // 由于尾块大概率是非整block，故采用毕昇变长向量对数据进行操作
      bisheng::vector_view<data_t> input_vec_v(input_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> exp_res_vec_v(exp_res_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> divisor_vec_v(divisor_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> res_vec_v(res_vec.data(), tail_elem_count);

      bisheng::vec_exp(exp_res_vec_v, input_vec_v);
      bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
    } else {
      // 本分支处理整block的情况
      // 由于指定了每个核处理的元素个数，故此处一定是整block的
      bisheng::vec_exp(exp_res_vec, input_vec);
      bisheng::vec_div(res_vec, exp_res_vec, divisor_vec);
    }

    // UB -> GM 内存搬移
    res_vec.store(
        sycl::global_ptr<data_t>(res_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  std::vector<data_t> res(input_sz, 0.0f);
  // GM -> host 内存搬移
  Q.memcpy(res.data(), res_buf, byte_count);
  Q.wait();
  // 释放资源
  sycl::free(input_buf, Q);
  sycl::free(res_buf, Q);

  return res;
}
```

## 功能验证

由于之前实现了Host端的逻辑，所以可以用Host端的计算结果来验证异构算子的正确性，测试代码如下。输入向量的数据是从(-1, 1)均的匀分布中取的随机数。

由于设备端会有一定的精度损失，故在测试过程中留有一定的宽容度，相对精度损失在2%以内均认为计算正确。

```c++
int main() {
  std::vector<data_t> vec;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<data_t> urdis(-1, 1);
  for (int i = 0; i < INPUT_COUNT; i++) {
    vec.push_back(urdis(gen));
  }

  std::vector<data_t> host_res = softmax(vec);
  std::vector<data_t> ascend_res = ascend_softmax(vec);

  for (int i = 0; i < host_res.size(); ++i) {
    if (std::fabs(host_res[i] - ascend_res[i]) / host_res[i] > 0.02) {
      std::cout << "Calculation error." << std::endl;
      return EXIT_FAILURE;
    }
  }
  std::cout << "Result correct." << std::endl;

  return EXIT_SUCCESS;
}
```

## 性能测试

性能测试使用不同长度的输入向量进行，长度分别为640、6400、64000、640000，数据类型为float。Host端的时间统计使用`<time.h>`中的结构体，设备端的时间统计利用毕昇C++提供的`profiling`系列接口统计。仅统计算子计算逻辑部分的时间，并执行5次取平均值后计算加速比。之后的性能测试均为该策略，后面不再赘述。

性能测试结果如下。

| 测试用例 | 640     | 6400     | 64000    | 640000   |
| -------- | ------- | -------- | -------- | -------- |
| 加速比   | 0.18163 | 1.103832 | 2.968345 | 3.784732 |

可以看到在向量长度达到6400之后才勉强与Host端计算逻辑持平，虽然在长向量的情况下有速度上的提升，但远不是令人满意的效果。

# 求和方式优化（方案二）

在初步方案中，求和的过程是由Host端完成的，故接下来将求和也尽量的向量化。对于求和无法做到元素级别的并行，故只能将长向量拆分为多个短向量，这些短向量之间的求和操作是并行的，但单个短向量内部的求和，只能是串行的。

由于求和的每一项也需要经过指数运算，故可以把指数运算与求和放在同一个核函数内进行，计算完成后将指数运算结果向量存储下来，则后面做向量除法时就不必再运算一遍了。

总体的异构方案如图所示。

<img src="https://github.com/Deleter-D/Images/assets/56388518/50df7b9b-35fb-4b9c-8229-df21215ec566" style="zoom: 67%;" />

## 方案实现

实现代码如下，只需关注有注释的部分，没有注释的部分与上一方案中的代码相同。

```c++
std::vector<float> ascend_softmax(std::vector<float> input) {
  std::size_t input_sz = input.size();
  std::size_t byte_count = input_sz * sizeof(float);

  if (byte_count < 32)
    return softmax(input);

  const std::size_t elem_per_group = 640;
  const std::size_t tail_elem_count = input_sz % elem_per_group;
  const std::size_t group_num = (tail_elem_count > 0)
                                    ? ((input_sz / elem_per_group) + 1)
                                    : (input_sz / elem_per_group);
  // 此处计算出每个group所处理的repeat个数，方便vec_cross_add()接口调用
  const std::size_t repeat_per_group = (elem_per_group * sizeof(float)) / 256;

  sycl::queue Q(sycl::ascend_selector{});

  auto input_buf = sycl::malloc_device<float>(group_num * elem_per_group, Q);
  // 这里在GM上多申请两块内存，用于存放指数运算结果和求和结果
  auto exp_res_buf = sycl::malloc_device<float>(group_num * elem_per_group, Q);
  auto sum_res_buf = sycl::malloc_device<float>(group_num * repeat_per_group, Q);
  auto res_buf = sycl::malloc_device<float>(group_num * elem_per_group, Q);

  // 由于vec_cross_add是按repeat为单位进行求和的，故申请的向量即为group数量乘以每个group的repeat数量
  std::vector<float> sum_res((group_num * repeat_per_group), 0.0f);
  std::vector<float> res(input_sz, 0.0f);

  Q.memcpy(input_buf, input.data(), byte_count);

  // 第一个核函数进行求和与指数运算的操作
  Q.launch<class Summary>(group_num, [=](sycl::group<1> group) {
    bisheng::vector<float, elem_per_group> input_vec;
    bisheng::vector<float, elem_per_group> exp_res_vec;
    bisheng::vector<float, repeat_per_group> sum_res_vec;
    std::size_t group_id = group.get_group_id();

    input_vec.load(
        sycl::global_ptr<float>(input_buf + group_id * elem_per_group).get(),
        elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      // 同样的处理尾块中存在运算，且为最后一个group的情况
      bisheng::vector_view<float> input_vec_v(input_vec.data(), tail_elem_count);
      bisheng::vector_view<float> exp_res_vec_v(exp_res_vec.data(), tail_elem_count);

      bisheng::vec_exp(exp_res_vec_v, input_vec_v);
      // 由于尾块中的元素大概率不是整repeat，所以采用标量运算的的方式
      for (int i = 0; i < tail_elem_count; ++i)
        sum_res_vec[0] += exp_res_vec_v[i];
      for (int i = 1; i < repeat_per_group; ++i)
        sum_res_vec[i] = 0.0f;
    } else {
      // 整block情况
      bisheng::vec_exp(exp_res_vec, input_vec);
      // 这里不仅确定是整block，也能确定是整repeat，故直接调用接口
      bisheng::vec_cross_add(sum_res_vec.data(), exp_res_vec);
    }

    // UB -> GM 内存搬移，将指数运算结果与求和结果均保存下来
    exp_res_vec.store(
        sycl::global_ptr<float>(exp_res_buf + group_id * elem_per_group).get(),
        elem_per_group);
    sum_res_vec.store(
        sycl::global_ptr<float>(sum_res_buf + group_id * repeat_per_group).get(),
        repeat_per_group);
  });

  Q.memcpy(sum_res.data(), sum_res_buf, group_num * repeat_per_group * sizeof(float));
  Q.wait();

  // 由于vec_cross_add求和后的结果是多个短向量的和
  // 依然是一个向量，故须在Host端进一步计算为标量
  float sum;
  for (auto x : sum_res)
    sum += x;
  
  // 第二个核函数进行向量除法的运算
  Q.launch<class Softmax>(group_num, [=](sycl::group<1> group) {
    // 只需将上个核函数计算到的指数运算结果向量搬移进来即可
    bisheng::vector<float, elem_per_group> exp_res_vec;
    bisheng::vector<float, elem_per_group> divisor_vec(sum);
    bisheng::vector<float, elem_per_group> res_vec;
    std::size_t group_id = group.get_group_id();

    exp_res_vec.load(
        sycl::global_ptr<float>(exp_res_buf + group_id * elem_per_group).get(),
        elem_per_group);

    // 此处分支大同小异，不再赘述
    if (tail_elem_count > 0 && group_id == group_num - 1) {
      bisheng::vector_view<float> exp_res_vec_v(exp_res_vec.data(), tail_elem_count);
      bisheng::vector_view<float> divisor_vec_v(divisor_vec.data(), tail_elem_count);
      bisheng::vector_view<float> res_vec_v(res_vec.data(), tail_elem_count);

      bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
    } else {
      bisheng::vec_div(res_vec, exp_res_vec, divisor_vec);
    }

    res_vec.store(
        sycl::global_ptr<float>(res_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  Q.memcpy(res.data(), res_buf, byte_count);
  Q.wait();

  sycl::free(input_buf, Q);
  sycl::free(exp_res_buf, Q);
  sycl::free(sum_res_buf, Q);
  sycl::free(res_buf, Q);

  return res;
}
```

## 功能验证

功能验证与上述方式相同，但在验证过程中出现了较为严重的问题。

在代码执行过程中，出现了计算结果时对时错的情况，在之前的项目开发过程中其实是踩过这样的坑的，所以出现这种情况时也没有特别慌张。这里分享一下定位问题的心路历程，总结一下设备端代码Debug的思路。

由于对毕昇C++和毕昇编译器这套逻辑了解不够充分，所以可能Debug的方式很笨，但只要能De出来Bug，那就是好方法。

- 首先可以确定的是，计算向量除法的部分一定没有问题，这是方案一里面验证过的。
- 其次是要确定访存过程中是否存在问题，是不是访问到了一些不该访问的地方。确定访存无误后再进行下一步。
- 将每一步的计算结果输出，具体查看到底哪一步计算出现了错误。

首先分析访存的问题，可以先将输入向量的长度和数据类型确定下来，然后带入这个向量长度计算每一次访存的范围。当然你也可以写个脚本来帮你完成这一步，但我懒，我选择草稿纸。一波计算后发现，访存并没有什么问题，每一步操作访问的范围也都是它们应该访问的，并没有访问的未定义数据。那么基本可以确定，这个Bug不是我自己的原因，那就看看每一步的计算结果。

因为第二个核函数已经经过了方案一的验证，所以没有过多纠结，分析第一个核函数。第一个核函数进行了两种运算，指数运算和求和运算。但指数运算也在方案一里验证过了，是没有问题的，所以直接就将问题定位在了求和过程中。使用同一个输入向量，分别输出Host算子中的和与`vec_cross_add()`计算的和。当然这里输出的和均为标量，`vec_cross_add()`返回的结果已经在Host端相加计算为了标量。

```
[Debug]: Host sum: 7550.05
[Debug]: Ascend sum: 7090.52
[Error]: Calculation error.
```

```
[Debug]: Host sum: 7549.67
[Debug]: Ascend sum: 7549.66
[Debug]: Result correct.
```

果然！是求和出现了问题，而且是时对时错的。然后有对这个问题进行了更详细的测试，主要是测试了两种情况，也即第一个核函数中的两个分支。

由于尾块是采用`for`循环计算的，理论上不会出现错误，但为了严谨还是进行了一些测试。将向量长度锁定在320，迫使它只执行尾块的逻辑，结果如下。

```
[Debug]: Host sum: 387.095
[Debug]: Ascend sum: 387.095
[Debug]: Result correct.
```

```
[Debug]: Host sum: 356.134
[Debug]: Ascend sum: 356.134
[Debug]: Result correct.
```

无论执行多少次，结果都是正确的。~~那现在基本可以确定是`vec_cross_add()`接口出现了问题。所以我们对代码进行修改，将`vec_cross_add()`接口用`for`循环代替。~~

> 此处的结论不正确，`vec_cross_add()`接口本身没有任何问题，详细测试及Softmax的重新实现详见[关于vec_cross_add接口的详细测试 - 亦初 (deleter-d.github.io)](https://deleter-d.github.io/posts/1040/)

由于使用标量运算，故每个group求和的结果不再是向量，而是标量。所以存放求和结果的内存空间大小需要做一定的调整，这里申请大小为`group_num * (32 / sizeof(data_t))`的空间。其实理论上，每个group求和的结果只需要一个`data_t`数据类型的大小即可，但为了按照block为粒度严格分离group的访存空间，所以申请了与`group_num`个block大小相同的内存空间来存放。

```c++
auto sum_res_buf = sycl::malloc_device<data_t>(group_num * (32 / sizeof(data_t)), Q);
```

具体求和过程则改为如下方式。

```c++
if (tail_elem_count > 0 && group_id == group_num - 1) {
      bisheng::vector_view<data_t> input_vec_v(input_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> exp_res_vec_v(exp_res_vec.data(), tail_elem_count);

      bisheng::vec_exp(exp_res_vec_v, input_vec_v);
      for (int i = 0; i < tail_elem_count; ++i)
        sum_res_buf[group_id * (32 / sizeof(data_t))] += exp_res_vec_v[i];
    } else {
      bisheng::vec_exp(exp_res_vec, input_vec);
      for (int i = 0; i < elem_per_group; ++i) {
        sum_res_buf[group_id * (32 / sizeof(data_t))] += exp_res_vec[i];
      }
    }
```

进一步测试后，时对时错的问题解决了，到这里就可以说功能验证正确了，可喜可贺！

## 性能测试

性能测试结果如下。

| 测试用例 | 640      | 6400     | 64000    | 640000   |
| -------- | -------- | -------- | -------- | -------- |
| 加速比   | 0.237432 | 1.478988 | 13.19605 | 87.72573 |

对比方案一可以观察到，对求和进行向量化的意义是非常大的，尤其是向量长度变得越来越长后，这种优化的提升尤为明显。现在的加速比可以说是令人比较满意的了。

# 空间上的优化（方案三）

通过观察方案二中的数据流动，我们可以发现，在空间利用上有些浪费的地方。先来看一下方案二的数据流动方式。

<img src="https://github.com/Deleter-D/Images/assets/56388518/4aa99b79-7b82-4ee9-9b04-a944b0d3187c" style="zoom: 50%;" />

> 注：图中只描述了GM与UB之间的数据流，其中还发生了GM与Host之间的数据搬移。例如求和结果将会搬回Host，计算为标量后，利用该标量对分母向量进行初始化。

观察上述数据流可以发现，在使用`exp_res_buf`存储指数运算结果的时候，输入`input_buf`已经失去了作用，且后面也不会再使用其中的数据。同理，在使用`res_buf`存储最终结果的时候，`exp_res_buf`也不再使用了，因为指数运算结果此时已经读入了UB中。所以，`input_buf`、`exp_res_buf`和`res_buf`三者是可以合一的。

继续对UB中的内存使用进行分析。

<img src="https://github.com/Deleter-D/Images/assets/56388518/e8638370-1802-4f21-ac61-b8229ee80851" style="zoom:50%;" />

初步的理论分析可知，Kernel 1中的`input`和`exp_res_vec`可以合一，Kernel 2中的`exp_res_vec`与`res_vec`可以合一。我们在此过程中使用了`vec_exp(dst, src)`和`vec_div(dst, src0, src1)`接口，这两个接口分别为一元运算和二元运算。

在毕昇C++中，对于基于`bisheng::vector`类型的通用一元运算函数接口，`dst`和`src`可以是同一个`bisheng::vector`对象，即原址计算。故Kernel 1中的`input`和`exp_res_vec`可以合一。而对于二元运算，目标数据和源数据在不同的repeat迭代之间不允许出现地址重叠，虽然有部分接口例外，但我们所使用的`vec_div`接口并不在这些例外中，故无法将Kernel 2中的`exp_res_vec`与`res_vec`合一。

经过上述一系列空间优化，最终的数据流如图所示。

<img src="https://github.com/Deleter-D/Images/assets/56388518/ac3a81be-bd23-4c3c-8c98-7135a0f3e78e" style="zoom:50%;" />

## 方案实现

代码与方案二几乎一致，只是改变了内存搬移的源地址与目的地址，这里就不再放代码了。

## 功能验证

经过测试，功能验证正确。

## 性能测试

性能测试结果如下。

| 测试用例 | 640      | 6400     | 64000    | 640000   |
| -------- | -------- | -------- | -------- | -------- |
| 加速比   | 0.224613 | 1.512394 | 13.30433 | 88.70575 |

可以观察到，与方案二相比，时间上几乎没有区别。但由于优化了空间利用率，所以使得设备端可以承载更大长度的向量，优化的意义是比较大的。

# 分核方案的优化（方案四）

> 下面所有的讨论均已float类型的数据为例。

分核方案的核心思想就是，尽可能利用所有物理核心，并在此基础上令每个核心处理尽可能多的数据。而我们上面采用的方案是临时将每个核处理的元素数量固定为640，这显然不是最优的方案。

首先是尽可能利用所有的物理核心，昇腾910拥有32个物理核心，所以我们要想办法让32个核心都在工作状态，尽量避免一核干活儿，多核围观的滑稽场景。

首先分析一个问题，逻辑核的数量如何确定？假设输入向量长度为$len$，每个逻辑核处理的运算个数为$n$，在不考虑有尾块的情况下，可以得到逻辑核数量的公式为$group\_num=len\div n$。$n$由用户指定，不是我们可以控制的，故我们只能在$len$和$group\_num$上做文章。为了更好的理解，我们变形一下公式$len=group\_num\times n$。这样就可以比较直观的看出，我们需要在$group\_num$和$n$之间做一个权衡。

这个权衡只有两种思考方式：

- 一种是确定$group\_num$，根据$group\_num$计算得到$n$。说人话就是把逻辑核的数量定死，然后根据用户给的向量长度计算每个核要处理的元素个数。
- 另一种是确定$n$，即把每个逻辑核要处理的元素数量定死，然后根据用户给的向量长度计算逻辑核数量。

## 情况一

先来考虑第一种情况，即将$group\_num$定死。假设我们就定为与物理核数相同的数量，即32。考虑一个问题，假设输入向量长度非常长，那么拆分成32份后依然非常长，长到$len\div 32$个元素的大小超出了UB的承载范围，那么此时算子就会崩溃。

这时候有人就要说了（假装有人要说）：那不能把逻辑核数量写大一点吗？

好！听你的，我们将逻辑核数量定为320，理想状态下，每个物理核将处理10个逻辑核。此时再考虑一种情况，用户给的输入向量非常短，短到没办法分为320份，此时$len\div 32$为0。意味着你的每个逻辑核中，要么是处理尾块，要么根本就没有元素，但320个逻辑核依然会开启。这显然是不够合理的。

## 情况二

再来考虑将$n$定死的情况，即将每个逻辑核要处理的元素个数定死，其实就是我们上面方案的使用的策略，这里我们暂时考虑`n`为640的情况。同样考虑一些比较极端的例子，假设输入向量非常长，此时$len\div n$即$len\div 640$会非常大，即逻辑核的数量会非常多。虽然这样能够充分利用所以物理核，但每个逻辑核的承载能力远不止640个元素，这样就浪费了单个逻辑核的能力，把资源都消耗在调度逻辑核上了。

这时候又有人要说了（依然假装有人说）：那不能把逻辑核处理的元素个数写大一点吗？

好！还是听你的，我们将$n$定为UB能够承载的上限$max$。这种情况下，我们甚至不用考虑输入向量长度非常短的情况，只考虑向量长度小于$max\times 31$的情况，即向量长度小于31个物理核同时工作时可以处理的最大元素个数。此时至少会有1个物理核心在看戏，若向量长度进一步缩短，那看戏的物理核只会越来越多。这显然也是不够合理的。

## 动态方案

分析完两种情况，可以得出一个结论，单纯的确定$n$与$group\_num$中的任何一个都是不合适的。

那我们应该怎么确定呢？动态确定！

首先确定一个问题，我们这个算子，每个group处理多少数据是UB的上限。经过测试，每个group最多可以处理87360字节的数据，即$87360\div \text{sizeof}(data\_t)$个元素。这个上限并不是所有算子都一样的，因为每个算子在UB上申请内存的情况不同，所以要具体问题具体分析。

我们继续上面那个公式$len = group\_num\times n$，这里为了通用性，我们换成处理的字节数$bytes$，而不是元素个数。进而$n=bytes\div\text{sizeof}(data\_t)$，那么公式变为$total\_bytes=len\times\text{sizeof}(data\_t) = group\_num\times (bytes\div\text{sizeof}(data\_t))$。

- 当$total\_bytes< 32\times 2560$时，将$bytes$定为1280，则$group\_num<32$，此时算子可能无法充分利用所有物理核。
- 当$total\_bytes\ge 32\times 2560$时，将$bytes$定为2560，则$group\_num\ge 32$，这意味着算子将充分利用所有物理核。
- 当$total\_bytes\ge 32\times 5120$时，将$bytes$定为5120，则$group\_num\ge 32$，也会充分利用所有物理核。
- 当$total\_bytes\ge 32\times 12800$时，将$bytes$定为12800，则$group\_num\ge 32$，也会充分利用所有物理核。
- 当$total\_bytes\ge 32\times 25600$时，将$bytes$定为25600，则$group\_num\ge 32$，同样充分利用所有物理核。
- 当$total\_bytes\ge 32\times 51200$时，将$bytes$定为51200，则$group\_num\ge 32$，同样充分利用所有物理核。
- 当$total\_bytes\ge 32\times 87360$时，将$bytes$定为87360，则$group\_num\ge 32$，同样充分利用所有物理核。

采取这种策略，虽然在输入向量总字节数小/于$32\times2560$时可能会出现某些物理核不工作的情况，但考虑到实际情况下，输入向量都是比较长的向量，这种偶尔的空闲是可接受的。

这里只是展示一种思路，条件分支中的阈值是可以根据实际情况进行调整的，并不是只能按照2560、5120等阈值进行分割。

## 方案实现

同样还是注意带注释的地方，其余地方与之前相同。

```c++
std::vector<data_t> ascend_softmax(std::vector<data_t> input) {
  std::size_t input_sz = input.size();
  std::size_t byte_count = input_sz * sizeof(data_t);

  if (byte_count < 32)
    return softmax(input);

  // 这里依照上面介绍的策略确定每个逻辑核所处理的元素个数
  std::size_t elem_per_group = 0;
  if (byte_count >= PHYSICAL_CORES * UB_MAX_BYTES)
    elem_per_group = UB_MAX_BYTES / sizeof(data_t);
  else if (byte_count >= PHYSICAL_CORES * 51200)
    elem_per_group = 51200 / sizeof(data_t);
  else if (byte_count >= PHYSICAL_CORES * 25600)
    elem_per_group = 25600 / sizeof(data_t);
  else if (byte_count >= PHYSICAL_CORES * 12800)
    elem_per_group = 12800 / sizeof(data_t);
  else if (byte_count >= PHYSICAL_CORES * 5120)
    elem_per_group = 5120 / sizeof(data_t);
  else if (byte_count >= PHYSICAL_CORES * 2560)
    elem_per_group = 2560 / sizeof(data_t);
  else
    elem_per_group = 1280 / sizeof(data_t);

  const std::size_t tail_elem_count = input_sz % elem_per_group;
  const std::size_t group_num = (tail_elem_count > 0)
                                    ? ((input_sz / elem_per_group) + 1)
                                    : (input_sz / elem_per_group);

  sycl::queue Q(sycl::ascend_selector{});

  auto dev_buf = sycl::malloc_device<data_t>(group_num * elem_per_group, Q);
  auto sum_res_buf = sycl::malloc_device<data_t>(group_num * (32 / sizeof(data_t)), Q);

  std::vector<data_t> sum_res(group_num * (32 / sizeof(data_t)), 0.0f);
  std::vector<data_t> res(input_sz, 0.0f);

  Q.memcpy(dev_buf, input.data(), byte_count);

  Q.launch<class Summary>(group_num, [=](sycl::group<1> group) {
    // 此处直接申请最大空间，因为定义毕昇向量时指定大小必须用常量表达式，大小需要在编译时确定
    // 由于前面采用了动态策略，所以不能直接使用elem_per_group来定义毕昇向量
    // 只需要在使用时控制访存范围即可，第二个核函数同理，不再赘述
    bisheng::vector<data_t, UB_MAX_BYTES / sizeof(data_t)> input_vec;
    std::size_t group_id = group.get_group_id();

    input_vec.load(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      bisheng::vector_view<data_t> input_vec_v(input_vec.data(), tail_elem_count);

      bisheng::vec_exp(input_vec_v, input_vec_v);
      for (int i = 0; i < tail_elem_count; ++i)
        sum_res_buf[group_id * (32 / sizeof(data_t))] += input_vec_v[i];
    } else {
      // 由于毕昇向量定义了最大长度，故即使是整block的情况，也需要用变长向量来控制访存范围
      bisheng::vector_view<data_t> input_vec_v(input_vec.data(), elem_per_group);
      bisheng::vec_exp(input_vec_v, input_vec_v);
      for (int i = 0; i < elem_per_group; ++i) {
        sum_res_buf[group_id * (32 / sizeof(data_t))] += input_vec_v[i];
      }
    }

    input_vec.store(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  Q.memcpy(sum_res.data(), sum_res_buf, group_num * (32 / sizeof(data_t)) * sizeof(data_t));
  Q.wait();

  data_t sum;
  for (int i = 0; i < sum_res.size(); i += 32 / sizeof(data_t))
    sum += sum_res[i];

  Q.launch<class Softmax>(group_num, [=](sycl::group<1> group) {
    bisheng::vector<data_t, UB_MAX_BYTES / sizeof(data_t)> exp_res_vec;
    bisheng::vector<data_t, UB_MAX_BYTES / sizeof(data_t)> divisor_vec(sum);
    bisheng::vector<data_t, UB_MAX_BYTES / sizeof(data_t)> res_vec;
    std::size_t group_id = group.get_group_id();

    exp_res_vec.load(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);

    if (tail_elem_count > 0 && group_id == group_num - 1) {
      bisheng::vector_view<data_t> exp_res_vec_v(exp_res_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> divisor_vec_v(divisor_vec.data(), tail_elem_count);
      bisheng::vector_view<data_t> res_vec_v(res_vec.data(), tail_elem_count);

      bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
    } else {
      bisheng::vector_view<data_t> exp_res_vec_v(exp_res_vec.data(), elem_per_group);
      bisheng::vector_view<data_t> divisor_vec_v(divisor_vec.data(), elem_per_group);
      bisheng::vector_view<data_t> res_vec_v(res_vec.data(), elem_per_group);
      bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
    }

    res_vec.store(
        sycl::global_ptr<data_t>(dev_buf + group_id * elem_per_group).get(),
        elem_per_group);
  });

  Q.memcpy(res.data(), dev_buf, byte_count);
  Q.wait();

  sycl::free(dev_buf, Q);
  sycl::free(sum_res_buf, Q);

  return res;
}
```

## 功能测试

功能测试验证正确。

## 性能测试

性能测试结果如下。

| 测试用例 | 640      | 6400     | 64000    | 640000   |
| -------- | -------- | -------- | -------- | -------- |
| 加速比   | 0.234373 | 1.436941 | 12.35908 | 80.59147 |

分析了许多，本想着动态分核结果会有惊喜。嘿！您猜怎么着？还真是大惊喜！

在动态分核的策略下，当向量长度总字节数不少于$32\times 2560$时，总能保证32个物理核都在工作，而且不至于令逻辑核数量过多，但神奇的事情来了，这种策略成功实现了负优化！！

本方案的性能测试结果看起来还不错，但我们继续增大向量长度，使得总字节数到达划分策略的阈值附近。以`float`类型的数据为例，令向量长度为698880，此时总字节数为$698880\times 4=2795520=87360\times 32$。此时这种策略将分配32个逻辑核，完美贴合物理核数量，每个核心处理87360字节的数据，完美贴合UB承载的上限。惊喜的事情来了，请看加速比

| 测试用例     | 698880   |
| ------------ | -------- |
| 方案三加速比 | 96.29706 |
| 方案四加速比 | 67.26953 |

什么鬼情况？！？！

我们分别观察一下它们的分核情况。

方案三如下：

```
[PERMORMANCE]: Host time cost: 93873327 ns
[Debug]: Group num: 1875 Elements per group: 640
[PERMORMANCE]: Ascend time cost: 728001 ns
```

方案四如下：

```
[PERMORMANCE]: Host time cost: 94127478 ns
[Debug]: Group num: 55 Elements per group: 21840
[PERMORMANCE]: Ascend time cost: 785999 ns
```

可以发现两者都充分利用了所有物理核，但方案四的策略使得加速比下降了，反观方案三的1875个逻辑核取得了完胜。但转念一想，是不是让每个逻辑核承载到UB的上限有点过分，那么再来测试一下正常压力下的表现。

还是以`float`类型数据为例，向量长度为102400，此时总字节数为$102400\times 4=409600=12800\times 32$，此时将分配32个逻辑核，每个核处理12800字节的数据，远不到UB承载的上限。

方案三分核情况如下：

```
[PERMORMANCE]: Host time cost: 7872144 ns
[Debug]: Group num: 160 Elements per group: 640
[PERMORMANCE]: Ascend time cost: 375999 ns
```

方案四分核情况如下：

```
[PERMORMANCE]: Host time cost: 7813204 ns
[Debug]: Group num: 32 Elements per group: 3200
[PERMORMANCE]: Ascend time cost: 400999 ns
```

加速比如下：

| 测试用例     | 102400   |
| ------------ | -------- |
| 方案三加速比 | 20.73252 |
| 方案四加速比 | 19.4274  |

依然是有略微的下降，这也排除了UB压力过大的问题。

# 结论

经过一系列分析，目前能够得出的结论是，尽可能多的逻辑核数量的收益要大于单逻辑核内处理尽可能多的数据。

异构分核的坑还是太多了，踩都踩不完，过程中有很多反直觉的情况，必须靠实验来佐证。

# 完整代码

最后贴上目前效果最好（方案三）的完整代码，其中包括一些自定义的Debug信息，不用太纠结。

```c++
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <bisheng/bisheng.hpp>
#include <sycl/sycl.hpp>

#define DEBUG
#define DEBUG_HEAD "\033[34m[Debug]: \033[0m"
#define ERROR_HEAD "\033[31m[Error]: \033[0m"
#define PERFORMANCE
#define PERFORMANCE_HEAD "\033[36m[PERMORMANCE]: \033[0m"
#define INPUT_COUNT 102400

std::vector<float> softmax(std::vector<float> input) {
#ifdef DEBUG
  std::cout << DEBUG_HEAD << "The host operator is called.\n";
#endif

#ifdef PERFORMANCE
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  auto start_time = time.tv_sec * 1000000000 + time.tv_nsec;
#endif

  float sum = 0.0;
  for (auto x : input) {
    sum += expf(x);
  }

#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Host sum: " << sum << "\n";
#endif

  std::vector<float> res;
  for (auto x : input) {
    res.push_back(expf(x) / sum);
  }

#ifdef PERFORMANCE
  clock_gettime(CLOCK_REALTIME, &time);
  auto end_time = time.tv_sec * 1000000000 + time.tv_nsec;
  std::cout << PERFORMANCE_HEAD << "Host time cost: " << end_time - start_time
            << " ns" << std::endl;
#endif

  return res;
}

std::vector<float> ascend_softmax(std::vector<float> input) {
  std::size_t input_sz = input.size();
  std::size_t byte_count = input_sz * sizeof(float);

  // call the host operator if input isn't enough a full block
  if (byte_count < 32) {
#ifdef DEBUG
    std::cout << DEBUG_HEAD
              << "The input vector is not enough for a full block.\n";
#endif
    return softmax(input);
  }

  // ascend code start
#ifdef DEBUG
  std::cout << DEBUG_HEAD << "The ascend operator is called.\n";
#endif

  // number of elements per group
  const std::size_t elem_per_group = 640;
  // number of elements in tail block
  const std::size_t tail_elem_count = input_sz % elem_per_group;
  // number of groups
  // if tail block is exist, apply for one more group
  const std::size_t group_num = (tail_elem_count > 0)
                                    ? ((input_sz / elem_per_group) + 1)
                                    : (input_sz / elem_per_group);

#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Group num: " << group_num
            << " Elements per group: " << elem_per_group << "\n";
#endif

  sycl::queue Q(sycl::ascend_selector{}, nullptr,
                {sycl::property::queue::enable_profiling()});

  // GM memory allocation
  auto dev_buf = sycl::malloc_device<float>(group_num * elem_per_group, Q);
  auto sum_res_buf =
      sycl::malloc_device<float>(group_num * (32 / sizeof(float)), Q);

  // Host memory allocation
  std::vector<float> sum_res(group_num * (32 / sizeof(float)), 0.0f);
  std::vector<float> res(input_sz, 0.0f);

  // host -> GM
  Q.memcpy(dev_buf, input.data(), byte_count);

#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Kernel function started.\n";
#endif

  sycl::event e0 =
      Q.launch<class Summary>(group_num, [=](sycl::group<1> group) {
        bisheng::vector<float, elem_per_group> input_vec;
        std::size_t group_id = group.get_group_id();

        // GM -> UB
        input_vec.load(
            sycl::global_ptr<float>(dev_buf + group_id * elem_per_group).get(),
            elem_per_group);

        if (tail_elem_count > 0 && group_id == group_num - 1) {
          // if tail block has element and this is the last group
          bisheng::vector_view<float> input_vec_v(input_vec.data(),
                                                  tail_elem_count);

          bisheng::vec_exp(input_vec_v, input_vec_v);
          for (int i = 0; i < tail_elem_count; ++i)
            sum_res_buf[group_id * (32 / sizeof(float))] += input_vec_v[i];
        } else {
          // full block data
          bisheng::vec_exp(input_vec, input_vec);
          for (int i = 0; i < elem_per_group; ++i) {
            sum_res_buf[group_id * (32 / sizeof(float))] += input_vec[i];
          }
        }

        // UB -> GM
        input_vec.store(
            sycl::global_ptr<float>(dev_buf + group_id * elem_per_group).get(),
            elem_per_group);
      });

  // GM -> Host
  Q.memcpy(sum_res.data(), sum_res_buf,
           group_num * (32 / sizeof(float)) * sizeof(float));
  Q.wait();

  float sum;
  for (int i = 0; i < sum_res.size(); i += 32 / sizeof(float))
    sum += sum_res[i];
#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Ascend sum: " << sum << "\n";
#endif

  sycl::event e1 =
      Q.launch<class Softmax>(group_num, [=](sycl::group<1> group) {
        // UB memory of exponent result
        bisheng::vector<float, elem_per_group> exp_res_vec;
        // UB memory of divisor
        bisheng::vector<float, elem_per_group> divisor_vec(sum);
        // UB memory of final result
        bisheng::vector<float, elem_per_group> res_vec;
        std::size_t group_id = group.get_group_id();

        // GM -> UB
        exp_res_vec.load(
            sycl::global_ptr<float>(dev_buf + group_id * elem_per_group).get(),
            elem_per_group);

        if (tail_elem_count > 0 && group_id == group_num - 1) {
          // if tail block has element and this is the last group
          bisheng::vector_view<float> exp_res_vec_v(exp_res_vec.data(),
                                                    tail_elem_count);
          bisheng::vector_view<float> divisor_vec_v(divisor_vec.data(),
                                                    tail_elem_count);
          bisheng::vector_view<float> res_vec_v(res_vec.data(),
                                                tail_elem_count);

          bisheng::vec_div(res_vec_v, exp_res_vec_v, divisor_vec_v);
        } else {
          // full block data
          bisheng::vec_div(res_vec, exp_res_vec, divisor_vec);
        }

        // UB -> GM
        res_vec.store(
            sycl::global_ptr<float>(dev_buf + group_id * elem_per_group).get(),
            elem_per_group);
      });

#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Kernel function finished.\n";
#endif

  // GM -> host
  Q.memcpy(res.data(), dev_buf, byte_count);
  Q.wait();

  sycl::free(dev_buf, Q);
  sycl::free(sum_res_buf, Q);

  // ascend code end

#ifdef PERFORMANCE
  const uint64_t e0_start_time =
      e0.get_profiling_info<sycl::info::event_profiling::command_start>();
  const uint64_t e0_end_time =
      e0.get_profiling_info<sycl::info::event_profiling::command_end>();
  const uint64_t e1_start_time =
      e1.get_profiling_info<sycl::info::event_profiling::command_start>();
  const uint64_t e1_end_time =
      e1.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << PERFORMANCE_HEAD << "Ascend time cost: "
            << (e0_end_time - e0_start_time) + (e1_end_time - e1_start_time)
            << " ns" << std::endl;
#endif

  return res;
}

int main() {
#ifdef DEBUG
  std::cout << DEBUG_HEAD << "Compile succeed" << std::endl;
#endif

  std::vector<float> vec;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> urdis(-1, 1);
  for (int i = 0; i < INPUT_COUNT; i++) {
    vec.push_back(urdis(gen));
  }

  std::vector<float> host_res = softmax(vec);
  std::vector<float> ascend_res = ascend_softmax(vec);

  for (int i = 0; i < host_res.size(); ++i) {
    if (std::fabs(host_res[i] - ascend_res[i]) / host_res[i] > 0.02) {
      std::cout << ERROR_HEAD << "Calculation error." << std::endl;
      return EXIT_FAILURE;
    }
  }
  std::cout << DEBUG_HEAD << "Result correct." << std::endl;

  return EXIT_SUCCESS;
}
```

