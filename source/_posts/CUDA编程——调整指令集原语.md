---
title: CUDA编程——调整指令集原语
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 53610
date: 2024-02-20 16:45:21
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## CUDA指令概述

### 浮点指令

> 一些前值知识：
>
> - 浮点型数值无法精确存储，只能在四舍五入后再存储；
> - 浮点数存在粒度问题，即浮点数只能在离散的区间间隔里存储数据。随着浮点数离零越来越远，表示数值的区间将随之增大；
> - C语言中的数学函数`nextafterf()`可以从给定值找到下一个最高位浮点数。

在浮点数值上进行操作的指令被成为浮点指令。CUDA支持所有在浮点数上常见的算术元算。CUDA遵循IEEE-754标准，支持32位和64位两种浮点精度。所有CUDA设备都支持单精度，计算能力1.3及以上的设备均支持双精度。

### 内部函数和标准函数

CUDA将所有算术函数分为内部函数和标准函数。

- 标准函数：可对主机和设备进行访问并标准化主机和设备的操作；
- 内部函数：只能对设备代码进行访问，在编译时对内部函数的行为会有特殊响应，从而产生更积极的优化和更专业化的指令生成。

在CUDA中，很多内部函数和标准函数是有关联的，存在着与内部函数功能相同的标准函数。如标准函数`sqrt()`对应的内部函数是`__dsqrt_rn()`。内部函数分解成了比与它们等价的标准函数更少的指令。这会导致内部函数比等价的标准函数更快，但数值精度更低。

### 原子操作指令

CUDA提供了在32位或64位全局内存或共享内存上执行“读-改-写”操作的原子函数。所有计算能力1.1及以上的设备都支持原子操作。

与标准函数和内部函数类似，每个原子函数都能实现一个基本的数学运算。不同于其他指令类型，当原子操作指令在两个竞争线程共享的内存空间进行操作时，会有一个定义好的行为。

原子运算函数分为3种：

- 算术运算函数：在目标内存位置上执行简单的算术运算；
- 按位运算函数：在目标内存位置上执行按位操作；
- 替换函数：可以用一个新值来替换内存位置上原有的值，可以是有条件的也可以是无条件的，无论成功与否，原子替换函数均返回最初的值。
  - `atomicExch()`可以无条件的替换已有的值；
  - `atomicCAS()`可以有条件的替换已有的值。

虽然原子函数没有精度上的顾虑，但它们的使用可能会严重降低性能。

## 程序优化指令

### 单精度与双精度

这里进行一个简单的实验，在主机端和设备端分别将一个单精度浮点数和一个双精度浮点数的值设为12.1，分别对比两种精度的浮点数。

```
Host single-precision representation of 12.1   = 12.10000038146972656250
Host double-precision representation of 12.1   = 12.09999999999999964473
Device single-precision representation of 12.1 = 12.10000038146972656250
Device double-precision representation of 12.1 = 12.09999999999999964473
Device and host single-precision representation equal? yes
Device and host double-precision representation equal? yes
```

> 详细代码参考[floating_point_accuracy.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/01_floating_point_accuracy.cu)。

可以发现，主机和设备上的数值都是近似于12.1，都不是精确值。这个例子中，双精度数值比单精度数值更接近于真实值。

双精度数值的精确性是以空间和性能消耗为代价的。这里再进行一个简单的实验进行验证。将一批单精度和双精度浮点数置于GPU中进行大量的数学运算，再将结果搬移回主机。

```
Input   Diff Between Single- and Double-Precision
------  -----------------------------------------
0       3.13614849292207509279e-02
1       2.67553565208800137043e-03
2       2.57291377056390047073e-03
3       7.82136313500814139843e-03
4       3.38051875296514481306e-02
5       4.95682619221042841673e-02
6       1.57542112574446946383e-02
7       1.02473393344553187490e-02
8       1.06261099135736003518e-02
9       2.36870593798812478781e-02

For single-precision floating point, mean times for:
  Copy to device:   178.990894 ms
  Kernel execution: 39.176985 ms
  Copy from device: 673.705701 ms
For double-precision floating point, mean times for:
  Copy to device:   356.848785 ms (1.99x slower than single-precision)
  Kernel execution: 1922.416699 ms (49.07x slower than single-precision)
  Copy from device: 1347.894141 ms (2.00x slower than single-precision)
```

> 详细代码参考[floating_point_perf.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/02_floating_point_perf.cu)。

这个例子说明单精度和双精度浮点运算在通信和计算上的性能差异是不可忽略的。同时也说明了单精度与双精度的结果有较大的数值差异，这些结果可能在迭代过程中不断被积累，导致最终结果偏差很大。

由于双精度数值所占空间是单精度数值的两倍，所以当在寄存器中存储一个双精度数值时，一个线程块总的共享寄存器空间会比使用单精度浮点数小的多。

### 标准函数与内部函数

为了比较标准函数和内部函数的差异，我们需要将代码编译成PTX代码来观察生成的类汇编指令。以下面两个核函数为例。

```cpp
__global__ void standardKernel(float a, float* out)
{
	*out = powf(a, 2.0f);
}

__global__ void intrinsicKernel(float a, float* out)
{
	*out = powf(a, 2.0f);
}
```

在编译时加上`--ptx`选项可以将该代码编译成PTX代码。

> 详细代码参考[intrinsic_standard_comp.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/03_intrinsic_standard_comp.cu)，对应的PTX代码参考[intrinsic_standard_comp.ptx](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/03_intrinsic_standard_comp.ptx)。

通过观察二者的PTX代码，最为明显的一点就是，标准函数的PTX代码量要远大于内部函数。进一步测试其性能表现。

```
Host calculated                 18345290.000000
Standard Device calculated      18345290.000000
Intrinsic Device calculated     18345288.000000
Host equals Standard?           Yes diff=0.000000e+00
Host equals Intrinsic?          No diff=2.000000e+00
Standard equals Intrinsic?      No diff=2.000000e+00

Mean execution time for standard function powf:    0.249888 ms
Mean execution time for intrinsic function __powf: 0.070720 ms
```

观察精度及性能表现可以发现，内部函数的性能比标准函数要好，但精度不如标准函数。

虽然CUDA代码转化为GPU指令集这一过程通常是编译器完成的，但可以通过一些手段来引导编译器倾向于精度或性能，或两者的平衡。主要有两种方法可以引导指令级优化的类型：

- 编译器标志；
- 内部或标准函数调用。

内部或标准函数调用在上面的例子中已经体现了，下面介绍通过编译器标志来引导编译器的代码生成。

有如下核函数，实现一个乘加运算。

```cpp
__global__ void fmad(float* ptr)
{
    *ptr = (*ptr) * (*ptr) + (*ptr);
}
```

`nvcc`编译器中提供一个选项`--fmad`来控制乘加运算是否融合，默认情况下为`true`。观察上述核函数的PTX代码可以发现，乘加运算被编译器融合为了一个运算。这样可以提高性能，但精度会有所损失。

```
fma.rn.f32 	%f2, %f1, %f1, %f1;
```

接着使用`--fmad=false`编译同样的代码，PTX代码如下。

```
mul.rn.f32 	%f2, %f1, %f1;
add.rn.f32 	%f3, %f1, %f2;
```

> 详细代码参考[manipulation_instruction_generation.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/04_manipulation_instruction_generation.cu)，对应的PTX代码参考[manipulation_instruction_generation.ptx](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/04_manipulation_instruction_generation.ptx)。

类似于`--fmad`的控制指令生成的选项还有很多。

| 选项                 | 描述                                                         | 默认值  | 性能影响                                              | 精度影响                                              |
| -------------------- | ------------------------------------------------------------ | ------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `--ftz=[bool]`       | 将所有单精度非正规浮点数置为零。                             | `false` | `true`时可能会提高性能，具体取决于待处理的值和算法。  | `false`时可能会提高精度，具体取决于待处理的值和算法。 |
| `--prec-div=[bool]`  | 提高了所有单精度除法和倒数数值的精度。                       | `true`  | `true`时可能会降低性能。                              | `true`时可能会提高与IEEE标准数值的兼容性。            |
| `--prec-sqrt=[bool]` | 强制执行精度更高的平方根函数。                               | `true`  | `true`时可能会降低性能。                              | `true`时可能会提高与IEEE标准数值的兼容性。            |
| `--fmad=[bool]`      | 控制编译器是否将乘加运算融合到一个FMAD指令中。               | `true`  | 若程序中存在浮点型变量的MAD运算，启动FMAD会提高性能。 | 启用FMAD可能会降低精度。                              |
| `--use_fast_math`    | 用等价的内部函数替换程序中所有的标准函数。<br />同时设置`--ftz=true`、`--prec-div=false`和`--prec-sqrt=false`。 | `false` | 启用该选项则表明启动了一系列提高性能的优化。          | 启用该选项可能会降低精度。                            |

除了`--fmad`选项，CUDA还包含一对控制FMAD指令生成的内部函数：`__fmul`和`__dmul`。这些函数不会影响乘法运算的性能，在有`*`运算符的地方调用可以组织编译器将乘法作为乘加优化的一部分来使用。

还是之前的乘加例子，除了使用`--fmad=false`来阻止乘加优化外，还可以用下列方式来实现相同的效果。

```cpp
__global__ void fmadBlocked(float* ptr)
{
    *ptr = __fmul_rn((*ptr), (*ptr)) + (*ptr);
}
```

> 这里调用函数时，实际调用的是`__fmul_rn()`，这个后缀显式地表达了四舍五入的模式，具体如下表所示。
>
> | 后缀 | 含义                                                         |
> | ---- | ------------------------------------------------------------ |
> | `rn` | 在当前浮点模式（单或双精度）下不能精确表示的数值，用可表示的最近似值来表示。这是默认模式。 |
> | `rz` | 总是向零取整。                                               |
> | `ru` | 总是向上取整到正无穷。                                       |
> | `rd` | 总数向下取整到负无穷。                                       |

这样就可以通过这些内部函数的调用来单独控制某些计算的精度，提升某些数值的健壮性，从而可以全局启用MAD优化。

### 原子指令

#### 原子函数

一个很重要的操作就是原子级CAS，它可以令程序员在CUDA中自定义原子函数。CAS接受三个参数：

- 内存地址；
- 存储在此地址中的期望值；
- 实际想要存储在此位置的新值。

CAS的整体流程为：

- 读取目标地址并将其存储值与预期值进行比较。
  - 若存储值与预期值相等，则新值将存入目标位置；
  - 若存储值与预期值不等，则目标位置不发生变化；
- 无论发生什么情况，CAS总是返回目标地址的值。使用返回值可以检查是否替换成功。若返回值等于预期值，则CAS一定成功了。

下面借助CAS来实现一个原子加操作。

```cpp
__device__ int myAtomicAdd(int *address, int incr)
{
    int expected = *address;                                      // 记录当前内存地址的值
    int oldValue = atomicCAS(address, expected, expected + incr); // 尝试增加incr，CAS会返回目标地址的值

    while (oldValue != expected) // 如果返回值与预期值不等，则CAS没有成功
    {
        // 重复执行CAS直到成功
        expected = oldValue;                                      // 获取目标地址的新值
        oldValue = atomicCAS(address, expected, expected + incr); // 继续尝试增加incr
    }

    return oldValue; // 为了匹配其他CUDA原子函数的语义，这里返回目标地址的值
}
```

> 详细代码参考[atomic_operation.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/05_atomic_operation.cu)。

CUDA中内置了一系列的原子函数，如下表所示。

| 函数         | 操作       | 支持的数据类型                                     |
| ------------ | ---------- | -------------------------------------------------- |
| `atomicAdd`  | 加法       | `int, unsigned int, unsigned long long int, float` |
| `atomicSub`  | 减法       | `int, unsigned int`                                |
| `atomicExch` | 无条件替换 | `int, unsigned int, unsigned long long int, float` |
| `atomicMin`  | 最小值     | `int, unsigned int, unsigned long long int`        |
| `atomicMax`  | 最大值     | `int, unsigned int, unsigned long long int`        |
| `atomicInc`  | 增量       | `unsigned int`                                     |
| `atomicDec`  | 减量       | `unsigned int`                                     |
| `atomicCAS`  | CAS        | `int, unsigned int, unsigned long long int`        |
| `atomicAnd`  | 与         | `int, unsigned int, unsigned long long int`        |
| `atomicOr`   | 或         | `int, unsigned int, unsigned long long int`        |
| `atomicXor`  | 异或       | `int, unsigned int, unsigned long long int`        |

#### 原子操作的代价

原子操作可能会付出很高的代价：

- 当在全局或共享内存中执行原子操作时，能保证所有的数值变化对所有线程都是立即可见的。如果原子指令操作成功，则必须把实际需要的值写入到全局或共享内存中；
- 共享地址冲突的原子访问可能要求发生冲突的线程不断地重试；
- 当线程在同一个线程束中时必须执行不同的指令，线程束执行是序列化的。若一个线程束中的多个线程在相同的内存地址发出一个原子操作，就会产生类似于线程冲突的问题；

下面用一个例子来展示原子操作的代价。实现一个核函数不断累加一个共享变量，然后实现一个功能类似，但线程不安全的核函数。

```cpp
__global__ void atomics(int* shared_var, int* values_read, int size, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    values_read[tid] = atomicAdd(shared_var, 1);

    for (int i = 0; i < iterations; i++)
    {
        atomicAdd(shared_var, 1);
    }
}

__global__ void unsafe(int* shared_var, int* values_read, int size, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    int old          = *shared_var;
    *shared_var      = old + 1;
    values_read[tid] = old;

    for (int i = 0; i < iterations; i++)
    {
        old         = *shared_var;
        *shared_var = old + 1;
    }
}
```

> 详细代码参考[atomic_ordering.cu](https://github.com/Deleter-D/CUDA/blob/master/06_adjusting_instruction-level_primitives/06_atomic_ordering.cu)。

测试两个核函数，结果如下。

```
In total, 30 runs using atomic operations took 18.691618 ms
  Using atomic operations also produced an output of 6400064
In total, 30 runs using unsafe operations took 2.103617 ms
  Using unsafe operations also produced an output of 100001
Threads performing atomic operations read values 0 1 2 3 4 5 6 7 8 9
Threads performing unsafe operations read values 0 0 0 0 0 0 0 0 0 0
```

发现原子操作保证了数值的正确性，但性能却大幅度下降。

为了应对上述情况，可以使用局部操作来增强全局原子操作，这些局部操作能从同一线程块的线程中产生一个中间结果。这些操作必须是顺序无关的，即操作的顺序不应影响最终的结果。
