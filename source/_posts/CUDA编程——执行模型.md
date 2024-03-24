---
title: CUDA编程——执行模型
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 47225
date: 2024-02-20 16:02:16
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## CUDA运行时

`cudart`库是CUDA运行时的实现，该库可以通过`cudart.lib`或`libcudart.a`静态链接到程序中，也可以通过`cudart.dll`或`libcudart.so`动态链接。需要动态链接的该库的程序，通常将`cudart.dll`或`libcudart.so`作为程序安装包的一部分。

只有在链接到同一个CUDA运行时实例的组件之间传递CUDA运行时符号地址才是安全的。

该库的所有API均带有`cuda`前缀。

### 初始化

从CUDA 12.0开始，`cudaInitDevice()`和`cudaSetDevice()`调用会初始化与指定设备关联的运行时和主上下文。如不进行这些调用，运行时会隐式地使用设备0，并按需进行自初始化来执行其他运行时API请求。

在CUDA 12.0之前，`cudaSetDevice()`不会初始化运行时，程序通常使用空操作运行时调用`cudaFree(0)`，将运行时初始化与其他API活动隔离开。

运行时将会为每个设备创建一个CUDA上下文，称之为主上下文（primary context）。该上下文将在调用第一个需要活动上下文的运行时函数时被初始化。主机端的所有线程共享该上下文。在创建上下文过程中，必要情况下会将设备代码即时编译并加载到设备内存中

当主机线程调用`cudaDeviceReset()`时，将销毁该线程当前操作设备的主上下文。任何拥有该设备的主机线程进行下一个运行时函数调用时，将为该设备创建一个新的主上下文。

## CUDA执行模型概述

### GPU架构概述

GPU是围绕流式多处理器（SM）的可扩展阵列搭建的，通过复制这种架构的构建块来实现GPU的硬件并行。

SM的核心组件如下：

- CUDA核心；
- 共享内存 / 一级缓存；
- 寄存器文件；
- 加载 / 存储单元；
- 特殊功能单元；
- 线程束调度器。

每个SM可以支持数百个线程并发执行，每个GPU通常有多个SM，所以一个GPU上并发执行数千个线程是有可能的。启动一个核函数时，线程块被分配在了可用的SM上，线程块一旦被调度到一个SM上，其中的线程只会在当前的SM上执行。多个线程块可能被分配在同一个SM上，是根据SM资源的可用性进行调度的。同一线程值的指令利用指令级并行性进行流水线化。

CUDA采用单指令多线程（SIMT）架构来管理和执行线程，每32个线程为一组，成为线程束（warp）。

在并行线程中共享数据可能会引起竞争，CUDA提供了一种用来同步线程块内线程的方法，但没有提供块间同步的原语。

虽然线程块内的线程束可以任意顺序调度，但活跃的线程束仍会受到SM资源的限制。当线程束闲置时，SM可以从同一SM上的常驻线程块中调度其他可用的线程束。在并发的线程束之间切换没有开销，因为硬件资源已经被分配到了SM上的所有线程和块中。

## 线程束的本质

### 线程束和线程块

线程束是SM中基本的执行单元。一旦线程块被调度到一个SM上，线程块中的线程会被进一步划分为线程束。一个线程束由32个连续的线程组成，在一个线程束中，所有的线程按照SIMT方式执行。

虽然线程块可以组织为一维、二维或三维的，但从硬件角度看，所有线程都被组织成了一维的。例如有一个128线程的一维线程块，它将被组织进4个线程束中，如下所示。

```
Warp 0: thread  0, thread  1, thread  2, ... thread 31
Warp 1: thread 32, thread 33, thread 34, ... thread 63
Warp 2: thread 64, thread 65, thread 66, ... thread 95
Warp 3: thread 96, thread 97, thread 98, ... thread 127
```

二维、三维的线程块是同理的，只需要计算出其唯一线程ID即可。

一个线程块的线程束数量由下式确定。
$$
\text{block}中的\text{warp}数量=\left\lceil\frac{\text{block}中的\text{thread}数量}{\text{warp}大小}\right\rceil
$$
线程束不会在不同的线程块之间分离，若线程块的大小不是线程束大小的整数倍，则在最后的线程束中会有些线程处于不活跃状态。例如有一个二维的$40\times2$的线程块，他会被分配在3个线程束中，最后一个线程束的后半段是不活跃的，但依然会占用SM的资源。

![](https://github.com/Deleter-D/Images/assets/56388518/25ef389c-18fd-4810-bc4a-bcfe47c374a5)

### 线程束分化

首先要注意一点，一个线程束中的所有线程在同一周期内必须执行相同的指令。考虑下列语句：

```cpp
if (cond) {
    ...
} else {
    ...
}
```

假设在一个线程束中，有16个线程的`cond`为`true`，另外16个线程为`false`。此时，一半的线程需要执行`if`语句块中的指令，另一半需要执行`else`中的指令。这种在同一线程束中的线程执行不同指令的现象，被称为线程束分化。

当发生线程束分化时，线程束将连续执行每个分支路径，同时禁用不执行这一路径的线程，这会导致性能明显下降。条件分支越多，并行性削弱越严重。

> 线程束分化只发生在同一线程束中，不同线程束的不同条件值不会引起线程束分化。

这里引入一个概念，分支效率，即未分化分支与全部分支之比。
$$
\text{分支效率}=100\times\left(\frac{\text{分支数}-\text{分化分支数}}{\text{分支数}}\right)
$$
设想以下三种情况：

- 情况一：线程ID为偶数的执行`if`，线程ID为奇数的执行`else`；
- 情况二：线程束ID为偶数的执行`if`， 线程束ID为奇数的执行`else`；
- 情况三：线程ID为偶数的执行`if`，线程ID为奇数的执行另一个`if`。

具体到代码即为：

```cpp
// 线程ID为偶数的执行if，线程ID为奇数的执行else
__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;
    c[tid] = a + b;
}
```

```cpp
// 线程束ID为偶数的执行if， 线程束ID为奇数的执行else
__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;
    c[tid] = a + b;
}
```

```cpp
// 线程ID为偶数的执行if，线程ID为奇数的执行另一个if
__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    bool ipred = (tid % 2 == 0);
    if (ipred)
        a = 100.0f;
    if (!ipred)
        b = 200.0f;
    c[tid] = a + b;
}
```

调用这三个核函数，使用`ncu`来统计分支效率，结果如下所示。

|                                             | 分支效率 |
| ------------------------------------------- | -------- |
| `mathKernel1(float *) (1, 1, 1)x(64, 1, 1)` | 80%      |
| `mathKernel2(float *) (1, 1, 1)x(64, 1, 1)` | 100%     |
| `mathKernel3(float *) (1, 1, 1)x(64, 1, 1)` | 71.43%   |

> 为了阅读方便，这里简化了`ncu`的输出信息，实际的输出比上述形式要丰富，后续的`ncu`分析结果也以同样的方式简化。

可以观察到，情况一由于发生了分支分化，导致分支效率降低。而情况二由于分支分化的粒度是线程束大小的整倍数，所以分支效率统计为100%。情况三在情况一的前提下改造为了两个`if`，这样可以使得分化分支的数量翻倍。

> 分化分支数量翻倍但效率没有降低至一半是因为，虽然在编译时加上了`-G`参数来阻止分支预测优化，但还有其他优化手段，以保证分支效率在50%以上。
>
> 详细代码示例参考[warp_divergence.cu](https://github.com/Deleter-D/CUDA/blob/master/02_execution_model/01_warp_divergence.cu)

### 资源分配

线程束的本地执行上下文主要由以下资源组成：

- 程序计数器；
- 寄存器；
- 共享内存。

由SM处理的每个线程束的执行上下文，在整个线程束的生存期中是保存在芯片内的。所以从一个执行上下文切换到另一个执行上下文没有损失。对于一个给定的核函数，同时存在于同一个SM中的线程块和线程束的数量，取决于在SM中可用的与核函数所需的寄存器和共享内存数量。

![](https://github.com/Deleter-D/Images/assets/56388518/49674a9f-2a18-4eae-ae62-0f70dddc6046)

如上图所示，每个线程消耗的寄存器较少，同一SM上就可以多分配一些线程。同理，每个线程块消耗的共享内存较少，同一SM上就可以多分配一些线程块。

如果每个SM没有足够的寄存器或共享内存去处理至少一个块，那么核函数就无法启动。

当计算资源已经分配给线程块时，线程块被称为活跃的块。它所包含的线程束被称为活跃的线程束。活跃的线程束可以分为以下三种类型：

- 选定的线程束：正在执行的活跃线程束；
- 阻塞的线程束：未准备好执行的线程束；
- 符合条件的线程束：准备执行但尚未执行的活跃线程束。

同时满足以下两个条件则线程束符合执行条件：

- 32个CUDA核心可用于执行；
- 当前指令中所有的参数都已就绪。

### 延迟隐藏

指令延迟是指在指令发出和完成之间的时钟周期数。指令可以被分为两种基本类型：

- 算术指令：一个算术操作从开始到产生输出之间的时钟周期，一般为10～20个周期；
- 内存指令：发送出的加载或存储操作和数据到达目的地之间的时钟周期，全局内存访问一般为400～800个周期。

当每个时钟周期内所有的线程调度器都有一个符合条件的线程束时，可以达到计算资源的完全利用。在GPU中往往有着大量的线程，通过在其他常驻线程束中发布其他指令来隐藏指令延迟至关重要。

可以通过利特尔法则来估算隐藏延迟所需的活跃线程束数量。
$$
\text{所需线程束数量}=\text{指令平均延迟}\times\text{欲达到的吞吐量}
$$
此处是一个粗略化的公式，并不是直接套用即可算出所需线程束数量，后面将具体的介绍如何运用该法则。

#### 算术延迟隐藏

对于算术运算，所需的并行数可以表示为隐藏算术延迟所需的操作数量。

假设某个算术指令延迟为20个周期，我们想要令SM保持32个操作的吞吐量，即每个周期进行32次操作。根据上面提到的利特尔法则，我们可以得到所需并行数为$20 \times 32 =640$，也就是说要保证程序中有640个该计算操作才能完全隐藏算术延迟。

我们再假设每个线程中仅执行一次该算术操作，则可以进一步得到线程束数量为$640 \div 32 = 20$个。

观察上述例子会发现，算术延迟隐藏所需的并行数可以用操作数量来表示，也可以用线程束数量来表示。这表明我们可以有两个不同的层次来提高并行：

- 指令级并行（ILP）：一个线程中有很多独立的指令；
- 线程级并行（TLP）：很多并发地符合条件的线程。

#### 内存延迟隐藏

 对于内存操作，所需的并行数可以表示为在每个周期内隐藏内存延迟所需的字节数。

假设某个内存指令延迟为800个周期，我们想要令设备保持200GB/s的吞吐量，根据内存频率可以将吞吐量的单位由GB/s转换为B/CP（字节/周期）。笔者的设备内存频率为10.501GHz，所以转换后为$200\text{GB\\s}\div 10.501\text{GHz}\approx 19\text{B\\CP}$。接着根据利特尔法则，我们可以得到所需的并行数为$800\times19=15200\text{B}$。

> 使用如下命令来获取内存频率。
>
> ```sh
> nvidia-smi -a -q -d CLOCK | grep -A 3 "Max Clocks" | grep "Memory"
> ```

我们再假设每个线程中仅从全局内存中读取一个浮点数到SM上用于计算，则根据并行数可以计算出所需的线程数，即$15200\text{B}\div 4\text{B}=3800$个线程。进一步得到线程束数量为$\lceil 3800\div 32\rceil=119$个。若每个线程执行多个独立的4字节加载，则隐藏内存延迟所需的线程就可以更少。

> 上述计算出的线程束数量只是下界，也就是说在相同假设下，提供更多的线程数量同样能够达到延迟隐藏的效果。

### 占用率

占用率是每个SM中活跃的线程束占最大线程束数量的比值。
$$
占用率=\frac{活跃线程束数量}{最大线程数数量}
$$
 最大线程束数量可以通过`cudaGetDeviceProperties()`获取到设备属性后，由其成员`maxThreadsPerMultiProcessor / 32`取得。详细代码参考[]()，笔者的设备获取到的结果如下所示。

```
Device 0: NVIDIA GeForce RTX 4070
Number of multiprocessors: 46
Total amount of constant memory: 64.00 KB
Total amount of shared memory per block: 48.00 KB
Total amount of registers available per block: 65536
Warp size: 32
Maximum number of threads per block: 1024
Maximum number of threads per multiprocessor: 1536
Maximum number of warps per multiprocessor: 48
```

CUDA官方以前提供一个占用率计算器，是一个Excel表格，可以填入一些核函数资源信息后，自动计算SM占用率。但目前在官网已经找不到该文件的下载途径了，可能是因为当前最新的设备已经不适合用这种方式来计算占用率了。

> 若想体验该计算器，笔者在Github上找到一个项目，提供相同的功能，但该项目仅支持计算能力8.6及以前的设备，CUDA版本仅支持11.0和11.1两个版本，链接：[cuda-calculator](http://karthikeyann.github.io/cuda-calculator/)。

为了提高占用率，需要调整线程块配置或重新调整资源的使用情况，以允许更多的线程束同时处于活跃状态并提高计算资源的利用率。要避免极端的情况：

- 线程块过小：每个块中的线程太少，会在所有资源被充分利用之前导致硬件达到每个SM的线程束数量限制；
- 线程块过大：每个块中的线程太多，会导致SM中每个线程可用的硬件资源较少。

### 同步

CUDA中提供了两个级别的同步原语：

- 系统级别：等待主机和设备完成所有工作；
- 块级别：在设备执行过程中等待一个线程块中所有线程到达同一点。

系统级别的同步通过`cudaDeviceSynchronize()`API实现，块级别的同步通过`__syncthreads()`实现。线程块中要注意避免各种访存冲突，例如读后写、写后读、写后写等。

不同块之间没有线程同步，实现块间同步可以通过全局变量+原子操作的方式实现。从CUDA 9.x开始提供了协作组的概念，`cooperative_groups::grid_group`下有一个`sync()`函数可以提供块间同步的功能。关于块间同步这里不过多展开。

### 可扩展性

能够在可变数量的计算核心上执行相同代码的能力被成为透明可扩展性。拥有这种能力的平台能够避免不同的硬件产生的变化，减轻了开发者的负担。

可扩展性比效率更重要，一个可扩展但效率很低的系统可以通过简单添加硬件核心来处理更大的工作负载，一个效率很高但不可扩展的系统可能很快就会达到性能上限。

CUDA核函数启动时，线程块分布在多个SM中，网格中的线程块以并行或连续或任意的顺序执行。这种独立性使得CUDA程序可以在任意数量的计算核心间扩展。

## 并行性表现

定义一个二维矩阵求和的核函数。

```cpp
__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
        C[idx] = A[idx] + B[idx];
}
```

> 详细代码参考[parallelism.cu](https://github.com/Deleter-D/CUDA/blob/master/02_execution_model/03_parallelism.cu)。

### 检测活跃线程束

我们利用不同的线程块大小设计来执行上面的核函数，统计核函数执行事件，并使用`ncu`分析不同情况下的占用率。

> 这里的占用率指的是：每周期内活跃线程束的平均数量与一个SM支持的线程束最大数量的比值。

这里对线程块采用四种不同的设计：`(32, 32)`、`(32, 16)`、`(16, 32)`、`(16, 16)`，得到的分析结果如下所示。

核函数耗时情况如下。

```
sumMatrixOnGPU2D <<<(512, 512), (32, 32)>>> elapsed 7.0817 ms
sumMatrixOnGPU2D <<<(512, 1024), (32, 16)>>> elapsed 7.02669 ms
sumMatrixOnGPU2D <<<(1024, 512), (16, 32)>>> elapsed 7.03258 ms
sumMatrixOnGPU2D <<<(1024, 1024), (16, 16)>>> elapsed 7.03968 ms
```

占用率分析结果如下。

|                                                 | 占用率 |
| ----------------------------------------------- | ------ |
| `sumMatrixOnGPU2D <<<(512, 512), (32, 32)>>>`   | 48.07% |
| `sumMatrixOnGPU2D <<<(512, 1024), (32, 16)>>>`  | 59.31% |
| `sumMatrixOnGPU2D <<<(1024, 512), (16, 32)>>>`  | 63.41% |
| `sumMatrixOnGPU2D <<<(1024, 1024), (16, 16)>>>` | 67.98% |

观察上述结果：

- 情况二`(32, 16)`中的线程块比情况一`(32, 32)`更多，所以设备可以有更多活跃的线程束，是其占用率更高可能的原因之一；
- 情况四`(16, 16)`的占用率最高，但并不是最快的，因此，更高的占用率并不一定代表更高的性能。

### 检测内存操作

上面提到的矩阵求和的核函数中有三个内存操作，两次加载和一次存储。同样使用`ncu`来分析核函数的内存读取效率和全局加载效率，分析结果如下。

> 全局加载效率指的是被请求的全局加载吞吐量占所需的全局加载吞吐量的比值，它衡量了程序的加载操作利用设备内存带宽的程度。

|                                                 | 内存读取效率 | 全局加载效率 |
| ----------------------------------------------- | ------------ | ------------ |
| `sumMatrixOnGPU2D <<<(512, 512), (32, 32)>>>`   | 296.77 GB/s  | 100%         |
| `sumMatrixOnGPU2D <<<(512, 1024), (32, 16)>>>`  | 297.72 GB/s  | 100%         |
| `sumMatrixOnGPU2D <<<(1024, 512), (16, 32)>>>`  | 295.40 GB/s  | 100%         |
| `sumMatrixOnGPU2D <<<(1024, 1024), (16, 16)>>>` | 297.79 GB/s  | 100%         |

如果阅读过《CUDA C编程权威指南》一书中的相关介绍，会发现我们这里得到的分析结果与书中提到的截然不同。书中描述的情况三和情况四下，全局加载效率会有明显的下降。但这里不同的线程块设计并没有导致太大的内存操作性能波动，笔者推测是因为nvcc编译器在这方面做了比以前更多的优化，来保证线程在SM中的调度更加合理。

根据书中的描述，在一节分析到的结果是，对网格和线程块的启发式算法来说，最内层的维数（`block.x`）应该是线程束大小的整倍数，这一结论还是有参考意义的。

### 增大并行性

我们来探讨一个问题，根据上一节得到的结论，继续增加`block.x`会增大吞吐量吗？同样是利用上一节中的例子，使用不同的线程块设计来执行核函数。

```
sumMatrixOnGPU2D <<<(256, 8192), (64, 2)>>> elapsed 7.03245 ms
sumMatrixOnGPU2D <<<(256, 4096), (64, 4)>>> elapsed 7.05654 ms
sumMatrixOnGPU2D <<<(256, 2048), (64, 8)>>> elapsed 7.02928 ms
sumMatrixOnGPU2D <<<(128, 8192), (128, 2)>>> elapsed 7.03757 ms
sumMatrixOnGPU2D <<<(128, 4096), (128, 4)>>> elapsed 7.03094 ms
sumMatrixOnGPU2D <<<(128, 2048), (128, 8)>>> elapsed 7.04643 ms
sumMatrixOnGPU2D <<<(64, 8192), (256, 2)>>> elapsed 7.10362 ms
sumMatrixOnGPU2D <<<(64, 4096), (256, 4)>>> elapsed 7.03165 ms
```

虽然笔者这里的测试结果差距并不明显，不必过多纠结，了解思想即可。

分析结果可以得出几条规律：

- 情况一`(64, 2)`中启动的线程块数量最多，但并不是速度最快的；
- 情况二`(64, 4)`与情况四`(128, 2)`相比，两者有相同数量的线程块（`(256, 4096)`与`(128, 8192)`），但情况四的表现优于情况二。这恰好印证了前一节中的结论，线程块最内层的维数对性能起着关键作用；
- 除了情况一、二、四外，其余情况的线程块数量均比最优情况少。故增大并行性是性能优化的一个重要因素。

接下来分析上述各个情况的占用率，分析方法同之前的[检测活跃线程束](#检测活跃线程束)。分析结果如下。

|                                                | 占用率 |
| ---------------------------------------------- | ------ |
| `sumMatrixOnGPU2D <<<(256, 8192), (64, 2)>>>`  | 82.38% |
| `sumMatrixOnGPU2D <<<(256, 4096), (64, 4)>>>`  | 73.27% |
| `sumMatrixOnGPU2D <<<(256, 2048), (64, 8)>>>`  | 61.94% |
| `sumMatrixOnGPU2D <<<(128, 8192), (128, 2)>>>` | 79.33% |
| `sumMatrixOnGPU2D <<<(128, 4096), (128, 4)>>>` | 64.16% |
| `sumMatrixOnGPU2D <<<(128, 2048), (128, 8)>>>` | 50.34% |
| `sumMatrixOnGPU2D <<<(64, 8192), (256, 2)>>>`  | 73.05% |
| `sumMatrixOnGPU2D <<<(64, 4096), (256, 4)>>>`  | 52.63% |

书中描述的情况一`(64, 2)`是占用率最低的情况，因为线程块是最多的，触及到了书作者当时的硬件瓶颈。但在笔者的环境下，情况一反而占用率是最高的，这也体现了硬件进步带来的效果。

虽然与书中描述的情况有所不同，但这也恰恰印证了提高并行性的重要程度，当硬件资源不再是限制和瓶颈的时候，更大的并行程度将带来更高的性能。

经过上面一系列的分析我们能够发现，性能最好的线程块设计，既没有最高的占用率，也没有最高的加载吞吐量。**可见，没有一个单独的指标可以直接优化性能，我们需要在几个相关的指标之间寻找一个平衡来获得全局最优性能。**

## 避免分支分化

### 并行归约问题

一般的并行求和是将较多的数据分块计算，每个线程负责一个数据块的求和，再对每个数据块的和求和即为最终结果。一个常用方法是使用迭代成对实现，一个数据块只包含一对元素，每个线程求得这对元素的和，作为下一次迭代的输入。当输出向量长度为1时，表明最终结果已经被计算出来了。

成对的并行求和可以进一步分为两种类型：

- 相邻配对：元素与它们相邻的元素配对；
- 交错配对：根据给定的步长配对元素。

![](https://github.com/Deleter-D/Images/assets/56388518/1cb84e0d-3bff-4d8e-8014-eccf15ef7f19)

虽然上述介绍的是加法，但任何满足交换律和结合律的运算都可以采用这种思路。

在向量中执行满足交换律和结合律的运算，被称为归约问题。并行归约问题是这种归约运算的并行执行。

### 并行归约中的分化

首先实现一个相邻配对的并行归约求和核函数。

```cpp
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 将全局数据指针转换为当前block的局部数据指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // 边界检查
    if (idx >= size) return;

    // 在全局内存中原地归约
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
            idata[tid] += idata[tid + stride];
        __syncthreads(); // 同步线程，保证下一轮迭代正确
    }

    // 将当前block的结果写入全局内存
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

两个相邻元素之间的距离成为步长（stride），初始化为1。每次归约循环后，步长被乘以2。由于块间同步很不方便，所以将每个块的求和结果拷贝回主机之后再进行串行求和。

具体求和过程如图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/bced3cb3-c047-4e97-9c44-126d312cf1ce)

测试后得到的性能如下所示。

```
Array Size: 16777216
cpu reduce      elapsed 16.2122 ms      cpu_sum: 206464799
gpu neighbored  elapsed 0.628512 ms     gpu_sum: 206464799      <<<32768, 512>>>
Result correct!
```

> 这里采用一维网格与一维线程块，详细代码参考[reduction.cu](https://github.com/Deleter-D/CUDA/blob/master/02_execution_model/04_reduction.cu)，后面将以这个核函数的表现作为性能基准。这个`cu`文件将伴随整个[避免分支分化](#避免分支分化)和[展开循环](#展开循环)两个小节。

### 改善并行归约的分化

注意上面核函数中的条件表达式。

```cpp
if ((tid % (2 * stride)) == 0)
```

我们在前面也介绍过，这会导致非常严重的线程束分化。第一次迭代只有ID为偶数的线程是活跃的，第二次迭代就只有四分之一的线程活跃了，但那些不活跃的线程依旧会被调度。

改进这一现象的方法是强制ID相邻的线程执行求和操作，线程束分化就可以被归约了。将核函数修改为如下形式。

```cpp
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= size) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // 将tid转换为局部数组索引
        int index = 2 * stride * tid;
        if (index < blockDim.x)
            idata[index] += idata[index + stride];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

这样就将具体的运算过程变为了如下所示的状态。

![](https://github.com/Deleter-D/Images/assets/56388518/6f8f03cc-7ed9-40b2-a971-696e77bfc3a6)

虽然这种改进在一定程度上降低了线程束分化的程度，但在最后几轮迭代中，还是会存在线程束分化的情况。例如对于一个有512个线程的块来说，第一轮迭代由前8个线程束完成，后8个线程束不处于活跃状态。前几轮迭代都同理，但当最后五轮迭代中，活跃的线程数量小于线程束大小的时候，还是会发生线程束分化。

性能测试的表现如下所示。

```
Array Size: 16777216
cpu reduce      elapsed 16.167 ms       cpu_sum: 206464799
gpu neighbored  elapsed 0.67648 ms      gpu_sum: 206464799      <<<32768, 512>>>
gpu neighboredL elapsed 0.424576 ms     gpu_sum: 206464799      <<<32768, 512>>>
Result correct!
```

虽然在最后几轮还是会发生线程束分化，但依旧比不做任何处理快了1.6倍左右。我们可以利用`ncu`来分析每个线程束中执行的指令数量和内存读取效率来解释这种现象，分析结果如下所示。

|                                                              | 每线程束执行指令数 | 内存读取效率 |
| ------------------------------------------------------------ | ------------------ | ------------ |
| `reduceNeighbored(int *, int *, unsigned int) (32768, 1, 1)x(512, 1, 1)` | 341.94 inst/warp   | 690.98 GB/s  |
| `reduceNeighboredLess(int *, int *, unsigned int) (32768, 1, 1)x(512, 1, 1)` | 115.38 inst/warp   | 1.27 TB/s    |

可以观察到，在改善线程束分化后，每个线程束执行的指令数量大幅下降。而且拥有更大的加载吞吐量，因为虽然I/O操作数量相同，但耗时更短。

### 交错配对的归约

与相邻配对的方法相比，交错配对的方法反转了元素步长的变化，初始化为数据块大小的一半，然后每轮迭代减少一半。

```cpp
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= size) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

具体运算过程如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/c6a05582-869a-471f-bdeb-059989948cf8)

性能测试表现如下。

```
Array Size: 16777216
cpu reduce      elapsed 15.991 ms       cpu_sum: 206464799
gpu neighbored  elapsed 0.676736 ms     gpu_sum: 206464799      <<<32768, 512>>>
gpu neighboredL elapsed 0.422176 ms     gpu_sum: 206464799      <<<32768, 512>>>
gpu interleaved elapsed 0.364032 ms     gpu_sum: 206464799      <<<32768, 512>>>
Result correct!
```

虽然交错配对的方式与优化后的相邻配对方式拥有相同的线程束分化情况，但仍然有性能的提升。这种性能提升是由全局内存加载 / 存储模式导致的，在后续的文章中会进一步讨论。

## 展开循环

循环展开是一种尝试减少分支出现频率和循环维护指令来优化循环的技术。在循环展开中，循环主体在代码中要多次编写，任何封闭循环都可以将它的迭代次数减少或完全消除。循环体的复制数量被成为循环展开因子，迭代次数以下列公式得到。
$$
\text{迭代次数}=\frac{原始循环迭代次数}{循环展开因子}
$$
为了方便理解，观察如下示例。

```cpp
for (int i = 0; i < 100; i++) {
    a[i] = b[i] + c[i];
}
```

如果像下面这样重复一次循环体，迭代次数就可以减少一半。

```cpp
for (int i = 0; i < 100; i += 2) {
    a[i] = b[i] + c[i];
    a[i + 1] = b[i + 1] + c[i + 1];
}
```

### 展开的归约

我们用上面的思路来将之前提到的交错配对的归约求和操作进行循环展开。

先将两个数据块汇聚到一个线程块中，每个线程作用于多个数据块，并处理每个数据块的一个元素。

```cpp
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

    // 与之前不同，这里将两个数据库汇总到一个线程块中
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < size)
        g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            idata[tid] += idata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

比较关键的修改如下，每个线程都添加一个来自于相邻数据块的元素。可以把它作为归约循环的一个迭代，可以在数据块间进行归约。

```cpp
if (idx + blockDim.x < size)
        g_idata[idx] += g_idata[idx + blockDim.x];
```

然后调整全局数组索引，只需要一半的线程块来处理数据。

```cpp
unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;
int *idata = g_idata + blockIdx.x * blockDim.x * 2;
```

进行性能测试，结果如下。

```
Array Size: 16777216
cpu reduce      elapsed 16.2422 ms      cpu_sum: 206464799
gpu interleaved elapsed 0.364096 ms     gpu_sum: 206464799      <<<32768, 512>>>
gpu unrolling2  elapsed 0.28352 ms      gpu_sum: 206464799      <<<16384, 512>>>
Result correct!
```

可以观察到性能得到了进一步的提升，我们尝试进一步提高展开程度，性能测试结果如下。

```
Array Size: 16777216
cpu reduce      elapsed 16.0142 ms      cpu_sum: 206464799
gpu interleaved elapsed 0.364608 ms     gpu_sum: 206464799      <<<32768, 512>>>
gpu unrolling2  elapsed 0.281024 ms     gpu_sum: 206464799      <<<16384, 512>>>
gpu unrolling4  elapsed 0.262944 ms     gpu_sum: 206464799      <<<8192, 512>>>
gpu unrolling8  elapsed 0.256736 ms     gpu_sum: 206464799      <<<4096, 512>>>
Result correct!
```

可以观察到，在一个线程中有更多的独立内存操作会得到更好的性能，因为内存延迟可以得到很好的隐藏。我们利用`ncu`来分析设备内存读取吞吐量来解释性能提升的理由。

|                                                              | 设备内存读取吞吐量 |
| ------------------------------------------------------------ | ------------------ |
| `reduceInterleaved(int *, int *, unsigned int) (32768, 1, 1)x(512, 1, 1)` | 187.98 GB/s        |
| `reduceUnrolling2(int *, int *, unsigned int) (16384, 1, 1)x(512, 1, 1)` | 293.78 GB/s        |
| `reduceUnrolling4(int *, int *, unsigned int) (8192, 1, 1)x(512, 1, 1)` | 335.35 GB/s        |
| `reduceUnrolling8(int *, int *, unsigned int) (4096, 1, 1)x(512, 1, 1)` | 342.42 GB/s        |

这里可以得到一个结论，归约的循环展开程度和设备读取吞吐量之间是成正比的。

### 展开线程的归约

上面提到过，当最后几轮迭代的时候，线程数量少于线程束大小时，线程束分化依旧会发生。由于线程束的执行时SIMT的模式，每条指令之后有隐式的线程束内同步。所以可以借助这一隐式同步，将最后几轮迭代用下列语句展开。

```cpp
if (tid < 32)
{
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
}
```

> 注意：变量`vmem`是被`volatile`修饰符修饰的，它告诉编译器每次赋值时必须将`vmem[tid]`的值存回全局内存中。如果省略了`volatile`修饰符，编译器或缓存可能优化对全局或共享内存的读写。若位于全局或共享内存中的变量有`volatile`修饰符，则编译器会假定其值可以被其他线程在任何时间修改或使用。故任何带有`volatile`修饰符的变量会强制直接读写内存，而不是简单的读写缓存或寄存器。

性能测试结果如下。

```
gpu unrolling8  elapsed 0.22992 ms      gpu_sum: 206464799      <<<4096, 512>>>
gpu unrolWarps8 elapsed 0.227296 ms     gpu_sum: 206464799      <<<4096, 512>>>
```

我们可以通过分析被阻塞线程束的占比来作证这个性能提升。

|                                                              | 阻塞线程束占比 |
| ------------------------------------------------------------ | -------------- |
| `reduceUnrolling8(int *, int *, unsigned int) (4096, 1, 1)x(512, 1, 1)` | 20.83%         |
| `reduceUnrollWarps8(int *, int *, unsigned int) (4096, 1, 1)x(512, 1, 1)` | 12.56%         |

可以观察到，通过展开最后的线程束，被阻塞的线程束占比大幅度下降，所以进一步提升了性能。

### 完全展开的归约

由于当前计算能力的设备，每个线程块最大的线程束是1024，且上述的归约核函数中循环迭代次数是基于一维网格与一维线程块的，所以完全展开归约循环是可行的。

```cpp
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int size)
{
    ...
    // 完全展开
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
	...
}
```

性能测试的结果如下所示，又有小小的提升。

```
gpu unrolWarps8 elapsed 0.227264 ms     gpu_sum: 206464799      <<<4096, 512>>>
gpu CmptUnroll8 elapsed 0.224992 ms     gpu_sum: 206464799      <<<4096, 512>>>
```

### 模板函数的归约

虽然可以手动展开循环，但使用模板函数有助于进一步减小分支消耗，关键代码如下。

```cpp
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int size)
{
	...
    if (iBlockSize >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (iBlockSize >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (iBlockSize >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (iBlockSize >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
	...
}
```

这样做的好处是，检查块大小的`if`语句在编译时会被评估，若这一条件为`false`，则该分支块在编译时就会被删除。这类核函数一定要在`switch-case`结构中被调用，这样可以使编译器为特定大小的线程块自动优化代码。

```cpp
switch (blocksize)
{
case 1024:
    reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
    break;
case 512:
    reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
    break;
case 256:
    reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
    break;
case 128:
    reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
    break;
case 64:
    reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
    break;
}
```

## 归约小结

至此，我们借助归约求和探讨了核函数的几个优化方案，大致分为避免分支分化和展开循环两个思路，细节上述部分已经充分讨论过了。下表中展示了从一开始的相邻配对归约，到改善了线程束分化问题，最后到完全展开循环核函数的性能对比。

| 核函数描述                                     | 耗时 (ms) | 单步加速 | 累计加速 | 加载效率 | 存储效率 |
| ---------------------------------------------- | --------- | -------- | -------- | -------- | -------- |
| 相邻配对（分化）                               | 0.674112  |          |          | 25.02    | 25       |
| 相邻配对（改善分化）                           | 0.581312  | 1.16     | 1.16     | 25.02    | 25       |
| 交错配对                                       | 0.364192  | 1.60     | 1.85     | 96.15    | 95.52    |
| 循环展开（2块）                                | 0.255392  | 1.43     | 2.64     | 98.04    | 97.71    |
| 循环展开（4块）                                | 0.252832  | 1.01     | 2.67     | 98.68    | 97.71    |
| 循环展开（8块）                                | 0.229664  | 1.10     | 2.94     | 99.21    | 97.71    |
| 循环展开（8块）+ 最后线程束展开                | 0.225184  | 1.02     | 2.99     | 99.43    | 99.40    |
| 循环展开（8块）+ 完全循环展开 + 最后线程束展开 | 0.224096  | 1.00     | 3.01     | 99.43    | 99.40    |
| 模板化核函数                                   | 0.211616  | 1.06     | 3.19     | 99.43    | 99.40    |

## 动态并行

CUDA的动态并行允许在GPU端直接创建和同步新的GPU核函数，可以在一个核函数的任意点动态增加并行性。动态并行可以在运行时才决定网格和线程块的大小。

### 嵌套执行

在GPU进行核函数调用的方法与主机端的调用方法相同。

在动态并行中，核函数执行分为双亲和孩子两种类型。父线程、父线程块或父网格启动一个新的网格，即子网格。子线程、子线程块或子网格被双亲启动。子网格必须在父线程、父线程块或父网格完成之前完成。只有在所有子网格都完成后，双亲才会完成。

> 若调用的线程没有显式地同步子网格，则CUDA运行时会保证双亲与孩子之间的隐式同步。

当双亲启动一个子网格，父线程块与孩子显式同步后，孩子才能开始执行。

关于动态并行的内存访问有以下几点：

- 父网格和子网格共享相同的全局和常量内存，但它们有不同的局部内存和共享内存；
- 双亲和孩子之间以弱一致性为保证，使得父子网格可以对全局内存并发存取；
- 在子网格开始和完成两个时刻，子网格和它的父线程见到的内存完全相同；
- 当父线程优先于子网格调用时，所有的全局内存操作要保证子网格可见；
- 当双亲在子网格完成时进行同步后，子网格所有的内存操作要保证双亲可见；

### 在GPU上嵌套Hello World

实现一个嵌套调用的核函数来在GPU上输出Hello World，具体的嵌套调用方式如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/27c38e9d-785d-420e-962a-2b60899e6d83)

实现核函数如下。

```cpp
__global__ void nestedHelloWorld(int const size, int depth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", depth, tid, threadIdx.x);

    if (size == 1) return;

    int threads = size >> 1;
    if (tid == 0 && threads > 0)
    {
        nestedHelloWorld<<<1, threads>>>(threads, ++depth);
        printf("-------> nested execution depth: %d\n", depth);
    }
}
```

> 详细代码参考[nestedHelloWorld.cu](https://github.com/Deleter-D/CUDA/blob/master/02_execution_model/05_nestedHelloWorld.cu)。注意需要编译选项-rdc为true，一些资料中提到还需要链接cudadevrt库，但笔者这里没有显式链接也正常执行了，推测是自动链接了。

### 嵌套归约

由于CUDA从11.6开始就不允许在设备端执行`cudaDeviceSynchronize()`来同步子网格，并且CUDA 12.x开始CDP（CUDA Dynamic Parallelism）替换成了CDP2，在细节上与CDP1有所不同。

首先我们实现一个嵌套归约求和的核函数。

```cpp
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // 递归中止条件
    if (size == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = size >> 1;
    if (stride > 1 && tid < stride)
        idata[tid] += idata[tid + stride];
    __syncthreads();

    // 嵌套调用生成子网格
    if (tid == 0)
        gpuRecursiveReduce<<<1, stride, 0, cudaStreamTailLaunch>>>(idata, odata, stride);
    __syncthreads();
}
```

注意下面的语句。

```cpp
gpuRecursiveReduce<<<1, stride, 0, cudaStreamTailLaunch>>>(idata, odata, stride);
```

这里迫使该核函数在`cudaStreamTailLaunch`特殊流中执行，这是CDP2的新特性，这个流允许父网格在完成工作后才启动新网格。在大多数情况下，可以使用该流来实现与`cudaDeviceSynchronize()`相同的功能。

**在实际的实验过程中，笔者发现大多数情况下，这种嵌套的归约计算结果是错误的，但在小数据量的情况下是正确的，具体原因还有待分析。**

> 详细代码参考[nested_reduce.cu](https://github.com/Deleter-D/CUDA/blob/master/02_execution_model/06_nested_reduce.cu)。

对该核函数进行性能测试，结果如下所示。

```
Array size: 524288
Execution Configuration: grid 1024 block 512
cpu reduce      elapsed 0.470947 ms     cpu_sum: 6451596
gpu nested      elapsed 83.5106 ms      gpu_sum: 6451596        <<<1024, 512>>>
Result correct!
```

CUDA官方文档对于CDP2的同步有如下描述：

任何线程的CUDA运行时操作，包括核函数启动，在网格中的所有线程中都是可见的。这意味着父网格中的调用线程可以执行同步，以控制由网格中的任意线程在网格中的任何线程创建的流上启动网格的顺序。直到网格中所有线程的所有任务都已完成，网格的执行才被视为完成。如果网格中的所有线程在所有子网格完成之前退出，则将自动触发隐式同步操作。

大概意思就是，从CDP2开始，开发者已经不需要再核函数内进行显式的同步操作了，一切同步交给流和编译器来控制。去掉显式同步后的核函数实现如下所示。

```cpp
__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int size)
{
    unsigned tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    if (size == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int stride = size >> 1;
    if (stride > 1 && tid < stride)
    {
        idata[tid] += idata[tid + stride];
        if (tid == 0)
            gpuRecursiveReduceNosync<<<1, stride, 0, cudaStreamTailLaunch>>>(idata, odata, stride);
    }
}
```

性能测试如下。

```
Array size: 524288
Execution Configuration: grid 1024 block 512
cpu reduce      elapsed 0.496094 ms     cpu_sum: 6451596
gpu nested      elapsed 78.52 ms        gpu_sum: 6451596        <<<1024, 512>>>
gpu nestedNosyn elapsed 69.1308 ms      gpu_sum: 6451596        <<<1024, 512>>>
Result correct!
```

可以观察到性能有一些提升。在这个版本的实现中，每个线程块产生一个子网格，并引起了大量的调用，具体过程如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/4730c47f-e3ca-44d9-ac45-31d59345574b)

为了减少其创建的子网格数量，可以将启动方式改为下图所示的方法。

![](https://github.com/Deleter-D/Images/assets/56388518/792ec296-6651-4df4-a97c-a3838098cc35)

即只令第一个线程块的第一个线程来启动子网格，每次嵌套调用时，子线程块大小就会减小到其父线程块的一半。对于之前的实现来说，每个嵌套层的核函数执行过程都会有一半的线程空闲。但在这种实现方式中，所有空闲线程都会在每次核函数启动时被移除。这样会释放更多的计算资源，使得更多的线程块活跃起来。

同时，由于子线程块的大小是父线程块的一半，为了正确的计算数据的偏移，必须将一开始父线程块的大小传递进去。

```cpp
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int stride, int const dim)
{
    int *idata = g_idata + blockIdx.x * dim;

    if (stride == 1 && threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    idata[threadIdx.x] += idata[threadIdx.x + stride];

    if (threadIdx.x == 0 && blockIdx.x == 0)
        gpuRecursiveReduce2<<<gridDim.x, stride / 2>>>(g_idata, g_odata, stride / 2, dim);
}
```

性能测试如下。

```
Array size: 524288
Execution Configuration: grid 1024 block 512
cpu reduce      elapsed 0.48291 ms      cpu_sum: 6451596
gpu nested      elapsed 76.7643 ms      gpu_sum: 6451596        <<<1024, 512>>>
gpu nestedNosyn elapsed 69.8579 ms      gpu_sum: 6451596        <<<1024, 512>>>
gpu nested2     elapsed 0.082464 ms     gpu_sum: 6451596        <<<1024, 512>>>
gpu neighbored  elapsed 0.025632 ms     gpu_sum: 6451596        <<<1024, 512>>>
Result correct!
```

虽然对比之前的嵌套归约提升很大，但其性能甚至不如之前实现的性能最差的相邻匹配的归约。
