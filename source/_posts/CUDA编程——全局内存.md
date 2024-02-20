---
title: CUDA编程——全局内存
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 47184
date: 2024-02-20 16:09:37
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## CUDA内存模型概述

### CUDA内存模型

对于开发者来说，存储器分为两大类：

- 可编程的：需要显式控制哪些数据存放在可编程内存中；
- 不可编程的：不能决定数据的存放位置，由程序自动生成存放位置。

CUDA内存模型提出了多种可编程内存：

- 寄存器；
- 共享内存；
- 本地内存；
- 常量内存；
- 纹理内存；
- 全局内存。

这些内存空间的层次结构如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/f3635cc5-2ef0-43a1-bd3d-23f7cb7d0601)

#### 寄存器

寄存器是GPU上速度最快的内存空间，在核函数中声明一个没有其他修饰符的变量，通常存储在寄存器中。在核函数中声明的数组，若用于引用该数组的索引是常量且能在编译时确定，则该数组也存储在寄存器中。

寄存器是线程私有的，寄存器变量与核函数生命周期相同。若核函数使用的寄存器超过了硬件限制，则会用本地内存替代多占用的寄存器。nvcc编译器使用启发式策略来最小化寄存器使用，以免寄存器溢出。也可以手动显式添加额外信息来辅助编译器优化。

```cpp
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) kernel(...) {}
```

`maxThreadsPerBlock`指出每个线程块可以包含的最大线程数，`minBlocksPerMultiprocessor`是可选参数，指出在每个SM中预期的最小常驻线程块数量。

还可以使用编译器选项`maxrregcount`来控制一个编译单元中所有核函数使用的寄存器的最大数量，例如`-maxrregcount=32`。

#### 本地内存

核函数中符合存储在寄存器中但不能进入被该核函数分配的寄存器空间中的变量将溢出到本地内存中。编译器可能存放到本地内存中的变量有：

- 在编译时使用未知索引引用的本地数组；
- 可能会占用大量寄存器空间的较大本地结构体或数组；
- 任何不满足核函数寄存器限定条件的变量。

溢出到本地内存中的变量本质上与全局内存在同一存储区域，因此本地内存的访问特点是高延迟和低带宽。

#### 共享内存

在核函数中使用`__shared__`修饰符修饰的变量存放在共享内存中。

由于共享内存是片上内存，所以与本地内存和全局内存相比，具有更高的带宽和更低的延迟，是可编程的。每个SM都有一定数量的由线程块分配的共享内存，因此不能过度使用共享内存，避免在不经意间限制活跃线程束的数量。共享内存在核函数内声明，生命周期伴随整个线程块。

共享内存是线程之间通信的基本方式，访问共享内存必须使用`__syncthreads()`来进行同步。该函数设置了一个执行障碍点，使得同一线程块中的所有线程必须在其他线程开始执行前到达该处。

SM中的一级缓存和共享内存都使用64KB的片上内存，它通过静态划分，但在运行时可以使用`cudaFuncSetCacheConfig()`来进行动态配置。该API传入两个参数，第一个是函数指针，第二个是CUDA提供的一个枚举类`cudaFuncCache`成员，它包含4个成员：

- `cudaFuncCachePreferNone`：没有参考值（默认）；
- `cudaFuncCachePreferShared`：建议48KB共享内存和16KB一级缓存；
- `cudaFuncCachePreferL1`：建议48KB一级缓存和16KB共享内存；
- `cudaFuncCachePreferEqual`：建议相同尺寸的一级缓存和共享内存，均为32KB。

> 这里的`cudaFuncSetCacheConfig()`API已经是比较旧的方式了，在计算能力7.x及以上的设备中，更推荐使用`cudaFuncSetAttribute()`来配置。该API传入三个参数，第一个参数是函数指针，第二个是CUDA提供的一个枚举类`cudaFuncAttribute`成员，第三个是具体的提示值。
>
> `cudaFuncAttribute`有9个成员，但常用的只有两个（其它成员是关于线程块集群的，这里不多描述）：
>
> - `cudaFuncAttributeMaxDynamicSharedMemorySize`：指定最大动态共享内存大小；
> - `cudaFuncAttributePreferredSharedMemoryCarveout`：首选的共享内存和一级缓存拆分大小；
>
> 通过第三个参数来指定第二个参数成员的具体值，例如下面的语句提示编译器将片上内存的50%分配给共享内存。
>
> ```cpp
> cudaFuncSetAttribute(MyKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
> ```
>
> `cudaFuncSetAttribute()`放松了对指定共享内存容量的强制，分割被视为一种提示。而旧的`cudaFuncSetCacheConfig()`将共享内存容量视为核函数启动的硬性要求。因此，使用不同共享内存配置的核函数将进行一些不必要地序列化。

#### 常量内存

常量内存驻留在设备内存中，并在每个SM专用的常量缓存中缓存，使用`__constant__`来修饰。

常量变量必须在全局空间内和所有核函数之外进行声明，所有计算能力的设备都只能声明64KB的常量内存。常量内存是静态声明的，并对同一编译单元中的所有核函数可见。

核函数只能从常量内存中读取数据，因此常量内存必须在主机端使用`cudaMemcpyToSymbol()`来初始化，大多数情况下这个函数是同步的。

线程束中的所有线程从相同的内存地址中读取数据时，常量内存表现最好。

#### 纹理内存

纹理内存驻留在设备内存中，并在每个SM的只读缓存中缓存。纹理内存是一种通过指定的只读缓存访问的全局内存。只读缓存包括硬件滤波的支持，它可以将浮点插入作为读过程的一部分来执行。纹理内存是对二维空间局部性的优化，所以线程束中使用纹理内存访问二维数据的线程可以达到最优性能。

#### 全局内存

全局内存是GPU中最大、延迟最高且最常使用的内存。它的声明可以在任何SM设备上被访问到，并贯穿程序的整个生命周期。一个全局内存变量可以被静态声明或动态声明，可以使用`__device__`在设备代码中静态声明一个变量。

从多个线程访问全局内存时要注意，由于线程的执行不能跨线程块同步，不同线程块内的多个线程并发修改全局内存的同一位置可能会导致未定义的行为。

全局内存常驻于设备内存中，可通过32字节、64字节或128字节的内存事务进行访问。这些内存事务必须自然对齐，也就是说首地址必须是32字节、64字节或128字节的倍数。当一个线程束执行内存加载 / 存储时，需要满足的传输数量取决于两个因素：

- 跨线程的内存地址分布；
- 每个事务内存地址的对齐方式。

 一般情况下，用来满足内存请求的事务越多，未使用的字节被传输回的可能性就越高，这就导致了数据吞吐率的降低。

#### GPU缓存

GPU缓存是不可编程的内存，有四种类型的缓存：

- 一级缓存；
- 二级缓存；
- 只读常量缓存；
- 只读纹理缓存。

每个SM都有一个一级缓存，所有的SM共享一个二级缓存。一级和二级缓存用于存储本地内存和全局内存中的数据，也包括寄存器溢出的部分。GPU上只有内存加载操作可以被缓存，内存存储操作不能被缓存。每个SM有一个只读常量缓存和只读纹理缓存，用于在设备内存只提高来自于各自内存空间中的读取性能。

#### CUDA变量声明小结

CUDA变量和类型修饰符总结如下表。

| 修饰符         | 变量类型    | 存储器   | 作用域 | 生命周期 |
| -------------- | ----------- | -------- | ------ | -------- |
|                | 标量        | 寄存器   | 线程   | 线程     |
|                | 数组        | 本地内存 | 线程   | 线程     |
| `__shared__`   | 标量 / 数组 | 共享内存 | 线程块 | 线程块   |
| `__device__`   | 标量 / 数组 | 全局内存 | 全局   | 应用程序 |
| `__constant__` | 标量 / 数组 | 常量内存 | 全局   | 应用程序 |

设备存储器的特征总结如下表。

| 存储器   | 位置 | 缓存          | 存取 | 范围          | 生命周期 |
| -------- | ---- | ------------- | ---- | ------------- | -------- |
| 寄存器   | 片上 | n/a           | R/W  | 一个线程      | 线程     |
| 本地内存 | 片外 | Yes (2.x以上) | R/W  | 一个线程      | 线程     |
| 共享内存 | 片上 | n/a           | R/W  | 块内所有线程  | 线程块   |
| 全局内存 | 片外 | Yes (2.x以上) | R/W  | 所有线程+主机 | 主机配置 |
| 常量内存 | 片外 | Yes           | R    | 所有线程+主机 | 主机配置 |
| 纹理内存 | 片外 | Yes           | R    | 所有线程+主机 | 主机配置 |

#### 静态全局内存

实现一段静态声明全局内存变量的代码，在主机端传入值，在核函数中对值进行修改，再传回主机端，核心代码如下。

```cpp
__device__ float devData;

__global__ void checkGlobalVariable()
{
    devData += 2.0f;
}

int main(int argc, char const *argv[])
{
	...
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
	...
}
```

值的注意的是，尽管设备的全局变量声明与主机代码在同一文件中，主机代码也不能直接访问设备变量。类似地，设备代码也不能直接访问主机变量。

唯一比较像是主机代码访问设备变量的地方是`(devData, &value, sizeof(float))`，但该接口是在CUDA运行时API中的，内部可以隐式的使用GPU来访问。而且在这里`devData`作为一个标识符，并不是全局内存变量的地址。在核函数中，`devData`被当作全局内存中的一个变量。

`cudaMemcpy()`并不能直接以下面语句中的变量地址传递数据给`devData`。

```cpp
cudaMemcpy(&devData, &value, sizeof(float), cudaMemcpyHostToDevice);
```

我们无法在主机端的设备变量中使用`&`运算符，因为它只是一个在GPU上表示物理位置的符号。但可以通过下面的语句显式获取一个全局变量的地址。

```cpp
float *dptr;
cudaGetSymbolAddress((void **)&dptr, devData);
```

然后就可以使用`cudaMemcpy()`来进行拷贝操作。

```cpp
cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice)
```

> 详细代码参考[global_variable.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/01_global_variable.cu)，其中展示了`cudaMemcpyToSymbol()`和`cudaMemcpy()`两种操作方式。

有一种例外可以直接从主机引用GPU内存：CUDA固定内存。将会在后续进行介绍。

## 内存管理

### 设备内存

设备内存可以作为线性内存分配，也可以作为CUDA 数组来分配。CUDA数组是为了纹理获取而优化过的不透明内存布局。

线性内存是由一个统一的地址空间分配的，分开分配的实体可以通过指针相互引用。地址空间的大小取决于主机系统（CPU）和所用GPU的计算能力。

| 计算能力  | x86_64 (AMD64) | POWER (ppc64le) | ARM64 |
| --------- | -------------- | --------------- | ----- |
| 5.3及之前 | 40bit          | 40bit           | 40bit |
| 6.0及以后 | 47bit          | 49bit           | 48bit |

线性内存使用`cudaMalloc()`分配，使用`cudaFree()`释放，主机内存与设备内存之间的数据搬移通过`cudaMemcpy()`完成。

下面以向量加法为例。

```c++
// 设备代码
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// 主机代码
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);
    // 在主机端申请输入向量h_A和h_B及结果向量h_C的内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    // 初始化输入向量
    ...
    // 在设备端申请向量内存
    float* d_A; cudaMalloc(&d_A, size);
    float* d_B; cudaMalloc(&d_B, size);
    float* d_C; cudaMalloc(&d_C, size);
    // 从主机端拷贝数据到设备端
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // 核函数调用
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // 从设备端拷贝数据到主机端
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // 释放主机内存
    ...
}
```

> 上例详细代码见[example_vec_add.cu](https://github.com/Deleter-D/CUDA/blob/master/00_CUDA_official_documentation/02_example_vec_add.cu)，代码中的注释是关于GPU算子开发的基本思路。

也可以通过`cudaMallocPitch()`和`cudaMalloc3D()`来分配线性内存。推荐用于2D或3D数组的分配，这样可以确保分配被适当填充，以满足内存对齐要求。同时可以保证在访问行地址或者执行2D数组与其他设备内存区域的拷贝操作时的性能（2D或3D内存拷贝使用`cudaMemcpy2D()`与`cudaMemcpy3D()`）。

必须使用返回的pitch（或stride）来访问数组元素，下面以分配一个`width * height`的二维浮点型数组为例。

```c++
// 主机代码
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// 设备代码
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
```

下面以分配`width * height * depth`的三维浮点型数组为例。

```c++
// 主机代码
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// 设备代码
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
```

下面是通过运行时API访问全局变量的各种方式的例子。

```c++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
```

`cudaGetSymbolAddress()`可以获取声明在全局内存空间中的已分配内存的变量地址，通过`cudaGetSymbolSize()`来获取分配内存的大小。

### 内存传输

内存传输使用`cudaMemcpy()`函数，其最后一个参数用来指定数据拷贝方向，有四个取值：

- `cudaMemcpyHostToHost`；
- `cudaMemcpyHostToDevice`；
- `cudaMemcpyDeviceToHost`；
- `cudaMemcpyDeviceToDevice`。

如果目的地址和源地址与最后一个参数指定的方向不一致，则`cudaMemcpy()`的行为是未定义的。大多数情况下该函数是同步的。

### 锁页主机内存（固定主机内存）

CUDA运行时提供了一些函数，来允许使用锁页主机内存（Page-Locked Host Memory）,也称为固定主机内存（Pinned Host Memory），与`malloc()`分配的可分页主机内存相对。

- `cudaHostAlloc()`和`cudaFreeHost()`分配并释放锁页主机内存；
- `cudaHostRegister()`将`malloc()`分配的内存中某范围内的页面锁定。

使用锁页主机内存的优势：

- 锁页主机内存与设备内存之间的数据搬移可以与[异步并发执行](https://deleter-d.github.io/posts/4919/#异步并发执行)中提到的某些设备的核函数并发执行；
- 在某些设备上，锁页主机内存可以映射到设备的地址空间中，无需在主机和设备之间搬移数据，[映射内存](#映射内存（零拷贝内存）)中有详细说明；
- 在具有前端总线（Front-side Bus, FSB）的系统上，若主机内存被分配为锁页内存，则主机内存和设备内存之间的带宽会变高；若主机内存还被分配为写组合内存，则带宽会更大，详见[写组合内存](#写组合内存)。

> 分配的主机内存默认是可分页的（pageable），但GPU不能在可分页主机内存上安全地访问数据。因为当主机的操作系统在物理位置上移动这些数据的时候，GPU时无法控制的。
>
> 当从可分页主机内存传输数据到设备时，CUDA驱动程序首先分配临时的锁页内存，将主机源数据拷贝到锁页内存中后，再将锁页内存中的数据拷贝到设备中。
>
> 我们对比在相同数据量下，可分页内存与设备内存之间的拷贝性能与锁页内存与设备内存之间的拷贝性能，详细代码参考[pageable_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/02_pageable_memory.cu)与[page_locked_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/03_page_locked_memory.cu)。使用`nsys`中的`nvprof`来分析内存拷贝操作的耗时。
>
> 可分页内存与设备内存之间的拷贝耗时如下。
>
> ```
> Time (%)  Total Time (ns)  Count   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)      Operation     
> --------  ---------------  -----  -----------  -----------  ---------  ---------  -----------  ------------------
>   50.0        1,144,966      1  1,144,966.0  1,144,966.0  1,144,966  1,144,966          0.0  [CUDA memcpy HtoD]
>   50.0        1,142,693      1  1,142,693.0  1,142,693.0  1,142,693  1,142,693          0.0  [CUDA memcpy DtoH]
> ```
>
> 锁页内存与设备内存之间的拷贝耗时如下。
>
> ```
> Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)      Operation     
> --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ------------------
>   51.9          686,915      1  686,915.0  686,915.0   686,915   686,915          0.0  [CUDA memcpy HtoD]
>   48.1          637,475      1  637,475.0  637,475.0   637,475   637,475          0.0  [CUDA memcpy DtoH]
> ```
>
> 可以观察到，锁页内存与设备内存之间的拷贝耗时要大幅度小于可分页内存与设备内存之间的拷贝操作。

#### 可移植内存

锁页内存可以与系统中的任何设备结合使用，但默认情况下，上述提到的锁页内存的优势只有分配这片锁页内存的设备可以享受到（如有与该设备共享统一地址空间（详见[统一虚拟地址空间](#统一虚拟地址空间)）的设备，则这些设备也能享受这些优势）。

可以通过将`cudaHostAllocPortable`标志传给`cudaHostAlloc()`来分配锁页内存，或将`cudaHostRegisterPortable`标志传给`cudaHostRegister()`来锁定页面，使得所有设备都能够享受上面提到的优势。

#### 写组合内存

默认情况下，锁页主机内存是作为可缓存状态申请的。此外有一种可选的申请方式，通过将`cudaHostAllocWriteCombined`标志传给`cudaHostAlloc()`来申请写组合内存（Write-Combining Memory）。

写组合内存释放了主机的L1和L2缓存，使程序的其他部分可以使用更多的缓存。此外，在PCI-E总线上传输时，写组合内存不会被窥探，使得传输性能提高40%。

从主机的写组合内存中读取数据速度非常慢，故写组合内存通常只用于仅主机写入内存的情况。

应该避免在写组合内存上使用CPU原子指令，因为不是所有CPU实现都能保证该功能。

#### 映射内存（零拷贝内存）

>  通常情况下，主机不能直接访问设备变量，设备也不能直接访问主机变量。但有一种主机和设备都可以访问的内存——零拷贝内存。

通过将`cudaHostAllocMapped`标志传给`cudaHostAlloc()`，或将`cudaHostRegisterMapped`标志传给`cudaHostRegister()`，可以使锁页主机内存映射到设备的地址空间中。因此，这样的内存区域通常有两个地址：一个在主机内存中，由`cudaHostAllco()`或`malloc()`返回；另一个在设备内存中，可以使用`cudaHostGetDevicePointer()`来检索，从而在核函数中访问该内存空间。

唯一的例外是使用`cudaHostAlloc()`分配的指针，以及主机和设备使用统一地址空间（详见[统一虚拟地址空间](#统一虚拟地址空间)）的情况下。

直接从核函数中访问主机内存并不能提供与设备内存相同的带宽，但确实具有一些优势：

- 无需在设备内存中分配空间，也不需要在主机内存和设备内存之间搬移数据，会根据核函数的需要隐式地进行数据传输；
- 无需使用流（详见[并发数据传输](https://deleter-d.github.io/posts/4919/#并发数据传输)）来使数据传输和核函数同时执行，核函数发起的数据传输将自动地与核函数同时执行。

但是，由于主机和设备共享映射后的锁页内存，因此程序必须使用流或事件同步内存访问（详见[异步并发执行](https://deleter-d.github.io/posts/4919/#异步并发执行)），以避免任何写后读、读后写或写后写等潜在危险行为。

为了能够检索到所有映射后的锁页内存的设备指针，在执行任何CUDA调用之前，必须通过将`cudaDeviceMapHost`标志传给`cudaSetDeviceFlags()`来启动锁页内存映射。否则`cudaHostGetDevicePointer()`将返回一个错误。

如果设备不支持锁页内存的映射，`cudaHostGetDevicePointer()`也会返回一个错误。程序可以通过检查设备属性`canMapHostMemory`来判断是否支持该功能，若支持，则该属性为1。

值得注意的是，从主机或其他设备的角度来看，在映射后的锁页内存上的原子操作并不是原子操作（详见官方文档[Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)）。

还要注意的是，从主机和其他设备的角度来看，CUDA运行时要求由设备发起的对主机内存的1字节、2字节、4字节和8字节天然对齐的加载和存储保留为单一访问。在某些平台上，对内存的原子操作可能会被设备分解为单独的加载和存储操作。这些加载和存储操作对天然对齐访问的保留有相同的要求。例如，某PCI-E拓扑将8字节天然对齐写入拆分为两个4字节写入，CUDA运行时不支持在主机和设备之间基于这种PCI-E总线拓扑进行访问。

> 我们尝试利用映射内存来执行一个向量求和的操作，对比使用设备内存和映射内存的情况，详细代码参考[mapped_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/04_mapped_memory.cu)。经过分析不同数据量情况下的性能，计算减速比，可以总结出下表。$减速比=\frac{使用映射内存的耗时}{使用设备内存的耗时}$。
>
> | 数据量 | 设备内存（ns） | 映射内存（ns） | 减速比  |
> | ------ | -------------- | -------------- | ------- |
> | 1KB    | 1,312          | 2,688          | 2.0488  |
> | 4KB    | 1,344          | 3,200          | 2.3810  |
> | 16KB   | 1,312          | 4,800          | 3.6585  |
> | 64KB   | 1,472          | 8,351          | 5.6732  |
> | 256KB  | 1,856          | 24,511         | 13.2064 |
> | 1MB    | 5,312          | 89,950         | 16.9334 |
> | 4MB    | 27,232         | 350,232        | 12.8610 |
> | 16MB   | 105,405        | 1,377,504      | 13.0687 |
> | 64MB   | 428,246        | 5,514,658      | 12.8773 |
>
> 从这样的结果可以看出，如果想要在主机和设备间共享的少量数据，映射内存是一个不错的选择。但对于大的数据量来说，映射内存并不是好的选择，会导致性能显著的下降。

### 统一虚拟地址空间

从CUDA 4.0开始引入了一种特殊的寻址方式，成为统一虚拟寻址（UVA）。通过CUDA API分配的所有主机内存以及支持UVA的设备分配的所有设备内存都在此虚拟地址范围内。

- 通过CUDA分配的任何主机内存，或使用统一地址空间的设备内存，可以使用`cudaPointerGetAttributes()`来获取指针的信息。
- 当与使用统一地址空间的任何设备之间发生内存拷贝时，`cudaMemcpy*()`的`cudaMemcpyKind`参数可以设置为`cudaMemcpyDefault`。这也适用于未通过CUDA分配的主机指针，只要当前设备使用统一寻址即可。
- 通过`cudaHostAlloc()`分配的内存可以自动移植到使用统一地址空间的设备上（详见[可移植内存](#可移植内存)），并且`cudaHostAlloc()`返回的指针可以直接从这些设备上运行的核函数中使用，即不需要像映射内存中描述的那样通过`cudaHostGetDevicePointer()`获取设备指针。

应用程序可以通过检查`unifiedAddressing`设备属性是否等于1来查询设备是否支持统一地址空间。

> 详细示例代码参考[unified_virtual_address.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/05_unified_virtual_address.cu)。

### 统一内存寻址

CUDA 6.0中引入了统一内存寻址，用于简化CUDA中的内存管理。统一内存中创建了一个托管内存池，内存池中已分配的空间可以用相同的指针在CPU和GPU上访问。底层在统一内存空间中自动在主机和设备之间进行数据传输。

统一内存寻址依赖于统一虚拟寻址（UVA），但它们是完全不同的技术。UAV只是为系统中所有处理器提供了单一的虚拟内存地址空间。但UAV不会自动改变数据的物理位置，这是统一内存寻址的一个特有功能。

托管内存指的是由底层系统自动分配的统一内存，与特定设备的分配内存可以互操作，如它们的创建都使用`cudaMalloc()`。故可以在核函数中使用两种内存：

- 由系统控制的托管内存；
- 由程序明确分配和调用的未托管内存。

在设备内存上的有效CUDA操作也同样适用于托管内存，主要区别是主机也能引用和访问托管内存。

托管内存可以被静态分配，也可以被动态分配。使用`__managed__`修饰符静态声明一个设备变量作为托管变量，该变量可以从主机或设备代码中直接被引用。

```cpp
__device__ __managed__ int y;
```

也可以使用CUDA运行时API`cudaMallocManaged()`来动态分配托管内存。

### 设备内存L2访问管理

当CUDA核心反复访问全局内存中的数据区域时，这种数据访问是持久化的。若数据只被访问一次，则这种数据访问是流式的。从CUDA 11.0开始，计算能力8.0及以上的设备能够影响L2缓存中的数据持久化，从而提供更高的带宽和更低的全局内存访问延迟。

#### 为持久化访问预留的L2缓存

可以预留一部分L2缓存用于持久化全局内存的数据访问，持久化访问优先使用L2缓存的这个部分。只有当持久化访问未使用这一部分时，普通或流式访问才能使用L2缓存。

用于持久化访问的L2预留缓存大小可以在一定范围内调整。

```c++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
// 预留3/4的L2缓存用于持久化访问，或用最大L2持久化缓存大小
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
```

当GPU配置为多实例GPU（MIG）模式时，将禁用L2缓存预留功能。当使用多进程服务（MPS）时，`cudaDeviceSetLimit()`无法改变L2缓存的预留大小，只能通过MPS服务器启动时的环境变量`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`来指定。

#### L2持久化访问策略

访问策略窗口指定了一个连续的全局内存区域及其持久化属性。

下面的例子使用CUDA流（CUDA Stream）设置L2持久化访问窗口。

```c++
// 流级别属性数据结构
cudaStreamAttrValue stream_attribute;
// 全局内存数据指针
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
// 持久化访问的总字节数，必须小于cudaDeviceProp::accessPolicyMaxWindowSize
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;

// 缓存命中率
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;
// 缓存命中时的访存方式
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
// 缓存未命中时的访存方式
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

// 将属性配置到cudaStream_t类型的CUDA流中
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

当核函数在CUDA流中执行时，全局内存范围`[ptr..ptr+num_bytes)`内的数据比其他地方的数据更有可能被持久化在L2缓存中。

下面的例子是L2缓存持久化在CUDA图核结点（CUDA Graph Kernel Node）中的应用。

```c++
// 核级别的属性数据结构
cudaKernelNodeAttrValue node_attribute;
// 全局内存数据指针
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
// 持久化访问的总字节数，必须小于cudaDeviceProp::accessPolicyMaxWindowSize
node_attribute.accessPolicyWindow.num_bytes = num_bytes;

// 缓存命中率
node_attribute.accessPolicyWindow.hitRatio  = 0.6;
// 缓存命中时的访存方式
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; 
// 缓存未命中时的访存方式
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

// 将属性配置到cudaGraphNode_t类型的CUDA图核结点中
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

`hitRatio`参数可以指定以`hitProp`方式访存的占比。在上面两个例子中，全局内存区域`[ptr..ptr+num_bytes)`内，60%的内存访问是持久化的，40%的访问是流式的。具体哪些内存访问是`hitProp`方式是随机的，这个概率是接近`hitRatio`的，概率分布取决于硬件结构和存储范围。

例如，若L2预留预测大小为16KB，`accessPolicyWindow.num_bytes`为32KB：

- 当`hitRatio = 0.5`时，硬件将随机选择32KB窗口中的16KB作为持久化缓存存入预留的L2缓存中；
- 当`hitRatio = 1`时，硬件会尝试在预留的L2缓存区中缓存整个32KB的窗口。但由于预留区小于窗口，缓存行将被移除，以将最近使用的32KB数据中的16KB持久化在L2缓存的预留区中。

`hitRatio`可以用来避免缓存行抖动，从宏观上减少进出L2缓存的数据量。可以利用低于1的`hitRatio`来手动控制不同`accessPolicyWindow`的并发CUDA流能够缓存在L2中的数据量。例如，假设L2的预留缓存大小为16KB，两个不同CUDA流中的核函数是并发的，每个核函数的`accessPolicyWindow`均为16KB，`hitRatio`均为1，在竞争共享的L2缓存时可能会移除彼此的缓存行。但如果两个`accessPolicyWindow`的`hitRatio`均为0.5，则不太可能会清楚自己或对方的持久缓存行。

#### L2访问属性

针对不同的全局内存数据访问，定义了3种类型的访问属性：

- `cudaAccessPropertyStreaming`：伴随流式属性发生的内存访问不太可能持久化在L2缓存中，因为这些访问会被优先清除；
- `cudaAccessPropertyPersisting`：伴随持久化属性发生的内存访问更可能持久化在L2缓存中，因为这些访问会优先被保留在L2缓存中的预留区；
- `cudaAccessPropertyNormal`：该访问属性会将之前持久化的访问强制重置为正常访问。之前CUDA核函数中的持久化属性的内存访问可能会在很长时间内留在L2缓存中，这个时间是远超预期的使用时间的。这种情况会导致后续的核函数不强制使用持久化属性内存访问时可用的L2缓存空间。而使用`cudaAccessPropertyNormal`可以重置访问属性，改变其持久状态，使得后续的核函数可以利用更多的L2缓存。

#### L2持久化示例

下面是为持久访问预留L2缓存的例子，通过CUDA流在CUDA核函数中使用预留的L2缓存并重置。

```c++
cudaStream_t stream;
// 创建CUDA流
cudaStreamCreate(&stream);

// CUDA设备属性变量
cudaDeviceProp prop;
// 查询GPU属性
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize);
// 预留3/4的L2缓存用于持久化访问，或用最大L2持久化缓存大小
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

// 取用户定义的num_bytes和最大窗口大小的较小值作为最终窗口大小
size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);

// 流级别属性数据结构
cudaStreamAttrValue stream_attribute;
// 全局内存数据指针
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);
// 持久化访问的总字节数
stream_attribute.accessPolicyWindow.num_bytes = window_size;
// 缓存命中率
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;
// 缓存命中时的访存方式
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
// 缓存未命中时的访存方式
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

// 将属性配置到CUDA流中
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

for(int i = 0; i < 10; i++) {
    // data1被核函数多次使用
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);
} // [data1..data1 + num_bytes)范围内的数据受益于L2持久化
// 在同一个CUDA流中的不同核函数也能受益于data1的持久化
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);

// 将窗口总字节数设置为0来禁用持久化
stream_attribute.accessPolicyWindow.num_bytes = 0;
// 覆写CUDA流的访问属性
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
// 将L2中的所有持久化缓存行重置为普通状态
cudaCtxResetPersistingL2Cache();

// 由于之前的清除操作，data2当前也可以以普通访存模式受益于L2持久化访问
cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);
```

#### 将L2访问重置为普通

主要有三种方式：

- 通过`cudaAccessPropertyNormal`访问属性来重置支持持久化内存区域的访存属性；
- 通过`cudaCtxResetPersistingL2Cache()`调用将所有持久化L2缓存行重置为普通状态；
- 最终未受影响的缓存行将自动重置为普通状态，在开发过程中不应该依赖于自动重置，因为自动重置所需的时间是不确定的。

#### 管理L2预留缓存的利用率

不同CUDA流中并发执行的多个CUDA核函数可能分配不同的访问策略窗口，但L2的预留缓存部分在所有核函数之间共享，故L2预留区的使用量是所有并发CUDA核函数使用量的总和。当持久化访问的容量超过了L2缓存的容量时，将内存访问指定为持久化状态的收益就会降低。

综上，程序应该考虑以下因素：

- L2缓存预留区的大小；
- 可能并发执行的CUDA核函数；
- 所有可能并发执行的核函数的访问策略窗口；
- 重置L2的时机和方式，以允许普通访问或流式访问以同等优先级利用L2缓存的预留区。

#### 查询L2缓存属性

与L2缓存相关的属性是`cudaDeviceProp`结构体的一部分，可以通过CUDA运行时API`cudaGetDeviceProperties`获取。

CUDA设备属性包括：

- `l2CacheSize`：GPU上可用的L2缓存容量；
- `persistingL2CacheMaxSize`：L2缓存中可用于持久化访存的最大预留容量；
- `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。

#### 控制持久化访存的L2缓存预留大小

用于持久化访存的L2预留缓存大小可以使用CUDA运行时API`cudaDeviceGetLimit`查询，使用`cudaLimit`通过`cudaDeviceSetLimit`设置预留区大小。该设置的最大值是`cudaDeviceProp::persistingL2CacheMaxSize`。

```c++
enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};
```

## 内存访问模式

CUDA执行模型的显著特征之一就是指令必须以线程束为单位进行发布和执行，存储操作也是同样的。在执行内存指令时，线程束中每个线程都提供了一个正在加载或存储的内存地址。在线程束的32个线程中，每个线程都提出了一个包含请求地址的单一内存访问请求，它并由一个或多个设备内存传输提供服务。根据线程束中内存地址的分布，内存访问可以分为不同的模式。

### 对齐与合并访问

全局内存通过缓存来实现加载 / 存储。全局内存是一个逻辑内存空间，可以通过核函数来访问。所有的程序数据最初存在DRAM上，即物理设备内存中。核函数的内存请求通常是在DRAM设备和片上内存间以128字节或32字节的内存事务来实现的。

- 若一个内存访问同时用到了一级和二级缓存，则该访问由128字节的内存事务实现；
- 若一个内存访问只用到了二级缓存，则该访问由32字节的内存事务实现。

> 对于允许使用一级缓存的设备，可以在编译时选择是否启用一级缓存。

![](https://github.com/Deleter-D/Images/assets/56388518/64c5a79c-ade5-4bc5-9da7-52ba127047bd)

一个一级缓存行是128字节，它映射到设备内存中的一个128字节对齐段。若线程束中的每个线程请求4字节的值，则每次请求就会获取128字节的数据，这恰好与一级缓存行大小一致。在优化程序时，需要注意内存访问的两个特性：

- 对齐内存访问：当设备内存事务的第一个地址是缓存粒度的偶数倍时（32B的L2或128B的L1），就会出现对齐内存访问，非对齐的加载会造成带宽浪费；
- 合并内存访问：当一个线程束中全部的32个线程访问一个连续的内存块时，就会出现合并内存访问。

对齐合并内存访问的理想状态是线程束从对齐内存地址开始访问一个连续的内存块。一个理想的对齐合并访问如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/afebf968-8c53-4a8f-a11b-e4b444616387)

这种情况下，只需要一个128字节的内存事务就可以从设备内存中完成读取。但下面这种情况就可能需要3个128字节的内存事务，大大浪费了带宽。

![](https://github.com/Deleter-D/Images/assets/56388518/0c9782e0-53cf-45c9-817b-9ccf148de8e5)

### 全局内存读取

在SM中，数据通过一级和二级缓存、常量缓存、只读缓存3种缓存路径进行传输，具体使用哪种方式取决于所引用的设备内存类型。一、二级缓存是默认路径。若想通过其他两种路径传递数据则需要显式说明，但若想提升性能还要取决于使用的访问模式。全局内存加载是否通过一级缓存取决于设备的计算能力和编译器选项两个因素。使用编译器选项`-Xptxas -dlcm=fg`来禁用一级缓存，`-Xptxas -dlcm=ca`来启动一级缓存。

若一级缓存被禁用，所有对全局内存的加载请求将直接进入二级缓存。若二级缓存未命中，则由DRAM完成请求。每次内存事务可由一个、两个或四个部分执行，每个部分32字节。

若一级缓存被启用，全局内存加载请求首先尝试通过一级缓存。若一级缓存未命中，则请求转向二级缓存。若二级缓存也未命中，则请求由DRAM完成。这种模式下，一个内存加载请求由一个128字节的设备内存事务实现。

#### 缓存加载

缓存加载操作经过一级缓存，在粒度为128字节的一级缓存行上由设备内存事务进行传输。缓存加载可以分为对齐、非对齐、合并、非合并几种情况。

下图为一个理性情况，即对齐与合并内存访问。线程束中的所有请求均在128字节的缓存行范围内。只需要一个128字节的事务，总线利用率为100%，事务中没有未使用数据。

![](https://github.com/Deleter-D/Images/assets/56388518/0b6a8546-c455-4079-9075-b204a276f655)

而下图则是另一种情况，访问是对齐的，但引用的地址不是连续的线程ID，是128字节内的随机值。只需要一个128字节的事务，总线利用率为100%，只有每个线程请求的地址均不同的情况下，该事务中才没有未使用数据。

![](https://github.com/Deleter-D/Images/assets/56388518/fc2d0442-43db-48ad-9f65-fd1b5f469210)

下图中线程束请求32个连续的4字节非对齐数据。需要两个128字节的事务，总线利用率为50%，两个事务中各有一半的数据是未使用的。

![](https://github.com/Deleter-D/Images/assets/56388518/49798f58-2d10-4625-a09c-aa090c8a0977)

下图线程束中的所有线程都请求相同的地址。需要一个128字节的事务，若请求的值是4字节的，则总线利用率为3.125%（$4\div 128$）。

![](https://github.com/Deleter-D/Images/assets/56388518/2b99e9cc-fbf9-4478-98c8-d8d1dc259eae)

下图则是最坏的情况，线程束中线程请求分散于全局内存中的32个不同地点。地址需要占用N个缓存行（$0<N\le32$），需要N个128字节的事务。

![](https://github.com/Deleter-D/Images/assets/56388518/c1d31787-147f-4fd0-bfd0-69c57914c87a)

#### 没有缓存的加载

没有缓存的加载不经过一级缓存，它在内存段的粒度上（32B）而非缓存池的粒度（128B）执行。这种更细粒度的加载，可以为非对齐或非合并的内存访问带来更好的总线利用率。

下图是对齐与合并的内存访问，128字节请求的地址占用了4个内存段，总线利用率为100%。

![](https://github.com/Deleter-D/Images/assets/56388518/f2e25ece-afa2-405d-98a0-333c93154d87)

下图的内存访问是对齐的，但线程访问是不连续的，而是在128个字节范围内随机进行。只要每个线程请求唯一的地址，则地址将占用4个内存段，且不会有加载浪费。这样的随机访问不会抑制核函数性能。

![](https://github.com/Deleter-D/Images/assets/56388518/b7bfd208-dcbc-4dcc-997c-f20c754d44f0)

下图中线程束请求32个连续的4字节元素，但加载没有对齐。请求的地址最多落在5个内存段内，总线利用率至少为80%。与类似情况的缓存加载相比，非缓存加载会提升性能，因为加载了更少的未请求字节。

![](https://github.com/Deleter-D/Images/assets/56388518/5b731cec-9862-4e87-be38-a15915a688ae)

下图线程束中所有线程请求相同的数据。地址落在一个内存段内，总线利用率为12.5%（$4\div 32$）。在这种情况下，非缓存加载的性能也是优于缓存加载的。

![](https://github.com/Deleter-D/Images/assets/56388518/de2ad730-291b-44fd-9df8-98adfb3fc56c)

下图则是最坏的情况，线程束请求32个分散在全局内存中的不同地方。请求的128个字节最多落在N个32字节的内存段内，而不是N个128字节的缓存行内，所以相比于缓存加载，即便是最坏情况也有所改善。

![](https://github.com/Deleter-D/Images/assets/56388518/0c36a2e3-aa91-47e0-a49d-10c359b5849f)

> [read_segment.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/06_read_segment.cu)是一个非对齐读取的示例，实现一个带偏移量的向量求和核函数。
>
> ```cpp
> __global__ void sumArraysReadOffset(float *A, float *B, float *C, const int size, int offset)
> {
>  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
>  unsigned j = tid + offset;
>  if (tid < size)
>      C[tid] = A[j] + B[j];
> }
> ```
>
> 这样可以通过`offset`来强制其进行非对齐内存访问，对不同的`offset`性能测试的结果如下。
>
> ```
> readOffset<<<8192, 512>>>       offset    0     elapsed 0.094464 ms
> readOffset<<<8192, 512>>>       offset   11     elapsed 0.102912 ms
> readOffset<<<8192, 512>>>       offset  128     elapsed 0.094112 ms
> ```
>
> 可以看到在`offset`为11的情况下速度是最慢的，此时两个输入向量的读取是非对齐的。我们借助`ncu`来分析这三种情况的全局加载效率和全局加载事务。
>
> ```
> 												全局加载效率	全局加载事务
> readOffset<<<8192, 512>>>       offset    0       100%		1048576
> readOffset<<<8192, 512>>>       offset   11        80%		1310716
> readOffset<<<8192, 512>>>       offset  128       100%		1048544
> ```
>
> 关于这里的全局加载效率，在《CUDA C编程权威指南》一书中，在开启一级缓存的情况下，全局加载效率仅有50%左右，但禁用一级缓存后提升到了80%。但笔者测试了开启和禁用一级缓存两种情况，加载效率均为80%。

#### 只读缓存

只读缓存最初是预留给纹理内存加载使用的，对计算能力3.5以上的GPU，只读缓存也支持使用全局内存加载代替一级缓存。

只读缓存的加载粒度是32字节，有两种方式可以指导内存通过只读缓存读取：

- 使用函数`__ldg`；
- 在间接引用的指针上使用修饰符。

例如下面的核函数。

```cpp
__global__ void copyKernel(int *out, int *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}
```

可以通过在核函数内部使用`__ldg`来通过只读缓存直接对数组进行读取访问。

```cpp
__global__ void copyKernel(int *out, int *in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = __ldg(&in[idx]);
}
```

也可以将限制修饰符`__restrict__`应用到指针上，该修饰符会使`nvcc`编译器将指针识别为无别名指针。`nvcc`将自动通过只读缓存来指导无别名指针的加载。

```cpp
__global__ void copyKernel(int * __restrict__ out, const int * __restrict__ in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}
```

### 全局内存写入

内存的存储操作相对简单，存储操作不能使用一级缓存进行，在发送到设备内存之前只通过二级缓存。存储操作在32字节段的粒度上被执行。内存事务可以同时被分为一段、两段或四段。

下图中为最理想的情况，内存访问是对齐的，且线程束中的所有线程访问一个连续的128字节范围。存储请求由一个四段事务实现。

![](https://github.com/Deleter-D/Images/assets/56388518/520844ca-de46-4121-a6f6-bec28934665e)

下图的内存访问是对齐的，但地址分散在192字节范围内，存储请求由三个一段事务实现。

![](https://github.com/Deleter-D/Images/assets/56388518/356bc0e5-565a-4a11-b1de-7672cf29146a)

下图中内存访问同样是对齐的，且地址访问在一个连续的64字节范围内，存储请求由一个两段事务实现。

![](https://github.com/Deleter-D/Images/assets/56388518/96106945-ac2c-447e-9382-ff21603fa0ef)

> [write_segment.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/07_write_segment.cu)是一个非对齐写入的示例，实现一个带偏移量的向量求和核函数。
>
> ```cpp
> __global__ void sumArraysWriteOffset(float *A, float *B, float *C, const int size, int offset)
> {
>  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
>  unsigned j = tid + offset;
>  if (j < size)
>      C[j] = A[tid] + B[tid];
> }
> ```
>
> 与上面非对齐读取的例子不同，这次将`C`与`A`、`B`的索引颠倒了过来。通过`offset`来强制其进行非对齐写入，性能测试结果如下。
>
> ```
> writeOffset<<<8192, 512>>>       offset    0     elapsed 0.109920 ms
> writeOffset<<<8192, 512>>>       offset   11     elapsed 0.111616 ms
> writeOffset<<<8192, 512>>>       offset  128     elapsed 0.110592 ms
> ```
>
> 类似地，利用`ncu`来分析全局存储效率和全局存储事务。
>
> ```
> 												全局存储效率	全局存储事务
> writeOffset<<<8192, 512>>>       offset    0       100%		 524288
> writeOffset<<<8192, 512>>>       offset   11        80%		 655358
> writeOffset<<<8192, 512>>>       offset  128       100%		 524272
> ```

值的注意的是，若写入的两个地址同属于一个128字节区域，但不属于一个对齐的64字节区域，则会执行一个四段事务，而不是两个一段事务。

### 结构体数组与数组结构体

C语言中有两种数据组织方式：

- 数组结构体（AoS）；
- 结构体数组（SoA）。

假设要存储一组成对的浮点数，两种不同的方式如下。

AoS方式：

```cpp
struct innerStruct {
    float x;
    float y;
};
struct innerStruct myAoS[N];
```

SoA方式：

```cpp
struct innerArray {
    float x[N];
    float y[N];
};
struct innerArray mySoA;
```

观察两种存储方式的内存布局。

![](https://github.com/Deleter-D/Images/assets/56388518/b71b43ca-fd5e-40d8-a1f9-3001d48eb20b)

可以发现SoA模式充分利用了GPU的内存带宽，由于没有相同字段元素的交叉存取，GPU上的SoA布局提供了合并内存访问，可以对全局内存实现更高效的利用。

> [array_of_structure.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/08_array_of_structure.cu)和[structure_of_array.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/09_structure_of_array.cu)是使用AoS模式和SoA模式下的对比，性能测试结果如下。
>
> ```
> AoS		innerStruct<<<8192, 128>>>      elapsed 0.033280 ms
> SoA		innerArray<<<8192, 128>>>       elapsed 0.032640 ms
> ```
>
> 通过分析它们的全局加载和存储效率可以印证上面的观点，即SoA布局充分利用了GPU的内存带宽。
>
> ```
> 									  全局加载效率	全局存储效率
> AoS		innerStruct<<<8192, 128>>>		 50%	 	   50%
> SoA		innerArray<<<8192, 128>>>       100%		  100%
> ```

### 性能调整

优化设备内存带宽利用率有两个目标：

- 对齐及合并内存访问，以减少带宽的浪费；
- 足够的并发内存操作，以隐藏内存延迟。

#### 展开技术

将之前提到的非对齐读取的例子[read_segment.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/06_read_segment.cu)，将其循环展开。

```cpp
__global__ void readOffsetUnroll4(float *A, float *B, float *C, const int size, int offset)
{
    unsigned int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int j = tid + offset;
    if (j + 3 * blockDim.x < size)
    {
        C[tid] = A[j] + B[j];
        C[tid + blockDim.x] = A[j + blockDim.x] + B[j + blockDim.x];
        C[tid + blockDim.x * 2] = A[j + blockDim.x * 2] + B[j + blockDim.x * 2];
        C[tid + blockDim.x * 3] = A[j + blockDim.x * 3] + B[j + blockDim.x * 3];
    }
}
```

性能测试结果如下。

```
offset<<<8192, 512>>>   offset    0     elapsed 0.107520 ms
unroll4<<<2048, 512>>>  offset    0     elapsed 0.099104 ms
offset<<<8192, 512>>>   offset   11     elapsed 0.109216 ms
unroll4<<<2048, 512>>>  offset   11     elapsed 0.100640 ms
offset<<<8192, 512>>>   offset  128     elapsed 0.108352 ms
unroll4<<<2048, 512>>>  offset  128     elapsed 0.098976 ms
```

分析其全局加载和存储效率，以及全局加载和存储事务。

```
										    全局加载效率 全局存储效率 全局加载事务 全局存储事务
offset<<<8192, 512>>>		offset   11        80%	     100%	   1310704	  524284
unroll4<<<2048, 512>>>		offset   11        80%	     100%	   1310716	  524287
```

> 这里笔者的测试结果没有太大的差距，原因是笔者开启和禁用一级缓存两种情况下，未展开的核函数差别本就不大。所以即使展开之后，性能提升也不明显，但这个优化手段是值的参考的。

#### 增大并行性

用不同的线程块大小测试上面展开的核函数，结果如下。

```
unroll4<<<1024, 1024>>> offset    0     elapsed 0.099104 ms
unroll4<<<2048, 512>>>  offset    0     elapsed 0.095840 ms
unroll4<<<4096, 256>>>  offset    0     elapsed 0.096096 ms
unroll4<<<8192, 128>>>  offset    0     elapsed 0.096320 ms

unroll4<<<1024, 1024>>> offset   11     elapsed 0.098624 ms
unroll4<<<2048, 512>>>  offset   11     elapsed 0.097504 ms
unroll4<<<4096, 256>>>  offset   11     elapsed 0.097568 ms
unroll4<<<8192, 128>>>  offset   11     elapsed 0.097408 ms
```

不管是对齐的还是非对齐的访问，增大并行性都可以带来一些提升。

## 核函数可达到的带宽

### 内存带宽

大多数核函数对内存带宽非常敏感，也就是说它们有内存带宽限制。全局内存中数据的排布方式，以及线程束访问该数据的方式都对带宽有显著的影响。一般分为两个概念：

- 理论带宽：当前硬件可以实现的绝对最大带宽；
- 有效带宽：核函数实际达到的带宽，是测量带宽，公式如下。

$$
有效带宽\text{(GB/s)}=\frac{读字节数+写字节数\times10^{-9}}{运行时间}
$$

### 矩阵转置问题

![](https://github.com/Deleter-D/Images/assets/56388518/56e8a8e5-3dec-440f-a1a9-24342d83fad2)

在主机端利用错位转置算法可以很容易的实现上述操作。

```cpp
void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
            out[ix * ny + iy] = in[iy * nx + ix];
}
```

观察原矩阵和转置矩阵的内存数据排布。

![](https://github.com/Deleter-D/Images/assets/56388518/6918a8de-27cf-4a21-a9f7-8abfc8cb11f9)

可以很容易的分析出，读取过程是访问原矩阵的行，是合并访问，写入过程是访问转置矩阵的列，是交叉访问。

核函数有两种主要方式来实现矩阵的转置：

- 按行读取，按列存储；![](https://github.com/Deleter-D/Images/assets/56388518/5f600f4f-92db-4800-aa22-1179c1ea0b3f)
- 按列读取，按行存储。![](https://github.com/Deleter-D/Images/assets/56388518/2f4e86e1-caf1-4398-b895-0cac22b44565)

#### 为转置核函数设置性能的上限和下限

实现两个核函数，一个读取和存储都按行，另一个读取和存储都按列。

```cpp
__global__ void copyRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[iy * nx + ix] = in[iy * nx + ix];
}

__global__ void copyCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[ix * ny + iy] = in[ix * ny + iy];
}
```

这两个核函数可以分别测得与转置操作相同内存操作情况下，全部使用合并访问（按行读写）以及全部使用交叉访问（按列读写）的有效带宽。

| 核函数    | 带宽（GB/s） | 备注 |
| --------- | ------------ | ---- |
| `CopyRow` | 1367.11      | 上限 |
| `CopyCol` | 595.78       | 下限 |

#### 朴素转置

分别实现[矩阵转置问题](#矩阵转置问题)中提到的两种转置方式，即按行加载按列存储与按列加载按行存储。

```cpp
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[ix * ny + iy] = in[iy * nx + ix];
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny)
        out[iy * nx + ix] = in[ix * ny + iy];
}
```

像上面那样测试有效带宽，对比结果如下，同时分析其全局加载、存储吞吐量和全局加载、存储效率。

| 核函数     | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） | 备注               |
| ---------- | ------------ | ------------------ | ------------------ | ------------- | ------------- | ------------------ |
| `NaiveRow` | 321.25       | 126.85             | 507.42             | 100           | 25            | 合并读取，交叉存储 |
| `NaiveCol` | 911.80       | 609.64             | 152.41             | 25            | 100           | 交叉读取，合并存储 |

可以发现两种方式的性能相近，这是因为在交叉读取的过程中，会有数据进入一级缓存。虽然读取的数据不连续，但在后续的读取过程中，仍然有可能发生缓存命中。禁用一级缓存后，有效带宽表现如下，

| 核函数     | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） | 备注                             |
| ---------- | ------------ | ------------------ | ------------------ | ------------- | ------------- | -------------------------------- |
| `NaiveRow` | 330.99       | 128.38             | 513.50             | 100           | 25            | 合并读取，交叉存储，禁用一级缓存 |
| `NaiveCol` | 489.07       | 472.76             | 118.19             | 25            | 100           | 交叉读取，合并存储，禁用一级缓存 |

可以看到，没有一级缓存的帮助后，交叉读取的有效带宽下降了。对于`NaiveCol`实现来说，由于写入是合并的，存储请求未被重复执行。但由于交叉读取，多次重复执行了加载请求。即使不是最好的加载方式，但有一级缓存的帮助，也能限制交叉读取对性能的负面影响。

> 后面的讨论都默认启用一级缓存。

#### 展开转置

利用循环展开技术改进两个朴素转置核函数。

```cpp
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;

    if (ix + blockDim.x * 3 < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
        out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
    }
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int ti = ix * ny + iy;
    unsigned int to = iy * nx + ix;
    if (ix + blockDim.x * 3 < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + blockDim.x] = in[ti + ny * blockDim.x];
        out[to + blockDim.x * 2] = in[ti + ny * blockDim.x * 2];
        out[to + blockDim.x * 3] = in[ti + ny * blockDim.x * 3];
    }
}
```

进行性能测试，总结如下。

| 核函数       | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） | 备注                     |
| ------------ | ------------ | ------------------ | ------------------ | ------------- | ------------- | ------------------------ |
| `Unroll4Row` | 72.88        | 117.29             | 469.16             | 100           | 25            | 合并读取，交叉存储，展开 |
| `Unroll4Col` | 324.44       | 1843.45            | 465.67             | 25            | 100           | 交叉读取，合并存储，展开 |

启用一级缓存，并展开后，可以观察到`Unroll4Col`的加载吞吐量有了质的提升。

#### 对角转置

当启动一个线程块网格时，线程块会被分配给SM。虽然编程模型可能将网格抽象成一维、二维或三维，但在硬件看来所有块都是一维的。当启动一个核函数时，线程块被分配给SM的顺序由块ID来确定。一开始可能还会以顺序来分配线程块，直到所有SM被完全占满。由于线程块完成的速度和顺序是不确定的，随着核函数的执行，活跃的线程块ID将变得不太连续。

虽然无法直接调控线程块的顺序，但可以利用对角坐标来间接调控，下图展示了直角坐标与对角坐标的区别。

![](https://github.com/Deleter-D/Images/assets/56388518/eeb536e8-ace6-494b-8e52-72ac1016e711)

可以利用对角坐标来确定线程块的ID，但仍需要直角坐标来访问数据。将`blockIdx.x`和`blockIdx.y`当作对角坐标后，对于方阵来说，可以用如下映射关系来访问正确的数据块。

```
blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
blk_y = blockIdx.x;
```

这里的`blk_x`和`blk_y`即为线程块对应的直角坐标。下面分别实现行读列写和列读行写的对角转置核函数。

```cpp
__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny)
{
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int blk_y = blockIdx.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[ix * ny + iy] = in[iy * nx + ix];
}

__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny)
{
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int blk_y = blockIdx.x;

    unsigned int ix = blockDim.x * blk_x + threadIdx.x;
    unsigned int iy = blockDim.y * blk_y + threadIdx.y;

    if (ix < nx && iy < ny)
        out[iy * nx + ix] = in[ix * ny + iy];
}
```

性能测试结果如下。

| 核函数        | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） | 备注                     |
| ------------- | ------------ | ------------------ | ------------------ | ------------- | ------------- | ------------------------ |
| `DiagonalRow` | 330.99       | 133.47             | 533.90             | 100           | 25            | 合并读取，交叉存储，对角 |
| `DiagonalCol` | 910.22       | 592.08             | 148.02             | 25            | 100           | 交叉读取，合并存储，对角 |

通过使用对角坐标来修改线程块的执行顺序，使得基于行读列写的核函数性能大幅度提升，但列读行写的核函数没有什么提升。对角核函数的实现依然可以使用展开技术来优化，但不像直角坐标那样直观。

这种性能的提升与DRAM的并行访问有关。核函数发起的全局内存请求由DRAM分区完成，设备内存中连续的256字节区域被分配到连续的分区。当使用直角坐标线程块时，全局内存的访问无法被均匀分配到DRAM的分区中，就可能发生分区冲突。进而导致内存请求在某些分区中排队，而某些分区一直未被调用。由于对角坐标是一种线程块与数据块之间的非线性映射，所以交叉访问不太可能会落入同一个分区中，进而带来了性能的提升。

#### 使用瘦块增加并行性

对之前实现的列读行写的朴素转置进行测试，分别使用不同的块大小设计，测试结果如下。

| 核函数     | 块大小   | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） |
| ---------- | -------- | ------------ | ------------------ | ------------------ | ------------- | ------------- |
| `NaiveCol` | (32, 32) | 491.14       | 1073.98            | 133.64             | 12.5          | 100           |
| `NaiveCol` | (32, 16) | 718.69       | 1076.43            | 133.88             | 12.5          | 100           |
| `NaiveCol` | (32, 8)  | 739.48       | 872.00             | 109.00             | 12.5          | 100           |
| `NaiveCol` | (16, 32) | 1061.31      | 778.74             | 194.69             | 25            | 100           |
| `NaiveCol` | (16, 16) | 963.76       | 583.35             | 145.84             | 25            | 100           |
| `NaiveCol` | (16, 8)  | 910.22       | 432.14             | 108.03             | 25            | 100           |
| `NaiveCol` | (8, 32)  | 1057.03      | 410.24             | 205.12             | 50            | 100           |
| `NaiveCol` | (8, 16)  | 1064.54      | 327.78             | 163.89             | 50            | 100           |
| `NaiveCol` | (8, 8)   | 731.22       | 233.07             | 116.53             | 50            | 100           |

性能最佳的为`(16, 32)`、`(8, 32)`和`(8, 16)`的块，这种性能提升是由瘦块带来的。可以观察到`(8, 32)`的存储吞吐量是最高的。

我们进一步测试`(8, 32)`的块在各个核函数下的性能表现。

| 核函数       | 块大小  | 带宽（GB/s） | 加载吞吐量（GB/s） | 存储吞吐量（GB/s） | 加载效率（%） | 存储效率（%） | 备注                     |
| ------------ | ------- | ------------ | ------------------ | ------------------ | ------------- | ------------- | ------------------------ |
| `CopyRow`    | (8, 32) | 1071.06      | 206.66             | 206.66             | 100           | 100           | 合并读取，合并存储       |
| `CopyCol`    | (8, 32) | 1057.03      | 422.47             | 422.47             | 50            | 50            | 交叉读取，交叉存储       |
| `NaiveRow`   | (8, 32) | 627.13       | 197.70             | 395.39             | 100           | 50            | 合并读取，交叉存储       |
| `NaiveCol`   | (8, 32) | 1097.98      | 426.77             | 213.39             | 50            | 100           | 交叉读取，合并存储       |
| `Unroll4Row` | (8, 32) | 138.26       | 239.83             | 479.65             | 100           | 50            | 合并读取，交叉存储，展开 |
| `Unroll4Col` | (8, 32) | 341.33       | 1236.83            | 598.49             | 50            | 100           | 交叉读取，合并存储，展开 |

笔者的测试结果与书中有所不同，书中测试结果最好的是`Unroll4Col`，但笔者这里最好的是`NaiveCol`。但从加载吞吐量来说，的确是`Unroll4Col`最优秀。

## 使用统一内存的矩阵加法*

用统一内存的方式实现矩阵加法，可以提高代码的可读性和易维护性，消除所有的显式内存副本。

> 具体代码参考[matrix_sum_managed.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/11_matrix_sum_managed.cu)和[matrix_sum_manual.cu](https://github.com/Deleter-D/CUDA/blob/master/03_global_memory/12_matrix_sum_manual.cu)。

性能测试结果如下。

```
sum matrix managed<<<(128, 128), (32, 32)>>> elapsed 17.284096 ms
sum matrix manual<<<(128, 128), (32, 32)>>> elapsed 0.470016 ms
```

可以观察到，虽然使用统一内存减少了编程的工作量，但性能却大幅度下降。更具体的性能测试如下。

| 任务                | 使用托管内存（ms） | 不使用托管内存（ms） |
| ------------------- | ------------------ | -------------------- |
| 数据初始化          | 355.60             | 354.77               |
| CPU侧计算           | 7.07               | 15.02                |
| CUDA memcpy HtoD    | 14.73              | 10.16                |
| GPU侧计算（核函数） | 19.90              | 0.50                 |
| CUDA memcpy DtoH    | 2.90               | 4.92                 |

可以观察到，在使用托管内存的情况下，`HtoD`任务要花费更长的时间，核函数计算也需要更长的时间。

> 有趣的一点是，在关于数据初始化耗时的测试结果上，笔者与书中的描述相差甚远。在书中，使用托管内存的情况下，数据初始化的耗时要大于不使用托管内存的情况，但笔者的两种方式却相差无几。通过这一点可以看到CUDA在统一内存方面的优化痕迹。
