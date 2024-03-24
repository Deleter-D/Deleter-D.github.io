---
title: CUDA编程——共享内存和常量内存
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 38038
date: 2024-02-20 16:22:30
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## 共享内存概述

GPU中有两种类型的内存：

- 板载内存：全局内存是较大的板载内存，延迟相对较高；
- 片上内存：共享内存是较小的片上内存，延迟相对较低，同时带宽比全局内存高得多。

共享内存可以视作一个可编程的缓存，一般用于块内线程的通信，全局内存数据的可编程管理缓存，以及高速暂存存储器，用于转换数据以优化全局内存访问模式。

### 共享内存

使用内存空间说明符`__shared__`分配共享内存（Shared Memory）。再次回顾下图中的内存层次结构。

![](https://github.com/Deleter-D/Images/assets/56388518/78acbc6e-a8c2-4440-a03d-1d9453aaa3cb)

共享内存比全局内存快的多，可以当作暂存器或由程序管理的高速缓存来使用，以最小化block对全局内存的访问。

下面是直接实现矩阵乘法的例子，未使用共享内存，每个线程读取矩阵A的一行和矩阵B的一列进行计算。矩阵A将被从全局内存中读取`B.width`次，矩阵B将被从全局内存中读取`A.height`次。

```cpp
// 矩阵为行优先存储
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// 定义block的大小
#define BLOCK_SIZE 16

// 矩阵乘法核函数
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // 每个线程计算一个C的元素
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// 主机代码
// 假设矩阵的维数能够被BLOCK_SIZE整除
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // 将A和B加载到设备内存
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    // 在设备上申请结果矩阵
    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // 调用核函数
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // 从设备内存中读取结果矩阵C
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
```

整个访存过程如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/9d695d44-1c15-4282-9d99-fedec7f36db6)

下面是使用共享内存实现矩阵乘法的例子。在下例中，每个block负责计算C的一个子矩阵`Csub`（方阵），block内的每个线程负责计算`Csub`的一个元素，`Csub`等于两个子矩阵的乘积。其中，A的子矩阵维度为`(A.width, block_size)`，行索引与`Csub`的行索引相同，B的子矩阵维度为`(block_size, A.width)`，列索引与`Csub`的列索引相同。

为了适配设备的资源，A、B的两个子矩阵被划分为多个维度为`block_size`的方阵，`Csub`即为这些方阵的乘积之和。将两个对应的方阵从全局内存中加载到共享内存中，其中每个线程加载两个对应方阵中的各一个元素，然后每个线程计算一个乘积。每个参与计算的线程都将乘积累加到一个寄存器中，完成累加后将结果写入全局内存。

通过这种方式，利用了共享内存的速度优势，节省了大量全局内存带宽。A只从全局内存中读取`(B.width / block_size)`次，而B只读取了`(A.height / block_size)`次。

下例还引入了stride，可以用相同的类型有效地表示子矩阵。利用了一个设备函数来获取和设置元素，并从矩阵中构建子矩阵。

```cpp
// 矩阵为行优先存储
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// 获取一个矩阵元素
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// 设置一个矩阵元素
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

// 获取A的BLOCK_SIZE*BLOCK_SIZE子矩阵Asub，通过row、col以及stride和BLOCK_SIZE来定位
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// 定义block的大小
#define BLOCK_SIZE 16

// 矩阵乘法核函数的前置声明
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// 矩阵乘法核函数
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // block的行和列
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // 每个block计算的子矩阵Csub
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // 每个线程计算Csub的一个元素
    float Cvalue = 0;
    // Csub中的线程的行和列
    int row = threadIdx.y;
    int col = threadIdx.x;
    // 循环计算Csub所需的A和B的所有子矩阵，将每对子矩阵相乘并累加结果
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // 获取A的子矩阵Asub
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // 获取B的子矩阵Bsub
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // 用于分别存储Asub和Bsub的共享内存
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // 将Asub和Bsub从设备内存加载到共享内存
        // 每个线程加载每个子矩阵的一个元素
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // 同步，确保在开始计算之前子矩阵加载完整
        __syncthreads();
        // 将Asub和Bsub相乘
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // 同步，确保下一次循环加载A和B的两个新子矩阵之前完成之前的计算
        __syncthreads();
    }
    // 将Csub写入设备内存，每个线程写入一个元素
    SetElement(Csub, row, col, Cvalue);
}

// 主机代码基本一致，此处省略
```

整个访存过程如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/3e53d361-1acc-4446-ac80-221577541d6a)

> 物理上，每个SM都有一个小的低延迟内存池，该内存池被当前正在该SM上执行的线程块中的所有线程共享。
>
> 当每个线程块开始工作时，会分配给它一定数量的共享内存。它的内容和创建时所在的线程块具有相同的生命周期。每个线程束发出的共享内存访问请求，理想情况下，每个请求在一个事务中完成。最坏情况下，每个共享内存的请求在32个不同的事务中顺序执行。若多个线程访问共享内存中的同一个字，一个线程读取该字后，会通过多播把它发给其他线程。
>
> 共享内存被SM中所有常驻线程块划分，因此共享内存是限制设备并行性的关键资源。一个核函数使用的共享内存越多，处于并发活跃状态的线程块就越少。

### 分布式共享内存

在计算能力9.0中引入的线程块集群为线程块集群中的线程提供了访问集群中所有参与线程块的共享内存能力。属于线程块集群的线程可以在分布式地址空间中读取、写入或执行原子，无论目标地址属于当前线程块还是集群中的其他线程块。无论核函数是否使用分布式共享内存（Distributed Shared Memory），共享内存范围仍然属于各自的线程块。分布式共享内存的大小就是每个集群的线程块数乘以每个线程块的共享内存大小。

访问分布式共享内存中的数据需要所有线程块都存在。使用`cluster Group`API中的`cluster.sync()`可以保证所有线程块都已开始执行。还需要确保所有分布式共享内存操作都发生在线程块退出之前。

下面是一个计算直方图的例子，以及如何使用线程块集群在GPU上优化计算。计算直方图的一个标准方法是在每个线程块的共享内存中进行计算，然后在全局内存中执行原子操作。这种方法的一个限制是共享内存的容量，如果直方图的数据量无法与共享内存适配，就只能在全局内存中直接进行原子操作。

而对于分布式共享内存，CUDA提供了一个中间步骤，可以根据直方图的数据量，选择在共享内存、分布式共享内存或全局内存中计算直方图。

下面的CUDA核函数实现了根据直方图数据量选择计算直方图的内存空间。

```c++
#include <cooperative_groups.h>

// 分布式共享内存计算直方图核函数
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, 
                                   const int *__restrict__ input, size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // 集群初始化，获取集群尺寸，并计算局部数据偏移
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //将共享内存中的直方图初始化为0
  }

  // 集群同步，确保集群中的所有block的共享内存初始化为0，并确保所有block都开始执行且同时存在
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    // 寻找正确的直方图数据
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    // 寻找目标block的行索引和行内偏移，以计算分布式共享内存的直方图
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    // 指向目标block共享内存的指针
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // 执行原子更新操作
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // 这里的集群同步是必须的，以确保所有分布式共享内存的操作都已完成
  // 并确保当某个block在访问分布式共享内存时，不会有其他block结束。
  cluster.sync();

  // 使用局部分布式内存中的直方图，计算全局内存的直方图
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

上面的核函数可以在运行时指定集群大小，集群的大小取决于分布式共享内存的需求。若直方图足够小，只能填满一个block的共享内存，则可以启动集群大小为1的核函数。

下面的代码实现了根据共享内存需求动态确定集群大小。

```c++
// Launch via extensible launch
// 
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // 集群大小取决于直方图的大小
  // cluster_size == 1意味着不会有分布式共享内存，只有block局部共享内存
  int cluster_size = 2; // 以集群大小2为例
  int nbins_per_block = nbins / cluster_size; // 每个block的动态共享内存大小
  
  // 分布式共享内存大小为cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, 
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    config.dynamicSmemBytes));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}
```

### 共享内存的分配

共享内存可以被静态或动态分配。上面也提到过，使用`__shared__`修饰符来申请共享内存。

下面的代码申请了一个共享内存中的二维数组。若该声明在核函数内，则该变量的作用域为该核函数中。若在文件的任何核函数外声明，则作用域对所有核函数来说是全局的。

```cpp
__shared__ float tile[size_y][size_x];
```

若共享内存的大小在编译时无法确定，则可以用`extern`关键字进行声明。该声明同样可以在核函数内或核函数外。

```cpp
extern __shared__ int tile[];
```

由于该数组的大小编译时是未知的，所以在核函数被调用时，需要动态分配共享内存。将所需的共享内存字节数在`<<<...>>>`的第三个参数传入。

```cpp
kernel<<<grid, block, size * sizeof(int)>>>(...)
```

注意：只能动态声明一维数组。

### 共享内存存储体和访问模式

#### 内存存储体

为了获取高内存带宽，共享内存被分为32个同样大小的内存模型，被成为存储体，它们可以被同时访问。共享内存是一个一维地址空间。根据GPU的计算能力，共享内存的地址在不同模式下会映射到不同的存储体中。

如果通过线程束发布共享内存加载或存储操作，且在每个存储体上只访问不多于一个的内存地址，则该操作可以由一个内存事务来完成。否则该操作由多个内存事务完成，这样就降低了内存带宽的利用率。

#### 存储体冲突

在共享内存中，多个地址请求落在同一个存储体时，就会发生存储体冲突，这会导致请求被重复执行。硬件会将存储体冲突的请求分割到尽可能多的独立无冲突事务中。当线程束发出共享内存请求时，有3种典型模式：

- 并行访问：多个地址访问多个存储体；
- 串行访问：多个地址访问同一个存储体；
- 广播访问：单一地址读取单一存储体。

并行访问是最常见的模式，这种模式下如果访问的不是范围内的所有地址，则至少有一些地址可以在一个内存事务中完成。最佳情况是每个地址都位于一个单独的存储体中，执行无冲突的共享内存访问。

串行访问是最坏的模式。如果线程束中的32个线程全都访问同一存储体中的不同地址，则需要32个内存事务，且是串行执行的。

广播访问的情况下，线程束中的所有线程都读取同一存储体的同一地址。若一个内存事务被执行，则被访问的字会广播到所有请求线程中。虽然只需要一个内存事务，但因为只有一小部分字节被读取，所以带宽利用率很差。

#### 访问模式

内存存储体的宽度和设备计算能力有关，根据官方文档的描述，总结如下：

- 计算能力2.x、5.x、6.x、7.x、8.x存储体数量均为32个，宽度均为32bit；
- 计算能力3.x比较特殊，存储体数量为32个，宽度64bit。

对于32个32位的存储体来说，每个存储体在每两个时钟周期内都有32位的带宽，连续的32位字映射到连续的存储体中，因此共享内存地址到存储体索引的映射可以由下列公式计算。
$$
存储体索引=(字节地址\div4)\%32
$$
映射关系如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/58dfa384-6b77-41f3-bb32-5def21815fbc)

对于计算能力3.x的设备，存储体宽度为64bit，但它有两种地址模式，64位模式与32位模式。

在64模式下，每个存储体在每两个时钟周期内都有64位的带宽，连续的64位字映射到连续的存储体中，因此共享内存地址到存储体索引的映射可以由下列公式计算。
$$
存储体索引=(字节地址\div8)\%32
$$
在32位模式下，在同一存储体中访问两个32位字不一定是重复操作。在一个时钟周期内读64位并只将32位请求传输给线程是允许的。由于64位宽度的存储体比较少见，这里不过多阐述。

#### 内存填充

内存填充是避免存储体冲突的一种方法。以下图为例，假设只有5个存储体，若所有线程都访问bank0的不同地址，则会发生存储体冲突。但在每N个字后添加一个字，会使得原先同在bank0中的字改变位置，从而解决这种冲突。这里的N是存储体的数量。

![](https://github.com/Deleter-D/Images/assets/56388518/799235c7-5a17-467e-9eb2-50c6e449dced)

### 配置共享内存容量

每个SM都有64KB的片上内存，共享内存和一级缓存共享硬件资源。CUDA为配置一级缓存和共享内存容量提供了两种方法：

- 按设备进行配置；
- 按核函数进行配置。

使用`cudaDeviceSetCacheConfig()`运行时API可以在设备层面设置一级缓存和共享内存大小，可选参数与下面的运行时API一致。

使用`cudaFuncSetCacheConfig()`运行时API可以在核函数层面设置，相关参数已经在[共享内存](https://deleter-d.github.io/posts/47184/#共享内存)阐述过了。

### 同步

CUDA提供的块内同步有两个基本方法：

- 障碍：所有调用的线程等待其余调用的线程达到障碍点；
- 内存栅栏：所有调用的线程必须等到全部内存修改对其余调用线程可见。

#### 弱排序内存模型

GPU线程在不同内存中写入数据的顺序，不一定和这些数据在源代码中访问的顺序相同。一个线程的写入顺序对其他线程可见时，它可能和写操作被执行的实际顺序不一致。

若指令间是相互独立的，线程从不同内存中读取数据的顺序和读指令在程序中出现的顺序不一定相同。

#### 显式障碍

设置障碍点的方法在前面已经见过了，即`__syncthreads()`函数。它要求块内的线程必须等待直到所有线程都到达该点。`__syncthreads()`还确保在障碍点之前，被这些线程访问的所有全局和共享内存对同一块中的所有线程可见。`__syncthreads()`用于协调同一块中线程间的通信。

在使用`__syncthreads()`时，必须保证一个条件能对整个线程块中的线程进行评估，否则执行很可能挂起甚至产生意料之外的问题。

```cpp
if (threadID % 2 == 0) {
    __syncthreads();
} else {
    __syncthreads();
}
```

上面的例子中，可能会导致线程无限期的等待对方，因为块中的所有线程没有到达相同的障碍点。

如果需要进行块间同步，可以尝试在同步点分割核函数并执行多个核函数启动，其中会产生隐式全局障碍以达到预期效果。

#### 内存栅栏

CUDA提供3种内存栅栏：

- 块：`__threadfence_block()`；
- 网格：`__threadfence()`；
- 系统：`__threadfence_system()`。

`__threadfence_block()`保证了栅栏前被调用线程产生的对共享内存和全局内存的所有写操作对栅栏后同一块中的其他线程可见。内存栅栏不执行任何线程同步，所以对于一个块中的所有线程来说，没必要实际执行这个指令。

`__threadfence()`会挂起调用的线程，直到全局内存中的所有写操作对同一网格内的所有线程均可见。

`__threadfence_system()`会挂起调用的线程，以确保该线程对全局内存、锁页主机内存和其他设备内存中的所有写操作对全部设备中的线程和主机线程都可见。

#### 内存同步域

##### 内存栅栏实例

某些CUDA程序可能会因为内存栅栏/刷新操作等待的事务多于CUDA内存一致性模型所需的事务而导致性能下降。

```c++
__managed__ int x = 0;
__device__ cuda::atomic<int, cuda::thread_scope_device> a(0);
__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);
```

Thread 1 (SM)：

```c++
x = 1;
a = 1;
```

Thread 2 (SM)：

```c++
while(a != 1);
assert(x == 1);
b = 1;
```

Thread 3 (CPU)：

```c++
while(b != 1);
assert(x == 1);
```

考虑上述例子，CUDA内存一致性模型将保证断言的条件为真，故在线程2写入`b`之前，线程1对`x`的写入必须对线程3可见。

释放和获取`a`所提供的内存排序仅能够使`x`对线程2可见，但线程3不可见，因为它是设备范围的操作。故由释放和获取`b`所提供的系统范围的内存排序需要确保从线程2发起的写入对线程3可见，同时要保证从线程2可见的其他线程发起的写入也是可见的。这要求称为累积性（cumulativity）。由于GPU在执行时不知道哪些写入在源码级别下是可见的，也不知道哪些写入仅在偶然的时间是可见的，所以它必须为所有活动状态的内存操作撒下一个保守的广域网。

上述情况会使GPU等待源码级别不需要的内存操作，进而使得内存栅栏/刷新操作花费不必要的时间，对整个程序产生干扰。

注意，内存栅栏可以在代码中显式的作为内部结构或原子操作出现，也可以隐式的实现任务边界上的同步关系。

##### 用域隔离流量

从Hopper架构的GPU和CUDA 12.0开始，内存同步域提供了一种减轻上述干扰的方法。使用代码来显式辅助，可以减少GPU的光撒网行为。每次核函数启动都会被赋予一个域ID。写操作和栅栏都用该ID来标识，而栅栏只会对与栅栏域匹配的写操作进行排序。通信的核函数可以放在不同的域中。

使用域时，代码必须使同一GPU上不同域之间的排序或同步需要系统范围的栅栏。而在同一域中，设备范围的栅栏就足够了。这种栅栏范围要求对于累积性来说是必要的，因为一个核函数的写操作不会被另一个域中的核函数发出的栅栏所包围。本质上，累积性是通过确保跨域流量提前刷新到系统范围来满足的。

注意，这会修改`thread_scope_device`的定义。但由于核函数将默认采用域0，所以保证了向后兼容。

##### 在CUDA中使用域

域可以通过新的启动属性`cudaLaunchAttributeMemSyncDomain`和`cudaLaunchAttributeMemSyncDomainMap`来访问。前者在逻辑域`cudaLaunchMemSyncDomainDefault`和`cudaLaunchMemSyncDomainRemote`之间选择，后者提供从逻辑域到物理域的映射。远程域可供核函数进行远程内存访问，以便将其内存流量与本地核函数隔离。选择特定的域不会影响核函数可以合法执行的内存访问。

可以通过设备属性`cudaDevAttrMemSyncDomainCount`来获取域的数量。Hopper有4个域。为了便于代码移植，域功能可以在所有设备上使用，在Hopper架构之前的架构下，该计数将返回1。

详细参考官方文档[Using Domains in CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-domains-in-cuda)。

#### `volatile`修饰符

在全局或共享内存中使用`volatile`修饰符声明一个变量，可以防止编译器优化，编译器优化可能会将数据暂时缓存在寄存器或本地内存中。当使用`volatile`修饰符时，编译器假定任何其他线程在任何时间都可以更改或使用该变量的值。因此，这个变量的任何引用都会直接被编译到全局内存读指令或全局内存写指令中，它们都会忽略缓存。

## 共享内存的数据布局

### 方形共享内存

方形矩阵可以很容易从二维线程索引计算出一维内存偏移，声明一个方形二维共享内存变量。

```cpp
__shared__ int tile[N][N];
```

它可以有两种访问方式：

- `tile[threadIdx.y][threadIdx.x]`
- `tile[threadIdx.x][threadIdx.y]`

显然，两种方式相比，第一种方式拥有更少的存储体冲突，因为邻近线程在最内层数组维度上访问相邻的阵列单元。

#### 访问方式实例

我们分别实现三种读写共享内存的方法：

- 按行写入，按行读取；
- 按列写入，按列读取；
- 按列写入，按行读取。
- 按行写入，按列读取。

> 详细代码参考[square_shared_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/01_square_shared_memory.cu)。

利用`nsys`来分析其耗时，并利用`ncu`来分析它们的共享内存加载和存储事务来体现存储体冲突，结果如下。

|                                                | 耗时 (ns) | 加载事务 | 存储事务 |
| ---------------------------------------------- | --------- | -------- | -------- |
| `writeRowReadRow(int *) (1, 1, 1)x(32, 32, 1)` | 1248      | 32       | 32       |
| `writeColReadCol(int *) (1, 1, 1)x(32, 32, 1)` | 1920      | 1024     | 1024     |
| `writeColReadRow(int *) (1, 1, 1)x(32, 32, 1)` | 1280      | 32       | 1024     |
| `writeRowReadCol(int *) (1, 1, 1)x(32, 32, 1)` | 1280      | 1024     | 32       |

可以看出行读行写的核函数性能最高，加载和存储事务最少，没有存储体冲突。不管是按列读还是写，都会存在存储体冲突，导致加载或存储事务大量增多。

#### 动态共享内存

使用动态声明共享内存的方式实现与上述功能相同的核函数。动态共享内存必须被声明为一个未定大小的一维数组，因此需要基于二维线程索引来计算内存访问索引。

再测试其性能，结果如下。

|                                                       | 耗时 (ns) | 加载事务 | 存储事务 |
| ----------------------------------------------------- | --------- | -------- | -------- |
| `writeRowReadColDynamic(int *) (1, 1, 1)x(32, 32, 1)` | 1280      | 1024     | 32       |

可以发现结果与`writeRowReadCol`相同。

#### 填充静态声明的共享内存

使用前面提到的[内存填充](#内存填充)来解决存储体冲突，并测试其性能。

|                                                       | 耗时 (ns) | 加载事务 | 存储事务 |
| ----------------------------------------------------- | --------- | -------- | -------- |
| `writeRowReadColPadding(int *) (1, 1, 1)x(32, 32, 1)` | 896       | 32       | 32       |

可以发现，通过填充完美的解决了存储体冲突。

#### 填充动态声明的共享内存

实现基于动态共享内存填充的核函数，并测试其性能。

|                                                      | 耗时 (ns) | 加载事务 | 存储事务 |
| ---------------------------------------------------- | --------- | -------- | -------- |
| `writeRowReadColDynPad(int *) (1, 1, 1)x(32, 32, 1)` | 896       | 32       | 32       |

可以发现基于动态共享内存的填充也是有效的。

### 矩形共享内存

将上述的方形共享内存推广到矩形这个更为一般的情况。实现与上述功能相同的几个核函数，并分析其性能。

|                                                       | 耗时 (ns) | 加载事务 | 存储事务 |
| ----------------------------------------------------- | --------- | -------- | -------- |
| `writeRowReadRow(int *) (1, 1, 1)x(32, 32, 1)`        | 1119      | 16       | 16       |
| `writeColReadCol(int *) (1, 1, 1)x(32, 32, 1)`        | 1248      | 256      | 256      |
| `writeColReadRow(int *) (1, 1, 1)x(32, 32, 1)`        | 928       | 16       | 256      |
| `writeRowReadCol(int *) (1, 1, 1)x(32, 32, 1)`        | 992       | 256      | 16       |
| `writeRowReadColDynamic(int *) (1, 1, 1)x(32, 32, 1)` | 960       | 256      | 16       |
| `writeRowReadColPadding(int *) (1, 1, 1)x(32, 32, 1)` | 896       | 16       | 16       |
| `writeRowReadColDynPad(int *) (1, 1, 1)x(32, 32, 1)`  | 896       | 16       | 16       |

> 详细代码参考[rectangle_shared_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/02_rectangle_shared_memory.cu)。

## 减少全局内存访问

### 使用共享内存的并行归约

将之前实现的线程束展开的并行归约作为性能基准，利用共享内存的操作代替全军内存的原地操作，观察两者性能差距。

> 详细代码参考[reduce_with_shared_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/03_reduce_with_shared_memory.cu)。

```
Array Size: 16777216
cpu reduce      elapsed 16.0859 ms      cpu_sum: 206464799
gpu Gemm        elapsed 0.384192 ms     gpu_sum: 206464799      <<<131072, 128>>>
gpu Semm        elapsed 0.278624 ms     gpu_sum: 206464799      <<<131072, 128>>>
Result correct!
```

|                                                              | 全局加载事务 | 全局存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `reduceGmem(int *, int *, unsigned int) (131072, 1, 1)x(128, 1, 1)` | 8912896      | 4325376      |
| `reduceSmem(int *, int *, unsigned int) (131072, 1, 1)x(128, 1, 1)` | 2097152      | 131072       |

可以看到，通过使用共享内存代替全局内存的原地操作，大幅度减少了全局内存事务，从而提升了总体性能。

### 使用展开的并行归约

在使用共享内存进行归约的基础上，再利用技术展开，每个线程处理4个数据块的元素，再次对比它们的性能。

```
Array Size: 16777216
cpu reduce      elapsed 16.0439 ms      cpu_sum: 206464799
gpu Gemm        elapsed 0.384 ms        gpu_sum: 206464799      <<<131072, 128>>>
gpu Semm        elapsed 0.278464 ms     gpu_sum: 206464799      <<<131072, 128>>>
gpu SemmUnroll  elapsed 0.214976 ms     gpu_sum: 206464799      <<<32768, 128>>>
Result correct!
```

分析其全局加载和存储事务。

|                                                              | 全局加载事务 | 全局存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `reduceGmem(int *, int *, unsigned int) (131072, 1, 1)x(128, 1, 1)` | 8912896      | 4325376      |
| `reduceSmem(int *, int *, unsigned int) (131072, 1, 1)x(128, 1, 1)` | 2097152      | 131072       |
| `reduceSmemUnroll(int *, int *, unsigned int) (32768, 1, 1)x(128, 1, 1)` | 2097152      | 32768        |

观察发现，虽然全局加载事务并没有减少，但全局存储事务却减少到了原来的1/4。

### 使用动态共享内存的并行归约

上述基于共享内存的展开并行归约也可以使用动态共享内存，性能表现与使用静态共享内存时接近，这里不再赘述。

## 合并的全局内存访问

使用共享内存可以避免对未合并的全局内存进行访问，矩阵转置是一个典型的例子，读操作是合并的，但写操作是交叉访问的。

### 朴素转置

实现一个朴素转置，核函数如下。

```cpp
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))

__global__ void naiveGmem(float *out, float *in, const int rows, const int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
        out[INDEX(col, row, rows)] = in[INDEX(row, col, cols)];
}
```

再实现一个读写操作都是合并访问的核函数，用来模拟性能的近似上界。

```cpp
__global__ void copyGmem(float *out, float *in, const int rows, const int cols)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
        out[INDEX(row, col, cols)] = in[INDEX(row, col, cols)];
}
```

> 详细代码参考[transpose_with_shared_memory.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/04_transpose_with_shared_memory.cu)。

这里使用$32\times16$的二维线程块来调用，经过测试，上面两个核函数的性能表现如下。

|             | 耗时 (ms) | 等效带宽 (GB/s) |
| ----------- | --------- | --------------- |
| `copyGmem`  | 0.291488  | 0.460457        |
| `naiveGmem` | 0.994176  | 0.135004        |

分析其每次请求中的全局内存事务数量，结果如下。

|                                                              | 全局加载事务 | 全局存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `copyGmem(float *, float *, int, int) (256, 256, 1)x(16, 16, 1)` | 4            | 4            |
| `naiveGmem(float *, float *, int, int) (256, 256, 1)x(16, 16, 1)` | 4            | 32           |

可以发现，由于朴素转置的写操作是交叉访问的，所以每次请求中的全局存储事务要更多。

### 使用共享内存的矩阵转置

可以使用二维共享内存来缓存原始矩阵的数据，从而避免交叉的全局内存写操作。首先从全局内存中读取块内的一行写入共享内存的一行，然后从共享内存读取一列写入全局内存的一行。

![](https://github.com/Deleter-D/Images/assets/56388518/e10502de-f723-49c2-928c-d672958a7ec0)

实现该核函数时，需要注意索引的计算方式。

```cpp
__global__ void transposeSmem(float *out, float *in, const int rows, const int cols)
{
    __shared__ float tile[BDIMY][BDIMX];

    // 原始矩阵索引
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];

    // 由于转置过程中，不仅block需要转置，block内的thread也需要转置
    // 所以利用irow和icol来代替原来的threadIdx的x和y维度
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // 转置矩阵中，blockDim和blockIdx的x维度计算列索引，y维度计算行索引，与原始矩阵相反
    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    __syncthreads();

    if (row < cols && col < rows)
        out[INDEX(row, col, rows)] = tile[icol][irow];
}
```

分析其性能与全局内存事务。

|                 | 耗时 (ms) | 等效带宽 (GB/s) |
| --------------- | --------- | --------------- |
| `copyGmem`      | 0.291072  | 0.461115        |
| `naiveGmem`     | 1.007616  | 0.133203        |
| `transposeSmem` | 0.343520  | 0.390713        |

|                                                              | 全局加载事务 | 全局存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `copyGmem(float *, float *, int, int) (256, 256, 1)x(16, 16, 1)` | 4            | 4            |
| `naiveGmem(float *, float *, int, int) (256, 256, 1)x(16, 16, 1)` | 4            | 32           |
| `transposeSmem(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 4            | 4            |

上面提到了，这种方式虽然在共享内存中读取列的时候依然会发生存储体冲突，但这样的结果已经比直接对全局内存进行交叉写入要好的多。共享内存中的存储体冲突可以通过分析其共享内存事务数量来解释。

|                                                              | 共享加载事务 | 共享存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `transposeSmem(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 8460165      | 533345       |

### 使用填充共享内存的矩阵转置

使用之前提到的[内存填充](#内存填充)技术来优化上面的核函数。

```cpp
__global__ void transposeSmemPad(float *out, float *in, int rows, int cols)
{
    __shared__ float tile[BDIMY][BDIMX + PAD];

    // 原始矩阵索引
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];

    // 转置block中的线程索引
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    row = blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;

    __syncthreads();

    if (row < cols && col < rows)
        out[INDEX(row, col, rows)] = tile[icol][irow];
}
```

这里选择填充2个位置，这样可以完全消除共享内存的存储体冲突。可以通过分析其性能及共享内存事务来证明这一点。

|                    | 耗时 (ms) | 等效带宽 (GB/s) |
| ------------------ | --------- | --------------- |
| `copyGmem`         | 0.292448  | 0.458946        |
| `naiveGmem`        | 0.987136  | 0.135967        |
| `transposeSmem`    | 0.404192  | 0.332064        |
| `transposeSmemPad` | 0.326272  | 0.411368        |

|                                                              | 共享加载事务 | 共享存储事务 |
| ------------------------------------------------------------ | ------------ | ------------ |
| `transposeSmem(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 8460165      | 533345       |
| `transposeSmemPad(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 545540       | 533686       |

### 使用展开的矩阵转置

在上述使用了内存填充的核函数基础上，使用展开技术进行优化，使每个线程处理两个元素。

```cpp
__global__ void transposeSmemUnrollPad(float *out, float *in, int rows, int cols)
{
    // 使用一维的共享内存
    __shared__ float tile[BDIMY][BDIMX * 2 + PAD];

    // 原始矩阵索引
    unsigned int col = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col + blockDim.x < cols)
    {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(row, col, cols)];
        tile[threadIdx.y][threadIdx.x + blockDim.x] = in[INDEX(row, col + blockDim.x, cols)];
    }

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    row = 2 * blockIdx.x * blockDim.x + irow;
    col = blockIdx.y * blockDim.y + icol;
    
    __syncthreads();

    if (row + blockDim.x < cols && col < rows)
    {
        out[INDEX(row, col, rows)] = tile[icol][irow];
        out[INDEX(row + blockDim.x, col, rows)] = tile[icol][irow + blockDim.x];
    }
}
```

进行性能对比后发现，相比使用内存填充的核函数，有略微的提升。

|                          | 耗时 (ms) | 等效带宽 (GB/s) |
| ------------------------ | --------- | --------------- |
| `copyGmem`               | 0.292864  | 0.458294        |
| `naiveGmem`              | 0.987008  | 0.135984        |
| `transposeSmem`          | 0.377856  | 0.355209        |
| `transposeSmemPad`       | 0.326656  | 0.410884        |
| `transposeSmemUnrollPad` | 0.322560  | 0.416102        |

使用展开技术使得更多的内存请求将同时处于运行状态，且会提高读写吞吐量。`ncu`分析设备内存的读写吞吐量可以佐证这一点。

|                                                              | 读吞吐量 (GB/s) | 写吞吐量 (GB/s) |
| ------------------------------------------------------------ | --------------- | --------------- |
| `naiveGmem(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 55.81           | 52.64           |
| `transposeSmem(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 193.00          | 177.18          |
| `transposeSmemPad(float *, float *, int, int) (128, 256, 1)x(32, 16, 1)` | 205.98          | 178.52          |
| `transposeSmemUnrollPad(float *, float *, int, int) (64, 256, 1)x(32, 16, 1)` | 206.65          | 186.28          |

## 常量内存

常量内存对于核函数来说是只读的，但对于主机来说是可读可写的。常量内存位于设备的DRAM上（与全局内存一样），且有一个专用的片上缓存。与一级缓存和共享内存类似，从每个SM的常量缓存中读取的延迟，比直接从常量内存中读取的延迟低得多。每个SM常量缓存大小限制为64KB。

常量内存与之前提到的所有类型的内存有着不同的最优访问模式。在常量内存中，若线程束中的所有线程都访问相同的位置，则这个访问模式是最优的。若线程束中的线程访问不同地址，则访问需要串行。

在全局作用域中必须使用`__constant__`修饰符来声明常量变量。常量内存变量的生命周期与应用程序的生命周期相同，对所有线程都是可访问的，并且可以通过运行时函数对主机也可访问。

由于设备只能读取常量内存，所以常量内存中的值必须通过运行时函数`cudaMemcpyToSymbol()`来初始化。

### 使用常量内存实现一维模板

在数值分析中，模板计算在点的集合上应用一个函数，并使用该函数的输出更新单一点的值。在一维中，位置$x$周围的的九点模板会给如下位置上的值应用一些函数。
$$
\{x-4h,x-3h,x-2h,x-h,x,x+h,x+2h,x+3h,x+4h\}
$$
我们不需要理解这个公式的实际意义，只需要观察到它会将上述的九个点作为输入，产生单一输出。下面使用一个实际公式作为示例。
$$
f'(x)=c_0(f(x+4h)-f(x-4h))+c_1(f(x+3h)-f(x-3h))-c_2(f(x+2h)-f(x-2h))+c_3(f(x+h)-f(x-h))
$$
可以比较容易的观察到，公式中$c_0,c_1,c_2,c_3$这些系数是不变的，所以很适合存入常量内存中。且线程束中的所有线程都是访问这几个常量，这恰好满足常量内存的最优访问模式。

计算过程如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/64aa9167-0992-4040-bd66-786758dbcd9b)

借助共享内存来缓存数据，同时在其两侧添加一些光环数据，类似于卷积中的填充操作，是为了计算的合法性。通过如下核函数来实现整个计算过程。

```cpp
__global__ void stancli1DGPU(float* in, float* out, int size)
{
    // 包含光环的共享内存
    __shared__ float smem[BDIM + 2 * RADIUS];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;

    while (idx < size + RADIUS)
    {
        // 共享内存的索引，为模板计算作准备
        int sidx = threadIdx.x + RADIUS;

        // 将数据部分写入共享内存
        smem[sidx] = in[idx];

        // 将光环部分度写入共享内存
        if (threadIdx.x < RADIUS)
        {
            smem[sidx - RADIUS] = in[idx - RADIUS];
            smem[sidx + BDIM]   = in[idx + BDIM];
        }

        __syncthreads();

        float tmp = 0.0f;

#pragma unroll
        for (int i = 1; i <= RADIUS; i++)
        {
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }

        out[idx] = tmp;

        idx += gridDim.x * blockDim.x;
    }
}
```

> 详细代码参考[constant_stencli.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/05_constant_stencli.cu)。

### 与只读缓存比较

只读缓存实质上是GPU的纹理流水线，用于存储全局内存中的数据。只读缓存是独立的，它拥有从标准全局内存读取的独立内存带宽，所以使用只读缓存可以为受制于内存带宽的核函数提供一些性能优势。

只读缓存不同于常量内存，其最优访问模式是线程束中的线程访问不同的位置。只读缓存的粒度为32字节。

当通过只读缓存访问全局内存时，需要在核函数中向编译器指出数据是只读的，可以通过如下两种方式：

- 内部函数`__ldg()`；
- 全局内存的限定指针；

内部函数`__ldg()`用于代替标准指针解引用，并强制加载通过只读数据缓存。

```cpp
__global__ void kernel(float* output, float* input) {
    ...
    output[idx] += __ldg(&input[idx]);
    ...
}
```

也可以限定指针为`const __restrict__`，以表明它应该通过只读缓存被访问。

```cpp
__global__ void kernel(float* output, const float* __restrict__ input) {
	...
    output[idx] += input[idx];
    ...
}
```

在只读缓存需要很多显式控制，或代码非常复杂以至于编译器无法检测到只读缓存的使用是否安全的情况下，内部函数`__ldg()`是更好的选择。

通过只读缓存加载的数据可以比较大，且能够在一个非统一的模式下进行访问。利用只读缓存来实现上述模板算法的核函数，唯一的区别就是函数声明。

```cpp
__global__ void stancliReadOnly(float* in, float* out, int size, const float* __restrict__ dcoef)
{
    ...
}
```

但要注意的是，不同于常量内存，在核函数调用之前，必须提前分配只读缓存的设备内存。

```cpp
const float h_coef[] = {a0, a1, a2, a3, a4};

// 使用常量内存只需要调用如下运行时API，无需申请设备内存
ERROR_CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS + 1) * sizeof(float)));

// 使用只读缓存时需要提前申请设备内存
float* d_coef;
ERROR_CHECK(cudaMalloc((void**)&d_coef, (RADIUS + 1) * sizeof(float)));
ERROR_CHECK(cudaMemcpy(d_coef, h_coef, (RADIUS + 1) * sizeof(float), cudaMemcpyHostToDevice));
```

使用`nsys`统计两个核函数的耗时可以观察到，对于以广播模式访问的数据来说，常量内存是更适合的。

```
 Time (%)  Total Time (ns)                         Name                         
 --------  ---------------  -----------------------------------------------------
     50.2          408,376  stancliReadOnly(float *, float *, int, const float *)
     49.8          405,463  stancliConstant(float *, float *, int)
```

## 线程束洗牌指令

从计算能力3.0开始加入了一种机制称为洗牌指令（shuffle instruction），只要两个线程在相同的线程束中，就允许这两个线程直接读取另一个线程的寄存器。这种直接的数据交换不是通过共享内存或全局内存来进行的，拥有比共享内存更低的延迟，且在执行数据交换时不消耗额外的内存。

这里引入一个概念——束内线程（lane），线程束中的每一个线程都是束内线程，每个束内线程都有一个唯一的束内线程索引。在一维线程块中，对于一个给定线程的束内线程索引和线程束索引可以通过如下方式计算。

```cpp
laneID = threadIdx.x % 32;
warpID = threadIdx.x / 32;
```

对于多维的线程块，可以将多维线程坐标转换为一维线程索引，再应用上述公式来计算。

### 线程束洗牌指令的不同形式

洗牌指令共有两组，一组用于整型变量，另一组用于浮点型变量。每组有4种形式的洗牌指令：

- 广播传递；
- 向上传递；
- 向下传递；
- 异或传递。

#### 广播传递

```cpp
int __shfl(int var, int srcLane, int width=warpSize);
```

其中`var`是待传递的变量，`srcLane`是提高该变量的线程的束内线程索引。最后一个参数`width`允许将线程束进一步划分为段，每段包含`width`个线程，取值范围是`[2, 32]`，每个段上会执行独立的洗牌操作。对于非32的其他`width`值，线程的束内线程索引可以通过`threadIdx.x % width`来确定。

![](https://github.com/Deleter-D/Images/assets/56388518/4de2e52e-0207-4644-86a2-392b55b5c9f6)

上图展示了`__shfl(val, 2)`的调用示例。

#### 向上传递和向下传递

向上传递和向下传递非常类似，两者的区别仅是传递方向不同。

```cpp
int __shfl_up(int var, unsigned int delta, int width=warpSize);
int __shfl_down(int var, unsigned int delta, int width=warpSize);
```

`delta`参数用来计算提供变量的束内线程索引：

- 向上传递时，当前束内线程接受来自于束内线程索引为`当前束内线程索引 - delta`线程中的变量`var`；
- 向下传递时，当前束内线程接受来自于束内线程索引为`当前束内线程索引 + delta`线程中的变量`var`；

若向上或向下传递过程中没有对应的源束内线程，则线程中的变量保持不变。

![](https://github.com/Deleter-D/Images/assets/56388518/24efc65e-2556-49a2-881c-abbe3baea285)

![](https://github.com/Deleter-D/Images/assets/56388518/2dc7ebab-592a-423e-9a10-518c46fb3c4f)

#### 异或传递

```cpp
int __shfl_xor(int var, int laneMask, int width=warpSize);
```

异或传递较为特殊，它根据自身的束内线程索引与`laneMask`按位异或来确定源束内线程。

![](https://github.com/Deleter-D/Images/assets/56388518/1ba1eb0e-524c-4cc1-b96b-e49d5b763f9c)

上图展示了蝴蝶寻址模式的示例。

上面提到的4种洗牌指令均有单精度浮点数版本。

> **注意：在6.0以后的PTX ISA中，已经不再支持这一系列的洗牌指令，新版本改为了带有`sync`的洗牌指令。**
>
> 上述的几个函数整体变化不大，只是在参数列表的最前面添加了一个参数`mask`：
>
> - `int __shfl_sync(unsigned mask, int var, int srcLane, int width=warpSize);`
> - `int __shfl_up_sync(unsigned mask, int var, int srcLane, int width=warpSize);`
> - `int __shfl_down_sync(unsigned mask, int var, int srcLane, int width=warpSize);`
> - `int __shfl_xor_sync(unsigned mask, int var, int srcLane, int width=warpSize);`
>
> 当然也有对应的单精度浮点数版本，并且拥有64位的版本，即`long`与`double`类型的接口。
>
> 该`mask`参数用于指定参与洗牌指令的束内线程，每个`bit`代表一个束内线程。`mask`给予编译器一个提示，为了保证正确性，所指向的束内线程必须全部参与洗牌指令，这样编译器就会生成一些必要的指令将这些线程重新汇聚起来。
>
> 简单来说，若使用默认的线程束大小，想要使所有束内线程都参与洗牌指令，则`mask`指定为`0xffffffff`即可。
>
> 对于`mask`参数，英伟达官方论坛有一个[帖子](https://forums.developer.nvidia.com/t/what-does-mask-mean-in-warp-shuffle-functions-shfl-sync/67697)对此进行了描述。
>
> 以及[shuffle_instruction.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/06_shuffle_instruction.cu)和[reduce_with_shuffle.cu](https://github.com/Deleter-D/CUDA/blob/master/04_shared_and_constant_memory/07_reduce_with_shuffle.cu)有一系列详细的示例。

