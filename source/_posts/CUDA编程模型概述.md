---
title: CUDA编程模型概述
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 57516
date: 2024-02-20 15:48:18
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## 核函数

CUDA的核函数（Kernel）是通过`__global__`说明符定义的，通过C++扩展语法`<<<...>>>`来启动。下面是一个向量加法的例子。

```c++
// 核函数定义
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // 核函数调用，该核函数包含N个线程
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

> 核函数的编写是无需考虑并行性的，并行调度是由编译器和GPU自动完成的，只需要针对每个线程需要处理的逻辑编写代码即可。
>
> 核函数的返回类型必须是`void`，限定符`__global__`和返回类型的顺序可以调换，上面的核函数也可以写成如下形式。
>
> ```c++
> void __global__ VecAdd(float* A, float* B, float* C);
> ```
>
> 核函数有一些注意事项：
>
> - 核函数只能访问GPU内存；
> - 核函数不能使用变长参数；
> - 核函数不能使用静态变量；
> - 核函数不能使用函数指针；
> - 核函数具有异步性；
>
> 具体例子参考[hello_world.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/01_hello_world.cu)。
>
> CUDA除了核函数还有两种函数，设备函数和主机函数。
>
> - 设备函数使用`__device__`限定符修饰，只能在设备上执行，只能被核函数或其他设备函数调用；
> - 主机函数使用`__host__`限定符修饰，可以与`__device__`限定符同时使用，编译器会针对主机和设备分别编译该函数；
>
> 注意：`__global__`不能与`__host__`或`__device__`同时使用。

> 除此之外，还需要掌握两个常用技巧，一个是错误处理，一个是性能分析。这两个技巧将在整个GPU算子开发和优化过程中起到重要的作用。关于错误处理，示例代码见[error_handle.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/02_error_handle.cu)。关于性能分析，详见[性能分析](#性能分析)，以及代码示例[profiling.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/03_profiling.cu)。

## 线程层级结构

为了方便的定位和访问线程，CUDA提供了线程索引`threadIdx`，这是一个具有三个分量的向量，可以使用一维、二维或三维的线程索引来标识线程。

线程的索引和ID对应关系如下：

- 一维block：索引与ID相等；
- 大小为`(Dx, Dy)`的二维block：索引`(x, y)`对应ID为`(x + y * Dx)`；
- 大小为`(Dx, Dy, Dz)`的三维block：索引`(x, y, z)`对应ID为`(x + y * Dx + z * Dx * Dy)`。

下面是一个矩阵加法的例子。

```c++
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    // 核函数调用，该核函数包括N * N * 1个线程
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

> 对于当前的GPU，每个线程块（block）最多可以容纳1024个线程。但每个核函数可以被多个形状相同的线程块执行，故总线程数为线程块数量 * 每块线程数。

线程块由一维、二维或三维的线程块网格（grid）组织起来，网格中线程块的数量通常由被处理的数据决定，通常这个数量是超过物理处理单元数量的。

grid和block的数量可以由`<<<...>>>`来指定，其中的数据类型可以是`int`或`dim3`。

每个block可以由一维、二维或三维的`blockIdx`唯一标识，block的维度可以通过`blockDim`在核函数内获取。

![](https://github.com/Deleter-D/Images/assets/56388518/42de655c-5f10-41d2-9bc8-e1a40f3fd95f)

扩展一下上面的矩阵加法的例子。

```c++
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}
int main()
{
    ...
    // 每个block有16 * 16共256个线程
    dim3 threadsPerBlock(16, 16);
    // grid大小是由矩阵的大小决定的，来确保每个矩阵元素都有线程处理
    dim3 numBlocks(N ∕ threadsPerBlock.x, N ∕ threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

上面的例子是基于每个grid的线程数在每个维度上都能被每个block的线程数整除，但实际中可能不是这样的。

同一个block中的线程可以通过共享内存（shared memory）来协作，并可以通过内部函数`__syncthreads()`来同步线程以协调访存，这些后面会再提到。

> 详细来说，CUDA可以组织最多三个维度的grid和block，如果不指定维度，则默认都是一维的。
>
> 有一些内建变量可以用来索引线程，这些变量只在核函数内有效，定义在`device_launch_parameters.h`头文件中。
>
> 其中，`blockIdx`和`threadIdx`是类型为`uint3`的结构体，分别都有`x, y, z`三个成员，该结构体源码长下面这样。
>
> ```c++
> struct __device_builtin__ uint3
> {
> unsigned int x, y, z;
> };
> ```
>
> 而`gridDim`和`blockDim`是类型为`dim3`的结构体，同样有`x, y, z`三个成员，源码是下面这样。
>
> ```c++
> struct __device_builtin__ dim3
> {
> unsigned int x, y, z;
> 	// 还有其他部分...
> };
> ```
>
> 对于上述几个变量，他们有下列范围关系：
>
> - `blockIdx.x`范围为`[0, gridDim.x-1]`；
> - `blockIdx.y`范围为`[0, gridDim.y-1]`；
> - `blockIdx.z`范围为`[0, gridDim.z-1]`；
> - `threadIdx.x`范围为`[0, blockDim.x-1]`;
> - `threadIdx.y`范围为`[0, blockDim.y-1]`;
> - `threadIdx.z`范围为`[0, blockDim.z-1]`;
>
> 定义多维grid和block可以使用C++构造函数的形式。
>
> ```c++
> dim3 grid_size(Gx, Gy, Gz);
> dim3 block_size(Bx, By, Bz);
> ```
>
> grid和block是有大小限制的：
>
> - `gridDim.x`最大值为$2^{31}-1$；
> - `gridDim.y`最大值为$2^{16}-1$；
> - `gridDim.z`最大值为$2^{16}-1$；
> - `blockDim.x`最大值为1024；
> - `blockDim.y`最大值为1024；
> - `blockDim.z`最大值为64；
>
> 但block有一个额外限制，总的block数最大为1024，即满足上述限制的同时要满足`blockDim.x * blockDim.y * blockDim.z <= 1024`。
>
> 具体例子参考[threads_management.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/04_threads_management.cu)

### 线程块集群

线程块集群（Thread Block Cluster）是一个可选的层级。在线程块中，线程会被协调安排在流式多处理器（streaming multiprocessor）中。类似地，一个线程块集群中的线程会被协调安排在GPU处理簇（GPU Processing Cluster, GPC）中。

与block类似，簇也有一维、二维或三维的形式。簇中的block数量是可以由用户定义的，但CUDA支持的可移植的簇中最多包含8个block。对于不能支持最大簇的GPU设备或MIG配置中，会相应减小这个最大数量。在特定体系结构下，可以用`cudaOccupancyMaxPotentialClusterSize`接口来查询簇支持的最大block数量。

> MIG是英伟达提供的多实例GPU框架，能够独立划分GPU资源，使得GPU在运行不同任务的时候不会争抢资源。

![](https://github.com/Deleter-D/Images/assets/56388518/48e6be32-5b22-4ec0-884d-00d767fa2893)

可以使用编译时核函数属性`__cluster_dims__(X, Y, Z)`或者使用CUDA核函数的启动API`cudaLanunchKernelEx`来启用线程块集群。若核函数使用编译时属性确定簇大小，则核函数启动时就无法改变簇大小。下面是使用编译时核函数属性使用线程块集群的例子。

```c++
// 编译时确定线程块集群尺寸为2 * 1 * 1
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output) {}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // grid的维度不受线程块集群启动方式影响
    // grid的维度必须是线程块集群尺寸的倍数
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

线程块集群大小也可以在运行时设置，并且使用CUDA核函数的启动API`cudaLaunchKernelEx`来启动内核。

```c++
__global__ void cluster_kernel(float *input, float* output) {}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    {
        cudaLaunchConfig_t config = {0};
        // grid的维度不受线程块集群启动方式影响
        // grid的维度必须是线程块集群尺寸的倍数
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // 线程块集群的X维度大小
        attribute[0].val.clusterDim.y = 1; // 线程块集群的Y维度大小
        attribute[0].val.clusterDim.z = 1; // 线程块集群的Z维度大小
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

在计算能力9.0的GPU中，线程块集群中所有的线程块都保证在单个GPC上调度，并且允许簇中的线程块使用`Cluster Group`中的API`cluster.sync()`来执行硬件支持的同步操作。同时`Cluster Group`还提供成员函数`num_threads()`和`num_blocks()`来获取簇中的线程数和线程块数。`dim_threads()`和`dim_blocks()`获取线程和线程块的维度排列方式。

对于同一线程块集群中的线程块来说，共享内存是分布式的，可以互相访问彼此的共享内存。

## 内存层级结构

![](https://github.com/Deleter-D/Images/assets/56388518/c3212b1c-8279-44eb-9420-df8763211819)

CUDA线程可以从不同的内存空间中访问数据。

- 每个线程拥有私有的本地内存（Local Memory）；
- 每个线程块拥有所有块内线程可见的共享内存（Shared Memory），这些块内线程与线程块有着相同的生命周期；
- 每个线程块集群中的线程块可以在彼此的共享内存中进行读、写和原子操作；
- 所有线程共享全局内存（Global Memory）。

此外还有两片可以被所有线程访问的读内存空间，常量内存（Constant Memory）和纹理内存（Texture Memory）空间。全局、常量、纹理内存针对不同的内存使用进行了优化，纹理内存还为一些特定的数据格式提供不同的访存模式以及数据过滤。这三片内存在同一个程序的核函数启动期间是持久的。

> CUDA的内存管理与标准C语言的内存管理非常类似。
>
> | 标准C函数 | CUDA C函数   | 功能       |
> | --------- | ------------ | ---------- |
> | `malloc`  | `cudaMalloc` | 内存分配   |
> | `memcpy`  | `cudaMemcpy` | 内存般移   |
> | `memset`  | `cudaMemset` | 内存初始化 |
> | `free`    | `cudaFree`   | 内存释放   |
>
> 函数签名的对比如下。
>
> | 标准C函数                                                    | CUDA C函数                                                   |
> | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | `extern void *malloc(size_t __size)`                         | `__host__ __device__ cudaError_t cudaMalloc(void **devPtr, size_t size)` |
> | `extern void *memcpy(void *__restrict __dest, const void *__restrict __src, size_t __n)` | `__host__ cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)` |
> | `extern void *memset(void *__s, int __c, size_t __n)`        | `__host__ cudaError_t cudaMemset(void *devPtr, int value, size_t count)` |
> | `extern void free(void *__ptr)`                              | `__host__ __device__ cudaError_t cudaFree(void *devPtr)`     |
>
> 上表中`cudaMemcpy`接口中的最后一个参数`cudaMemcpyKind`是一个枚举类，有五个成员。
>
> - `cudaMemcpyHostToHost`：主机 -> 主机；
> - `cudaMemcpyHostToDevice`：主机 -> 设备；
> - `cudaMemcpyDeviceToHost`：设备 -> 主机；
> - `cudaMemcpyDeviceToDevice`：设备 -> 设备；
> - `cudaMemcpyDefault`：默认方式，仅允许在支持统一虚拟寻址的系统中使用。
>
> 详细代码见[memory_management.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/05_memory_management.cu)。

## 异构编程

CUDA 编程模型假设主机和设备维护自己独立的内存空间，分别称为主机内存（Host Memory）和设备内存（Device Memory），程序通过CUDA运行时来管理设备内存，包括内存申请、释放以及主机和设备之间的数据搬移。

CUDA还提供了统一内存管理，贯通了主机内存和设备内存，使得系统中所有CPU和GPU都能够通过一个单一的、连续的内存视图来管理内存。统一内存使得设备端可以超额订阅内存，且不必再显式地将数据从主机端镜像到设备端，从而简化程序的移植。

## 异步SIMT编程模型

在CUDA中，线程是运算或内存操作的最低层级抽象。异构编程模型定义了CUDA线程的异步操作，为CUDA线程之间的同步定义了异步屏障（Asynchronous Barrier）行为。该模型还解释并定义了`cuda::memcpy_async`如何在GPU计算的同时从全局内存中异步搬移数据。

### 异步操作

异步操作的定义是，由CUDA线程发起，并且看起来像由另一个线程异步执行的操作。在良好的程序中，一个或多个CUDA线程与异步操作同步。发起异步操作的CUDA线程不需要位于同步线程中。这样的异步线程（类似线程）始终与启动异步操作的CUDA线程相关联。

![](https://github.com/Deleter-D/Images/assets/56388518/e85f06cf-a5d4-4cb8-9d8e-3d47d4b84127)

异步操作使用同步对象来同步操作的完成，同步对象可以由用户显式管理（`cuda::memcpy_async`）或由库隐式管理（`cooperative_groups::memcpy_async`）。同步对象可以是`cuda::barrier`或`cuda::pipeline`，后续会详细介绍。这些同步对象可以在不同的线程域中使用，详见下表。

| 线程域                                    | 描述                                                     |
| ----------------------------------------- | -------------------------------------------------------- |
| `cuda::thread_scope::thread_scope_thread` | 仅在发起异步操作的线程内同步。                           |
| `cuda::thread_scope::thread_scope_block`  | 在发起异步操作的线程所属的block内同步所有或任意线程。    |
| `cuda::thread_scope::thread_scope_device` | 在发起异步操作的线程所属的GPU设备上同步所有或任意线程。  |
| `cuda::thread_scope::thread_scope_system` | 在发起异步操作的线程所属的整个系统中同步所有或任意线程。 |

## 计算能力

设备的计算能力由版本号表示，也称为“SM 版本”。 该版本号标识 GPU 硬件支持的功能，并由应用程序在运行时使用来确定当前 GPU 上可用的硬件功能和指令。计算能力包括主版本号X和次版本号Y，用X.Y表示。

- Hopper架构：9
- Ampere架构：8
- Volta架构：7
  - Turing架构：7.5
- Pascal架构：6
- Maxwell架构：5
- Kepler架构：3

若想知道自己的GPU对应的计算能力可以在英伟达官网给出的列表中查询[GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)。

支持的虚拟架构计算能力见[Virtual Architecture Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)，实际架构计算能力见[GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)。

## 性能分析

这里对性能分析手段只需有一个初步了解，不讨论过多细节。CUDA性能分析大致有6种方法：

- CPU计时
- 事件计时
- NVVP（Nidia Visual Profiler）
- `nvprof`
- Nsight Systems（`nsys`）
- Nsight Compute（`ncu`）

### CPU计时

是最简单粗暴的方式，统计精度不高，不推荐使用。

### 事件计时

通过CUDA提供的事件API来统计时间，精度比CPU计时高，但只能统计到某代码段的运行时间，没有其他性能信息。

### NVVP和`nvprof`

NVVP是带有图形界面的性能分析工具，`nvprof`是针对命令行的。这两个工具是有局限性的，当文件较大的时候，性能分析的速度会很慢。

这两个工具是上一代的CUDA性能分析工具，工具定位也不是很清楚。对于计算能力7.5及以上的设备已经不支持使用`nvprof`来进行性能分析了。

### Nsight Systems和Nsight Compute

这两个是新一代的CUDA性能分析工具，Nsight Systems是系统层面的分析工具，不仅会分析GPU的使用情况，还会给出CPU使用情况，以及CPU和GPU之间的交互情况。而Nsight Compute则用于分析核函数。

Nvidia官方推荐的使用顺序是：

- 先使用Nsight Systems从系统层面进行分析，优化不必要的同步和数据传输等操作。
- 如果是计算程序，则使用Nsight Compute进行分析来优化核函数性能；
  如果是图形程序，则使用Nsight Graphics进行分析，由于重点在于计算算子的开发，所以不对这一工具作过多介绍。
- 优化过核函数后，再使用Nsight Systems重新进行系统层面的分析，继续优化。

重复以上过程，直到达到一个比较满意的性能。

## 组织并行线程

在此之前，我们已经了解了CUDA的基本编程模型，接下来用一系列实例来加深理解。

### 二维grid二维block

使用二维grid和二维block可以最好的理解矩阵加法映射到CUDA线程的过程，如果将grid中的每个block平铺开，再将每个block中的线程也平铺开，那么正好每个线程就对应矩阵的一个元素。

![](https://github.com/Deleter-D/Images/assets/56388518/58d83140-598a-4aa4-a7cc-0cd35fdfb6b5)

这种情况下，可以先分别定位线程在x维度和y维度上的索引：

- x维度索引`ix = blockDim.x * blockIdx.x + threadIdx.x`；
- y维度索引`iy = blockDim.y * blockIdx.y + threadIdx.y`；

然后再得到全局索引`idx = iy * nx + ix`，当然这个的前提是矩阵为行优先存储。

详细代码见[example_sum_matrix_2D-grid_2D-block.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/06_example_sum_matrix_2D-grid_2D-block.cu)。

### 一维grid一维block

使用一维grid一维block就意味着每个线程必须处理矩阵的一列数据，如图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/454a29b9-5130-4d16-b4aa-9ed0b7004bf6)

此时，x维度上的线程索引仍然为`ix = blockDim.x * blockIdx.x + threadIdx.x`。但y维度的线程只有1个，所以需要在线程内部遍历该线程待处理列向量中的每一个元素。元素在线程内部的索引为`idx = iy * nx + ix`，其中`iy`的范围是`[0, ny)`。

详细代码见[example_sum_matrix_1D-grid_1D-block.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/07_example_sum_matrix_1D-grid_1D-block.cu)。

### 二维grid一维block

这种情况下，每个线程都只处理矩阵的一个元素，可以看作是[二维grid二维block](#二维grid二维block)的一种特殊情况，其中block的y维度为1，映射关系如图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/65280438-b3a6-459f-979e-627144af3459)

还是从x和y两个维度来分别索引线程：

- x维度索引`ix = blockDim.x * blockIdx.x + threadIdx.x`；
- y维度索引`iy = blockIdx.y`；

然后得到线程的全局索引为`idx = iy * nx + ix`。

详细代码见[example_sum_matrix_2D-grid_1D-block.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/08_example_sum_matrix_2D-grid_1D-block.cu)。

## 设备管理

### 运行时API查询GPU信息

可以通过运行时API`cudaGetDeviceProperties()`来获取关于GPU的所有信息，通过给该API传入一个`cudaDeviceProp`和设备id来获取带有设备信息的结构体。结构体的成员详见[cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp)。

常用的一些属性请参考[device_management.cu](https://github.com/Deleter-D/CUDA/blob/master/01_programming_model/09_device_management.cu)，代码中还给出了在多GPU系统中确定最优GPU的方法。

### 使用nvidia-smi查询GPU信息

`nvidia-smi`

选项`-L`可以列出系统中安装了多少个GPU，以及每个GPU的设备ID。

```shell
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 4070 (UUID: GPU-2792043a-40d8-faf2-cf4e-e94d68836d2f)
```

选项`-q -i ${DeviceID}`可以获取指定设备的详细信息。

```shell
$ nvidia-smi -q -i 0
```

可以通过下列参数精简显示的信息：

- MEMORY
- UTILIZATION
- ECC
- TEMPERATURE
- POWER
- CLOCK
- COMPUTE
- PIDS
- PERFORMANCE
- SUPPORTED_CLOCKS
- PAGE_RETIREMENT
- ACCOUNTING

```shell
$ nvidia-smi -q -i 0 -d MEMORY
```

```shell
$ nvidia-smi -q -i 0 -d UTILIZATION
```

### 通过环境变量设置设备

可以通过环境变量`CUDA_VISIBLE_DEVICES`来指定设备。

- 若`CUDA_VISIBLE_DEVICES = 2`，则设备2将在CUDA程序中以设备0的身份出现；
- 若`CUDA_VISIBLE_DEVICES = 2,3`，则设备2，3将在CUDA程序中以设备0，1的身份出现。
