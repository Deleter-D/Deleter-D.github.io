---
title: CUDA编程——流和并发
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 4919
date: 2024-02-20 16:36:28
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## 异步并发执行

CUDA将以下几个操作暴露为可以互相并发执行的操作：

- 主极端的运算；
- 设备端的运算；
- 从主机端到设备端的数据传输；
- 从设备端到主机端的数据传输；
- 给定设备内的数据传输；
- 设备之间的数据传输。

这些操作之间实现的并发级别将取决于设备的特性集和计算能力，如下所述。

### 主机和设备之间的并发执行

主机和设备的并发执行是通过在设备完成所请求的任务之前，将控制权交还给主机线程完成的，这一过程由异步库函数实现。使用异步调用，多个设备可以同时排队，以便在设备资源满足时，由CUDA驱动程序执行。这减轻了主机线程管理设备的责任，使其有更多的精力去执行其他操作。

以下设备操作对于主机是异步的：

- 核函数启动；
- 在单个设备内的内存拷贝；
- 从主机到设备的64KB或更小的内存拷贝；
- 由带有`Async`后缀的函数发起的内存拷贝；
- 内存集合（Memory Set）函数调用。

可以将环境变量`CUDA_LAUNCH_BLOCKING`设为1来全局禁用所有CUDA程序的核函数异步启动。该功能仅用于调试，不应该用于生产环境。

若通过分析软件（Nsight，Visual Profiler等）收集硬件计数器，则核函数启动时同步的，除非启用了并发核函数分析。若异步内存拷贝没有涉及锁页内存，则该拷贝也可能是同步的。

### 并发核函数执行

部分计算能力2.x以及更高的设备能够并发执行多个核函数。程序可以通过设备属性`concurrentKernels`来检查这种能力，该属性为1则表示支持该功能。

核函数的最大并发量取决于设备的计算能力，详见[官方文档表格](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)。

来自一个CUDA上下文中的核函数无法与另一个CUDA上下文中的核函数并发执行。GPU可能以时间划分来为每个上下文提供前向过程。若用户想在SM上同时执行多个过程的核函数，则必须启动MPS。

若核函数使用了很多纹理内存或大量的本地内存，则与其他核函数并发执行的可能性会更小。

### 数据传输与核函数的交叠

一些设备可以利用核函数并发地执行（与设备之间的）异步内存拷贝。可以通过设备属性`asyncEngineCount`来检查这种能力，若该值大于0则表明设备支持该功能。如果内存拷贝涉及到了主机内存，则必须是锁页内存。

通过核函数同时执行设备内部的内存拷贝（需要设备支持设备属性`concurrentKernels`）和与设备之间的内存拷贝（需要设备支持设备属性`asyncEngineCount`）也是可能的。设备内部的内存拷贝通过使用目的地址与源地址处于同一设备的标准内存拷贝函数发起。

### 并发数据传输

部分计算能力2.x以及更高的设备可以将与设备之间的内存拷贝操作交叠。可以通过设备属性`asyncEngineCount`来检查这种能力，若该值为2则表明设备支持该功能。为了实现这种交叠，任何涉及到的主机内存必须是锁页内存。

## 流和事件

CUDA流是一系列异步的CUDA操作，这些操作按照主机代码确定的顺序在设备上执行。流可以封装这些操作，保持操作的顺序，允许操作在流中排队，并使其在先前的所有操作之后执行，并且可以查询排队操作的状态。流中操作的执行相对于主机总是异步的。在同一个CUDA流中的操作有严格的执行顺序，而不同流中的操作在执行顺序上不受限制。

CUDA的API函数一般分为同步和异步。具有同步行为的函数会阻塞主机端线程，直到它们完成。具有异步行为的函数被调用后，会立即将控制权归还给主机。

### CUDA流

##### 创建和销毁流

通过创建一个流对象并将其指定为一系列核函数启动与`host <-> device`间内存拷贝的参数来定义流。下列代码创建了两个流对象，并在锁页主机内存中分配一个`float`型的数组`hostPtr`。

```cpp
cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);
```

下面的代码通过三个操作定义了一系列流，三个操作分别是`host -> device`内存拷贝、核函数启动、`device -> host`内存拷贝。

```cpp
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}
```

上面的例子中，每个流将输入数组`HostPtr`的一部分拷贝到设备内存中的数组`InputDevPtr`，通过调用核函数`MyKernel()`来处理这些数据，最后将结果`outputDevPtr`拷贝回`HostPtr`的同一位置。注意：`hostPtr`必须指向锁页内存才会发生交叠行为。

通过调用`cudaStreamDestroy()`来释放流。

```cpp
for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);
```

如果在调用`cudaStreamDestroy()`时设备仍在流中工作，则该函数将立即返回，并且一旦设备完成流中的所有工作，与流关联的资源将被自动释放。

> CUDA提供两个API来检查流中的操作是否都已完成：
>
> - `cudaStreamSynchronize()`：强制阻塞主机，直到所给流中的操作全部完成；
> - `cudaStreamQuery()`：检查流中的所有操作是否完成，在完成之前不会阻塞主机。所有操作都完成时会返回`cudaSuccess`，当还有操作仍在执行或等待执行时返回`cudaErrorNotReady`。

##### 默认流

当启动核函数与执行`host <-> device`间内存拷贝时未指定流参数，或将流参数指定为0，命令将提交到默认流。他们是顺序执行的。

使用编译选项`--default-stream per-thread`编译的代码，或在引入CUDA头文件之间定义了宏`CUDA_API_PER_THREAD_DEFAULT_STREAM`的代码，默认流是常规流，每个主机线程均有自己的默认流。

对于使用了编译选项`--default-stream legacy`编译的代码，默认流是称为空流（NULL Stream）的特殊流，每个设备都有一个用于所有主机线程的空流。空流是特殊的，因为它会导致[隐式同步](#隐式同步)。

对于未指定编译选项`--default-stream`的代码，默认值为`--default-stream legacy`。

### 流调度

从逻辑上看，所有流都可以同时执行，但映射到硬件上时并不总是这样。

#### 虚假的依赖关系

虽然GPU支持多个`grid`同时执行，但所有流最终是被多路复用到单一的硬件工作队列中的。当选择一个`grid`执行时，在队列前面的任务由CUDA运行时调度。运行时会检查任务的依赖关系，若仍有任务在执行，则将等待该任务依赖的任务执行完成。当所有依赖关系都执行结束后，新任务才会被调度到可用的SM上。

上述的这种单一流水线可能会导致虚假的依赖关系。图中的A、P与X本来没有依赖关系，但由于队列是单一的，故P只能等待A-B均执行完成才会和C一起被并发调度，X同理。

![](https://github.com/Deleter-D/Images/assets/56388518/ea7d2a6b-8ceb-4f99-8abb-bf705846ee7d)

#### Hyper-Q技术

Hyper-Q使用多个硬件工作队列，从而减少虚假的依赖关系。该技术通过在主机和设备之间维持多个硬件管理上的连接，允许多个CPU线程或进程在单一GPU上同时启动工作。

![](https://github.com/Deleter-D/Images/assets/56388518/baccc996-b6f1-44ea-a7c6-98d07d642c58)

这种技术可以实现全流级并发，并具有最小的虚假流间依赖关系。

### 流的优先级

流的相对优先级可以在创建时使用`cudaStreamCreateWithPriority()`来指定。可以使用`cudaDeviceGetStreamPriorityRange()`函数获取允许的优先级范围，按 [最高优先级，最低优先级] 排序。在运行时，较高优先级流中的挂起工作优先于较低优先级流中的挂起工作。

下面的代码展示了如何获取当前设备允许的优先级范围，并创建具有最高和最低优先级的流。

```cpp
// 获取当前设备允许的优先级范围
int priority_high, priority_low;
cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
// 创建具有最高和最低优先级的流
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
```

> 流优先级不会影响数据传输操作，只对计算核函数有影响。
>
> 如果指定的优先级超出了设备定义的范围，它会被自动限制为定义范围内的最低值或最高值。

### CUDA事件

CUDA运行时提供了一种方法来密切监视设备的进度，并通过让应用程序异步记录程序中任意点的事件并查询这些事件何时完成来执行准确的计时。当事件之前的所有任务（或者给定流中的所有命令）都完成时，事件就完成了。在所有流中的所有前面的任务和命令完成之后，流0中的事件才会完成。

#### 创建和销毁事件

下面的代码创建两个事件。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

用如下方式销毁。

```cpp
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

> 对于事件，CUDA也提供了检查先前操作是否完成的API：
>
> - `cudaEventSynchronize()`：强制阻塞主机，直到该事件所在流中先前的操作执行结束，与`cudaStreamSynchronize()`类似；
> - `cudaEventQuery()`：不阻塞主机，与`cudaStreamQuery()`类似。

#### 记录事件和计算运行时间

下面的代码展示了如何利用事件统计运行时间。

```cpp
cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size, size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size, size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
```

> 事件的启动和停止不必在同一个流中。
>
> 注意：若在非默认流中记录启动事件或停止事件，返回的时间可能比预期的要大。因为`cudaEventRecord()`函数是异步的，并且不能保证计算的延迟正好处于两个事件之间。

### 流同步

在非默认流中，所有操作对于主机线程都是非阻塞的，因此会遇到需要在同一个流中令主机和运算操作同步的情况。

从主机角度来讲，CUDA操作可以分为两大类：

- 内存操作；
- 核函数启动。

对于主机来说，核函数启动总是异步的。很多内存操作本质上是同步的，但CUDA也提供了异步版本。

CUDA的流可以分为两种：

- 异步流（非默认流）；
- 同步流（默认流）。

在主机上，非默认流是异步流，其上所有操作都不阻塞主机执行。而被隐式声明的默认流是同步流，大多数添加到默认流上的操作都会导致主机在先前所有的操作上阻塞。

非默认流可以进一步分为两种：

- 阻塞流；
- 非阻塞流。

虽然非默认流对于主机是非阻塞的，但非默认流中的操作可以被默认流中的操作所阻塞。若一个非默认流是阻塞流，则默认流可以阻塞该非默认流中的操作。若一个非默认流是非阻塞流，则它不会阻塞默认流中的操作。

#### 阻塞流和非阻塞流

使用`cudaStreamCreate()`创建的流是阻塞流，即这些流中的操作可以被阻塞，一直等到默认流中先前的操作执行结束。默认流是隐式流，在相同的CUDA上下文中它和其他所有的阻塞流同步。一般情况下，当操作被发布到默认流中，在该操作被执行之前，CUDA上下文会等待所有先前的操作发布到所有的阻塞流中。此外，任何发布到阻塞流中的操作会被挂起等待，指导默认流中先前的操作执行结束才开始执行。

CUDA运行时提供了一个函数：

```cpp
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
```

参数`flags`决定了所创建流的行为，可选如下两个值：

- `cudaStreamDefault`；
- `cudaStreamNonBlocking`。

指定为`cudaStreamNonBlocking`使得默认流对于非默认流的阻塞行为失效。

#### 隐式同步

若主机线程在来自不同流的两个命令之间发出以下任何一个操作，它们均不能同时执行：

- 锁页主机内存分配；
- 设备内存分配；
- 设备内存初始化；
- 同一设备内存地址之间的内存拷贝；
- 任何提交至空流的CUDA命令；
- L1/共享内存配置之间的切换（见[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x)）。

一些操作需要依赖关系检查，这些操作包括与被检查的启动项相同流中的任何其他命令，以及该流上对`cudaStreamQuery()`的任何调用。故程序应该遵循以下原则，以提高核函数的并发潜力：

- 所有独立操作应该在有依赖关系的操作之前提交；
- 任何形式的同步操作都应该尽可能的延迟。

#### 显式同步

以下多种方法可以显式同步流。

- `cudaDeviceSynchronize()`：会等待此调用前的所有主机线程中的所有流中的所有命令全部完成；
- `cudaStreamSynchronize()`：接受一个流作为参数，等待给定流中的此调用前的所有命令全部完成。可以用于使主机与特定流同步，从而允许其他流继续在设备上执行；
- `cudaStreamWaitEvent()`：接受流和事件作为参数，使该调用之后添加到给定流的所有命令延迟执行，直到给定事件完成。该函数允许跨流同步；
- `cudaStreamQuery()`：返回流中此调用前的所有命令的完成情况。

#### 可配置事件

与流类似，CUDA也提供了API来创建不同行为的事件：

```cpp
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
```

参数`flags`的可选项有4个：

- `cudaEventDefault`；
- `cudaEventBlockingSync`；
- `cudaEventDisableTiming`；
- `cudaEventInterprocess`。

指定为`cudaEventBlockingSync`的事件在同步时会阻塞调用的线程。`cudaEventSynchronize()`函数的默认操作是围绕事件进行的，使用CPU周期不断检查事件的状态。而指定为`cudaEventBlockingSync`后，会将这种轮询交给另一个线程，而调用线程本身继续执行，直到事件依赖关系满足才通知调用线程。这样可以减少CPU周期的浪费，但也会使得事件满足依赖关系与激活调用线程之间的延迟被拉长。

指定为`cudaEventDisableTiming`表明事件只能用来同步，节省了记录时间戳带来的开销。

指定为`cudaEventInterprocess`表明时间可能被用作进程间事件。

## 并发核函数执行

### 非默认流中的并发核函数

现在已经几乎不存在不支持Hyper-Q的GPU设备了。想要实现核函数的并发执行，必须要令核函数在非默认流中执行。使用类似如下代码使核函数在不同的非默认流中并发执行。

```cpp
for (int i = 0; i < stream_count; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_3<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_4<<<grid, block, 0, streams[i]>>>(d_data);
}
```

> 详细代码参考[hyper-Q_depth.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/01_hyper-Q_depth.cu)。

通过`NVIDIA Nsight System`分析核函数在流中的执行情况，可以借助可视化流水线来观察核函数在各个流中的执行情况，分析结果如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/169eb417-183d-483f-bee3-d5588c6b8e1a)

可以观察到核函数在`stream 14`，`stream 15`，`stream 16`中是并发执行的。

> 关于这里有一些待解决和讨论的问题，为什么`stream 13`中的核函数没有和其他三个流中的核函数并发执行。我们来尝试一下每个流执行一个、两个和三个核函数，分别得到了如下结果。
>
> ![](https://github.com/Deleter-D/Images/assets/56388518/03e3a790-e45f-4bd0-9f2e-ec729631ca99)
>
> ![](https://github.com/Deleter-D/Images/assets/56388518/18f080d8-528b-44e3-abc6-ba5ce23d8bd0)
>
> ![](https://github.com/Deleter-D/Images/assets/56388518/9263efa6-cd30-4809-b329-a33b0614fef1)
>
> 可以观察到，每个流只有一个核函数时，并发执行情况是符合预期的。但每个流中执行的核函数超过一个，就会先在某个流中全部执行一遍，然后在其他流中并发执行。在英伟达开发者论坛中询问后，官方人员给出了解释，这是由于核函数初始化时机导致的，这里放上帖子的[链接](https://forums.developer.nvidia.com/t/kernels-executing-concurrently-in-different-streams-do-not-behave-as-expected/276389)。
>
> 从CUDA 11.7开始，引入了一种懒加载机制，并在CUDA 11.8中进行了重大更新，相见官方文档[Lazy Loading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading)。懒加载会使得核函数在真正要执行之前才进行初始化，而不是先全部初始化后再根据安排来执行。我们可以通过环境变量`CUDA_MODULE_LOADING`来控制是否开启懒加载，开启则设为`LAZY`，不开启则设为`EAGER`。
>
> 我们将环境变量设置为`EAGER`便可以得到所有流同时启动核函数的结果。
>
> ![](https://github.com/Deleter-D/Images/assets/56388518/82ab2a82-9340-4056-8efc-0e817910f0e1)

### 虚假依赖关系的并发核函数

接下来我们模拟一下不支持Hyper-Q技术的设备。可以通过环境变量`CUDA_DEVICE_MAX_CONNECTIONS`来控制队列个数，这里设为1来模拟只有一个硬件队列的情况。

```cpp
setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
```

在此基础上使用`nsys`分析核函数执行情况。

![](https://github.com/Deleter-D/Images/assets/56388518/dd30ae2d-d4c9-4e5f-a81a-286ccff6cb8a)

可以观察到，如同[虚假的依赖关系](#虚假的依赖关系)中介绍的那样，核函数的并发执行只发生在了流的边缘。在这种情况下，可以通过广度优先的调用方式来缓解。

```cpp
for (int i = 0; i < stream_count; i++)
    kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
for (int i = 0; i < stream_count; i++)
    kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
for (int i = 0; i < stream_count; i++)
    kernel_3<<<grid, block, 0, streams[i]>>>(d_data);
for (int i = 0; i < stream_count; i++)
    kernel_4<<<grid, block, 0, streams[i]>>>(d_data);
```

> 详细代码参考[hyper-Q_breadth.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/02_hyper-Q_breadth.cu)。

深度优先和广度优先的调用方式的区别可以通过下图来理解。

![](https://github.com/Deleter-D/Images/assets/56388518/52e64391-7d1b-454e-9847-974df653fdf2)

这样就可以在只有一个硬件队列的情况下，实现核函数的并发执行。不过鉴于现代的GPU设备基本都支持Hyper-Q，所以此处简单了解即可。通过`nsys`分析来佐证上述的观点。

![](https://github.com/Deleter-D/Images/assets/56388518/b03d99e9-397d-402d-a5e3-ee222f5f41d1)

### 使用OpenMP的调度操作

借助OpenMP开启多线程，每个线程来管理一个流。

```cpp
omp_set_num_threads(stream_count);
#pragma omp parallel
{
    int i = omp_get_thread_num();
    kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_3<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_4<<<grid, block, 0, streams[i]>>>(d_data);
}
```

> 详细代码参考[hyper-Q_OpenMP.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/03_hyper-Q_OpenMP.cu)。

注意，主机代码中若包含OpenMP的内容，则在编译CUDA程序时需要添加如下几个编译器选项。

```
-Xcompiler -fopenmp -lgomp
```

### 默认流的阻塞行为

将[非默认流中的并发核函数](#非默认流中的并发核函数)中提到的`kernel_3`放在默认流中调用，以此来理解默认流的阻塞行为。

```cpp
for (int i = 0; i < stream_count; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_3<<<grid, block>>>(d_data);
    kernel_4<<<grid, block, 0, streams[i]>>>(d_data);
}
```

核函数执行情况如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/8fd6a798-48e9-4509-a772-23309ad473b2)

可以观察到，由于`kernel_3`在默认流中启动，所以在非默认流上的所有之后的操作都会被阻塞，直到默认流中的操作完成。

### 创建流间依赖关系

在复杂的应用程序中，引入流间依赖关系是非常有用的，它可以在一个流中阻塞操作，直到另一个流中的指定操作完成。事件可以用来添加流间依赖关系。

通过如下代码来创建同步事件。

```cpp
cudaEvent_t* kernelEvent = (cudaEvent_t*)malloc(stream_count * sizeof(cudaEvent_t));
for (int i = 0; i < stream_count; i++)
{
    ERROR_CHECK(cudaEventCreateWithFlags(&kernelEvent[i], cudaEventDisableTiming));
}
```

通过如下方式调用，令每个流完成时记录不同的事件，然后使最后一个流等待其他所有流完成。

```cpp
for (int i = 0; i < stream_count; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_2<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_3<<<grid, block, 0, streams[i]>>>(d_data);
    kernel_4<<<grid, block, 0, streams[i]>>>(d_data);

    // 每个流完成时记录不同的事件
    ERROR_CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
    // 使最后一个流等待其他所有流
    ERROR_CHECK(cudaStreamWaitEvent(streams[stream_count - 1], kernelEvent[i], 0));
}
```

> 详细代码参考[hyper-Q_dependence.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/04_hyper-Q_dependence.cu)。

观察下图可以看出，最后一个流被其他流阻塞，直到所有流都完成操作后，最后一个流才执行。

![](https://github.com/Deleter-D/Images/assets/56388518/5e746b78-49ee-495a-8fff-2da8bc5f1820)

## 核函数执行与数据传输的交叠

想要令核函数执行与数据传输交叠，不仅需要设备支持，还需要在内存申请和搬移的时候注意。要实现交叠，必须使用异步的内存搬移函数，而异步的内存搬移函数所搬移的内存又需要通过`cudaHostAlloc`来申请。

```cpp
float *h_A, *h_B, *hostRef, *gpuRef;
ERROR_CHECK(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocDefault));
ERROR_CHECK(cudaHostAlloc((void **)&h_B, bytes, cudaHostAllocDefault));
ERROR_CHECK(cudaHostAlloc((void **)&hostRef, bytes, cudaHostAllocDefault));
ERROR_CHECK(cudaHostAlloc((void **)&gpuRef, bytes, cudaHostAllocDefault));
```

> 详细代码参考[multi_add_depth.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/05_multi_add_depth.cu)。

这里以向量加法为例，实现一个向量加法的核函数，并分别使用不交叠和交叠的调用方式来观察其执行情况。

下图是不交叠的调用方式，可以观察到核函数的执行必须要等到`H2D`的内存搬移完成后才能开始，`D2H`的内存搬移也只能等核函数执行完毕后才可以开始。

![](https://github.com/Deleter-D/Images/assets/56388518/c8e60104-8ccc-4a23-90f8-a144d7b59cd6)

但当我们使用多个流，将这些操作分散在各个流中，同时使用异步的内存搬移操作后，效果如下。

![](https://github.com/Deleter-D/Images/assets/56388518/b8ff4b3b-53e9-4f9a-ac91-aa21f8e6ace5)

可以观察到，不同流中的内存搬移操作和核函数执行产生了交叠，大大提高了并发度。

## 流回调

流回调是另一种可以到CUDA流中排列等待的操作。一旦流回调之前所有的流操作全部完成，被流回调指定的主机端函数就会被CUDA运行时调用。流回调允许任意主机端逻辑插入到CUDA流中。

在较老版本的CUDA中通过如下函数来调用。

```cpp
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void *userData, unsigned int flags);
```

- `callback`：传入自定义的回调函数；
- `userData`：传入回调函数的数据；
- `flags`：是保留参数，必须指定为0。

回调函数以如下格式定义。

```cpp
void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data)
{
    printf("callback from stream %d\n", (int*)data);
}
```

上述的流回调方式会在未来被弃用，但目前还没有代替方案，所以依然是有效的。

在较新版本的CUDA中通过如下函数来调用。

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
```

回调函数以如下格式定义。

```cpp
void CUDART_CB my_callback(void* data)
{
    printf("callback from stream %d\n", (int*)data);
}
```

> 详细代码参考[stream_callback.cu](https://github.com/Deleter-D/CUDA/blob/master/05_stream_and_concurrence/06_stream_callback.cu)。同时，关于上述问题，Stack Overflow上有一个[帖子](https://stackoverflow.com/questions/56448390/how-to-recover-from-cuda-errors-when-using-cudalaunchhostfunc-instead-of-cudastr)提到了。
