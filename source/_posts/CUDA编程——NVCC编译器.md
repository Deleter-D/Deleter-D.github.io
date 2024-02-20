---
title: CUDA编程——NVCC编译器
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 50741
date: 2024-02-20 15:59:25
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## 使用NVCC编译

核函数可以使用CUDA指令集架构（PTX）撰写，也可以使用C++撰写，两种方式都需要使用`nvcc`编译器编译。`nvcc`简化了编译PTX或C++代码的过程。

> PTX（Parallel Thread Execution）是CUDA平台为基于GPU通用计算而定义的虚拟机和指令集，类似于针对GPU的汇编代码。
>
> 在编译CUDA C++程序时，nvcc会将设备代码编译为PTX代码，以适应更多的实际架构，再将PTX代码编译为cubin对象进行执行与调用。从CUDA C++编译为PTX代码的过程是与实际GPU设备无关的。

### 编译工作流

#### 离线编译

`nvcc`编译的源文件可以同时包含主机代码和设备代码，它会将两者自动分离。

- 将设备代码编译成汇编形式（PTX代码）或二进制形式（cubin对象）；
- 修改主机代码，通过CUDA运行时函数调用替换`<<<...>>>`，从PTX代码或cubin对象中加载和启动编译后的核函数。

修改后的主机代码要么作为C++代码输出，留给其他工具编译，要么直接作为目标代码输出，令`nvcc`在最后的编译阶段调用主机的编译器。

应用程序可以：

- 链接到编译后的主机代码（最常见的情况）；
- 或忽略修改后的主机代码，并使用CUDA驱动程序API加载和执行PTX代码或cubin对象。

#### 即时编译

即时编译大多情况是针对PTX代码的，这里不过多赘述，详见[Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)。

NVRTC（CUDA C++的运行时编译库）可以在运行时将CUDA C++设备代码编译为PTX代码，是作为`nvcc`编译CUDA C++设备代码的替代方案的。

### 二进制兼容性

二进制代码是特定于体系结构的，可以使用编译器的`-code`选项生产cubin对象。例如使用`-code=sm_80`可以针对计算能力为8.0的设备编译生成二进制代码。

二进制代码对计算能力的小版本是向前兼容的，但小版本无法向后兼容，大版本之间也无法兼容。即为计算能力X.y的设备生成的二进制代码，只能在计算能力X.z的设备上执行，其中$z\ge y$。

### PTX兼容性

某些PTX指令只在具有更高计算能力的设备上支持，包含这些指令的代码必须使用编译器的`-arch`选项指定合适的计算能力。

为某些特定计算能力生成的PTX代码可以被编译为具有更大或相同计算能力设备的二进制代码。但从较早版本的PTX代码编译而来的二进制文件可能不会利用某些硬件新特性，可能导致性能不如使用新版PTX代码编译的二进制文件。

### 应用程序兼容性

要在具有特定计算能力的设备上执行代码，应用程序必须加载与该计算能力兼容的二进制或PTX代码，如果要考虑将代码在未来的体系结构上执行，则尽量选择即时编译。

使用编译器的`-arch`和`-code`选项或`-gencode`选项来控制将哪些PTX代码嵌入到CUDA C++应用程序中。

```shell
nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
```

 嵌入二进制代码兼容5.0和6.0的计算能力设备，PTX和二进制代码兼容7.0的计算能力设备。

主机代码会在运行时加载和执行最合适的代码，以上述编译命令为例：

- 计算能力5.0和5.2的设备会选择5.0的二进制代码；
- 计算能力6.0和6.1的设备会选择6.0的二进制代码；
- 计算能力7.0和7.5的设备会选择7.0的二进制代码；
- 计算能力8.0和8.6的设备会选择编译为二进制代码的PTX代码。

`nvcc`编译器的`-arch`，`-code`和`-gencode`选项有一些简写方式，例如`-arch=sm_70`是`-arch=compute_70 -code=compute70,sm_70`的简写，等价于`-gencode arch=compute_70,code=\"compute_70,sm_70\"`。

> 使用`-arch=compute_XY`来指定一个虚拟架构的计算能力，用`-code=sm_XY`来指定一个实际架构的计算能力。
>
> - 虚拟架构应该尽可能低，以适配更多计算能力的设备；
> - 真实架构应该尽可能高，以充分发挥GPU的实际计算能力。
>
> 例如`nvcc helloworld.cu -o helloworld -arch=compute_61`编译出的可执行文件只能在计算能力大于等于6.1的设备上执行。
>
> 指定实际架构计算能力时必须指定虚拟架构计算能力，并且实际架构计算能力必须不小于虚拟架构计算能力。
>
> 例如`nvcc helloworld.cu -o helloworld -arch=compute_61 -code=sm_60`将是不合法的编译命令。
>
> `nvcc`可以同时指定多个GPU版本进行编译，使得编译出来的可执行文件能够在不同计算能力的设备上执行。使用编译器选项`-gencode arch=compute_XY,code=sm_XY`来指定各个版本的计算能力。注意与`-gencode arch=compute_XY,code=compute_XY`选项的差异，随后会介绍这个差异。
>
> 例如
>
> ```shell
> nvcc helloworld.cu -o helloworld_fat \
>         -gencode arch=compute_50,code=sm_52 \
>         -gencode arch=compute_60,code=sm_61 \
>         -gencode arch=compute_80,code=sm_89
> ```
>
> 上面的编译命令编译出的可执行文件包含3个二进制版本，称为胖二进制文件（fatbinary）。在该例子中，执行该编译命令的CUDA版本必须支持8.9的计算能力。
>
> 上面的例子中，每个`-gencode`分别指定了虚拟架构和实际架构的计算能力，这样可以针对不同的实际架构编译出不同的二进制文件，并将这些二进制文件整合进一个胖二进制文件中。
>
> 而对于选项`-gencode arch=compute_XY,code=compute_XY`，注意`arch`和`code`选项的值均为`compute_XY`即虚拟架构，且`arch`和`code`指定的`compute_XY`必须完全一致。
>
> 回看官方文档中的例子：
>
> ```shell
> nvcc x.cu
>         -gencode arch=compute_50,code=sm_50
>         -gencode arch=compute_60,code=sm_60
>         -gencode arch=compute_70,code=\"compute_70,sm_70\"
> ```
>
> 前两个`-gencode`编译出了针对实际架构计算能力5.0和6.0的二进制代码，最后一个`-gencode`编译出了针对实际架构计算能力7.0的二进制代码，和针对虚拟架构计算能力7.0的PTX代码，当一个计算能力更高的设备（如8.0）来调研该胖二进制文件时，由于没有针对更高计算能力的二进制代码，故会自动选择编译好的PTX代码，采用即时编译的方式为更高计算能力的设备编译二进制代码并调用执行。
>
> 当不指定任何虚拟架构和实际架构的计算能力时，会指定为所使用的CUDA版本的默认值，具体的默认值可以在官方文档中找到。
>
> 另外，关于PTX代码，可以使用`nvcc`的`-ptx`选项，将`.cu`文件编译为一个`.ptx`文件，其中存放的就是PTX代码。
>
> 例如`nvcc hellowrold.cu -ptx`。
>
> 详细代码见[compute_capability.cu](https://github.com/Deleter-D/CUDA/blob/master/00_CUDA_official_documentation/01_compute_capability.cu)与[compute_capability.ptx](https://github.com/Deleter-D/CUDA/blob/master/00_CUDA_official_documentation/01_compute_capability.ptx)。

### C++兼容性

编译器前端按照C++语法规则处理CUDA源文件，主机代码支持完整的C++，而设备代码仅支持C++的一个子集。

### 64位兼容性

64位版本的`nvcc`会以64位模式编译设备代码，64位模式编译的设备代码只支持64位模式编译的主机代码。
