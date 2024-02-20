---
title: CUDA编程——性能分析工具
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 50255
date: 2024-02-20 16:29:56
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## 概述

在之前的CUDA版本中，所附带的性能分析工具主要是`nvvp`（NVIDIA Visual Profiler）和`nvprof`，前者是带有图形界面的分析工具，后者是命令行工具。

在CUDA官方文档中有这样一段描述：

> **Note that Visual Profiler and nvprof will be deprecated in a future CUDA release.** The NVIDIA Volta platform is the last architecture on which these tools are fully supported. It is recommended to use next-generation tools [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) for GPU and CPU sampling and tracing and [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) for GPU kernel profiling.

所以`nvvp`与`nvprof`已经是过去时了，后面的介绍都围绕新的性能分析工具`nsys`（Nsight System）和`ncu`（Nsight Compute）。

`nsys`是系统层面的分析工具，可以分析主机与设备端的信息。`ncu`则是用于分析核函数的工具。两者均有图形界面版本和命令行版本。

## Nsight System

`nsys`是系统级别的分析工具，它可以捕捉CPU和GPU上的各种事件以及两者之间的交互。可以通过流水线观察整个应用程序的性能瓶颈，帮助从整体层面优化程序。`nsys`还支持`NVTX`，可以更加精准的分析流水线。

> `NVTX`即`Nvidia Tools Extension`，是对CUDA代码的一种注释，使其在profiling过程中能够获取到更加细粒度的信息。在CUDA C代码中只需引入`nvToolsExt.h`头文件即可，下面是一个示例。
>
> ```cpp
> #include <cuda_runtime.h>
> #include "nvToolsExt.h"
> 
> __global__ void Kernel() {
>  ...
> }
> 
> void kernelLaunch() {
>  nvtxRangePushA(__FUNCTION__);
>  ...
>  nvtxRangePushA("Kernel");
>  Kernel<<<grid, block>>>();
>  nvtxRangePop();
>  nvtxRangePop();
> }
> ```

打开`nsys`后看到如下界面。

![](https://github.com/Deleter-D/Images/assets/56388518/8fd51e06-221e-45a3-ad18-df8e8018dfde)

左侧是项目管理器，右侧是项目的配置页面。点击`Select target for profiling...`可以选择要进行性能分析的目标机器。可以选择本机，也可以使用`SSH`连接远程机器。这里以本机为例，选择本机后看到如下界面。

![](https://github.com/Deleter-D/Images/assets/56388518/5ee961e8-d4d6-4c7d-b03d-11fc0e7dd894)

这里需要先在`Command line with arguments`中填写执行命令以及所需参数，`Working directory`指的是执行所填命令的目录，如果所填的命令全部为绝对路径，则`Working directory`就不是很重要了。填写后往下拉可以看到另一些配置项。

![](https://github.com/Deleter-D/Images/assets/56388518/f9980d3c-e30b-4202-a4f7-6bff2ded3f90)

上图是默认情况，这些选项就根据需要选择即可。例如在程序中使用了`OpenMP`，若想将`OpenMP`带来的影响也体现在分析结果中，则需勾选`Collect OpenMP trace`，其他的选项同理。

完成配置后就可以点击右侧的`Start`开始profiling，结束后就会看到生成了一个分析报告，如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/51d721fe-c71b-41f9-8995-0e880a7f2c5f)

到此，使用`nsys`的分析工作就结束了。

> 使用`nsys`时可能会遇到如下报错。
>
> ```
> Collection of CPU IP/backtrace samples or context switch data disabled. perf event paranoid level is 4.
> Change the paranoid level to 2 to enable CPU IP/backtrace sample or context switch data collection. Change the paranoid level to 1 to enable CPU kernel sample collection.
> Try
> sudo sh -c 'echo [level] >/proc/sys/kernel/perf_event_paranoid'
> where 'level' equals 1 or 2.
> ```
>
> 根据提示执行命令：
>
> ```sh
> sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
> ```
>
> 这里建议直接将`level`修改为1，如果`level`为2可能还会存在一些警告。

## Nsight Compute

`ncu`是核函数级别的分析工具，它可以捕捉核函数执行过程中的各种数据。能够从显存使用、SM占用、`warp`状态等角度来分析核函数的瓶颈所在。

打开`ncu`后看到如下界面。

![](https://github.com/Deleter-D/Images/assets/56388518/f4b6dfc8-4d27-413b-844e-a96261767732)

左侧是项目管理器，双击项目即可开始配置。

![](https://github.com/Deleter-D/Images/assets/56388518/644fdf63-cd16-48c9-b48b-c75f3b2dda1e)

这里的必填项是上方带黄色感叹号的`Application Executable`和下方的`Output File`。`Application Executable`和`Working Directory`与`nsys`的同理，`Output File`是指输出的性能分析结果的文件名。填写完成后就可以点击`Launch`开始性能分析。

> `Output File`可以使用`%i`来在文件名中生成变量，例如`output%i.log`。这样生成的报告就会以`output1.log`、`output2.log`来命令，避免了之前的报告被覆写。

![](https://github.com/Deleter-D/Images/assets/56388518/b19c5412-e6bd-4e49-8d4c-7e7b4e6cf99c)

到此，使用`ncu`的分析工作就结束了。

> 初次使用`ncu`可能会遇到权限问题。
>
> ```
> ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
> ```
>
> 这个问题就参考给出的链接，`Linux`的解决方案为在`/etc/modprobe.d`目录下创建一个后缀为`.conf`的文件，在其中加上下面一行语句。
>
> ```
> options nvidia NVreg_RestrictProfilingToAdminUsers=0
> ```
>
> 这样就将访问权限开放给了所有用户，如果想要仅`root`可用，则将上面的选项修改为`1`。添加好文件后`reboot`即可。

