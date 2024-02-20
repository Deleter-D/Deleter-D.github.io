---
title: AscendC算子开发及单算子调用
toc: true
mathjax: true
tags:
  - 异构计算
  - AscendC
categories: [高性能计算,AscendC]
abbrlink: 19902
date: 2023-10-16 11:42:58
---

摘要记录一下Ascend C的学习过程，整体来说Ascend C的核心部分还是比较易用的，唯一的小缺点就是学习初期不得不被一些无关算子核心逻辑的工程文件所干扰。

<!-- more -->

# Ascend C算子开发

笔者在阅读Ascend C官方文档的过程中发现，对于初学者来说，尤其是第一次接触异构编程思想的初学者，有很大一部分内容是无需开发者关注的，例如算子工程的相关的`CmakeLists.txt`，以及单算子调用的一些通用工具类等文件。同时，在环境配置的过程中，也发现了一些需要注意的地方，特此记录备忘。

## 环境准备

笔者的硬件及系统环境如下：

- 操作系统：openEuler release 20.03 (LTS-SP3)
- 设备：Ascend 910B

开发环境需要准备三个`run`包，分别是驱动、固件和`cann-toolkit`开发套件，笔者这里使用当前的最新版CANN包，版本号为`7.0.RC1.alpha003`。并在官网下载好对应的驱动和固件的`run`包。

### 安装流程

上述准备的三个包，按照驱动 -> 固件 -> CANN包的顺序来安装。

首先安装驱动，执行如下命令：

```sh
/path/to/Ascend-hdk-910-npu-driver_23.0.rc2_linux-aarch64.run --full --install-for-all
```

> 注意：笔者使用root用户进行安装，以`full`模式执行`run`包，并加上`install-for-all`选项来为所有用户安装。

接下来安装固件：

```sh
/path/to/Ascend-hdk-910-npu-firmware_6.4.12.1.241.run --full
```

驱动和固件都安装完成后，最好重启一次系统：

```sh
reboot
```

重启完成后，安装CANN包：

```sh
path/to/Ascend-cann-toolkit_7.0.RC1.alpha003_linux-aarch64.run --full --install-for-all
```

安装完成后，开发环境就准备好了。

### 安装过程中可能的问题

笔者在安装过程中，遇到了一个问题，很蠢，但值得注意。

问题的表现是，在按照上述的流程安装好开发环境之后，除`root`用户外的其他普通用户使用`msopgen`工具生成算子工程时，出现了权限不足的问题。但因为加上了`install-for-all`选项，所以不应该是CANN包的权限问题。然后又查看`msopgen`的代码发现，该工具将python解释器指定为了`root`用户下的`conda`环境中的解释器。

```python
#!/root/miniconda3/bin/python3
# coding=utf-8
"""
Function:
This file mainly involves main function of op generation module.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""
```

原来是`root`用户下的`conda`配置为了默认激活`base`环境，笔者安装时没有注意这一点，导致在CANN包安装的过程中，选择到了`conda`环境下的python解释器，这样一来，其他用户肯定是没有权限的。在关闭`base`环境重新安装CANN包后，问题解决。

## 算子开发流程

至此，环境准备好后，开始正式的算子开发步骤。

### 算子工程配置文件

CANN包中提供了一个自动生成算子工程的工具`msopgen`，该工具可以通过一个`json`配置文件来生成完整的算子工程，具体的编写方式请参考[官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/70RC1alpha003/operatordevelopment/ascendcopdevg/atlas_ascendc_10_0023.html)。

这里以`sinh`算子为例，该算子是一元操作，所以只需要一个输入，且输出形状与输入形状一致。根据该特征来编写`json`文件，为了贴合Ascend C官方建议的编程范式，将文件命名为`sinh_custom.json`。为了简洁，这里我们只实现一种数据类型的操作。

```json
[
    {
        "op": "SinhCustom",
        "language": "cpp",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "fp16"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND"
                ],
                "type": [
                    "fp16"
                ]
            }
        ]
    }
]
```

### 生成算子工程

创建一个文件夹用作算子工程目录，使用`msopgen`工具执行如下命令来生成算子工程。

```sh
mkdir /path/to/SinhCustom
/path/to/msopgen gen -i /path/to/sinh_custom.json -c ai_core-Ascend910B -lan cpp -out /path/to/SinhCustom
```

命令行会输出类似如下的信息：

```sh
2023-10-07 14:58:42 (942445) - [INFO] Start to generate AI Core operator files.
2023-10-07 14:58:42 (942445) - [INFO] Start to parse the ir template:/path/to/SinhCustom/sinh_custom.json
2023-10-07 14:58:42 (942445) - [INFO] Start to parse the op: SinhCustom
2023-10-07 14:58:42 (942445) - [INFO] Start to parse the input_desc: x
2023-10-07 14:58:42 (942445) - [INFO] Start to parse the output_desc: y
2023-10-07 14:58:42 (942445) - [WARNING] The "attr" value is invalid or no "attr" exists in the map.
2023-10-07 14:58:42 (942445) - [INFO] Start to check the type and format between the inputs/outputs in IR template.
2023-10-07 14:58:42 (942445) - [INFO] Start to generate a new project.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/cmake/config.cmake generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/op_host/sinh_custom_tiling.h generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/op_host/sinh_custom.cpp generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/op_kernel/sinh_custom.cpp generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/framework/tf_plugin/tensorflow_sinh_custom_plugin.cc generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] File /path/to/SinhCustom/framework/tf_plugin/CMakeLists.txt generated successfully.
2023-10-07 14:58:42 (942445) - [INFO] Generation completed.
```

此时会发现指定的输出目录只已经生成了一系列的算子工程文件。

```sh
SinhCustom
├── build.sh
├── cmake
├── CMakeLists.txt
├── CMakePresets.json # 这个配置项需要修改
├── framework
├── op_host
│   ├── CMakeLists.txt
│   ├── sinh_custom.cpp # 算子host侧核心逻辑
│   └── sinh_custom_tiling.h # 算子tiling结构体定义
├── op_kernel
│   ├── CMakeLists.txt
│   └── sinh_custom.cpp # 算子kernel侧核心逻辑
├── scripts
└── sinh_custom.json # 笔者此处将工程配置文件和算子工程目录放在了一起
```

我们只需要专注于上述带有注释的几个文件即可。

此处先修改与算子核心逻辑无关的配置项`CMakePresets.json`，官方文档中也描述的非常清楚，只需要将`ASCEND_CANN_PACKAGE_PATH`配置项修改为实际的CANN包安装路径即可。在`root`用户下安装的默认路径为`/usr/local/Ascend/ascend-toolkit/latest`。

以上将所有无关算子逻辑的内容修改完毕，接下来就可以专注于算子开发了。

### 算子逻辑开发

官方文档中推荐先实现`kernel`侧的逻辑，但笔者有一些不同的看法。我推荐先实现算子`tiling`结构体的定义与具体策略，这样做的好处是，可以提前将`tiling`策略所需的变量确定下来，并且借助于CANN包只提供的一系列宏，这一过程并不需要很大的工作量。在实现`kernel`侧逻辑的过程中，这些变量将有助于思考数据在逻辑核上如何具体分配和执行，当然这只是笔者的观点，可以根据自己的编程习惯来作调整。

#### `tiling`结构体定义及策略实现

首先确定`tiling`过程中所需的变量，参考官方样例，需要定义整块、尾块的个数及其中的元素个数，还需要定义最小对齐单位。`op_host/sinh_custom_tiling.h`代码如下：

```cpp
#ifndef SINH_CUSTOM_TILING_H // 头文件保护记得加上，自动生成的文件中不包含
#define SINH_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);    // 整块个数
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);      // 尾块个数
  TILING_DATA_FIELD_DEF(uint32_t, formerLength); // 整块内元素个数
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);   // 尾块内元素个数
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);     // 最小对齐单位，元素个数
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(SinhCustom, TilingData)
}

#endif
```

然后在`op_host/sinh_custom.cpp`中实现具体的`tiling`策略，代码如下：

```cpp
namespace optiling
{
    constexpr uint32_t BLOCK_DIM = 24;                        // 划分核心数量
    constexpr uint32_t SIZE_OF_HALF = 2;                      // 数据类型的字节数
    constexpr uint32_t BLOCK_SIZE = 32;                       // 昇腾设备上的数据block为32字节
    constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_HALF; // 最小对齐单位
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(BLOCK_DIM);

        // 使输入向上对齐
        uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        // 计算整块和尾块个数
        uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
        uint32_t tailNum = BLOCK_DIM - formerNum;
        // 计算整块和尾块的元素个数
        uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;

        // 设置tiling参数
        tiling.set_formerNum(formerNum);
        tiling.set_tailNum(tailNum);
        tiling.set_formerLength(formerLength);
        tiling.set_tailLength(tailLength);
        tiling.set_alignNum(ALIGN_NUM);

        // 以下为固定写法，不用纠结
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}
```

#### `kernel`侧实现

有了上述实现的`tiling`策略，我们就可以根据数据划分的逻辑来确定`kernel`侧的具体实现。根据官方推荐的矢量编程范式，我们可以先将算子类的框架写出来，再慢慢填充内容。在`op_kernel/sinh_custom.cpp`中写出算子类框架。

```cpp
using namespace AscendC; // 记得开启AscendC命名空间
constexpr int32_t BUFFER_NUM = 2; // TQue的缓冲数量，此处开启双Buffer

class KernelSinh
{
public:
    __aicore__ inline KernelSinh() {} // 类构造函数，无须任何代码
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 				  // 初始化函数的参数为输入、输出
                                uint32_t formerNum, uint32_t tailNum, // 以及上面定义的一系列tiling参数
                                uint32_t formerLength, uint32_t tailLength,
                                uint32_t alignNum) { /* TODO */ }
    __aicore__ inline void Process() { /* TODO */ }

private:
    __aicore__ inline void CopyIn() { /* TODO */ }
    __aicore__ inline void Compute() { /* TODO */ }
    __aicore__ inline void CopyOut() { /* TODO */ }

private:
    /* TODO */
};
```

第一步应该做的是分析算子类的私有数据成员，首先一定需要的是用来管理内存的`Tpipe`，同时需要输入输出分别对应的`TQue`和`GlobalTensor`，同时每个逻辑核还 需要直到当前处理的数据个数，所以需要一个变量`tileLength`来确定分片大小。

再来分析算子，公式如下所示。
$$
{\bf y}=\text{sinh}({\bf x})=\frac{e^{\bf x}-e^{-{\bf x}}}{2.0}
$$
可以观察到，我们需要计算两个中间结果，分别是$e^{\bf x}$和$e^{-{\bf x}}$，所以需要相应的数据结构来存放这两个中间结果，Ascend C提供的`TBuf`可以很好的承担这一责任。

至此我们就将算子类需要的私有数据成员确定了下来。

```cpp
TPipe pipe;                                      // 用于操作队列
TBuf<QuePosition::VECCALC> tempBuf;              // 存放中间结果
TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;   // 输入队列
TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; // 输出队列
GlobalTensor<DTYPE_X> xGm;                       // 输入数据对应的GM内存空间
GlobalTensor<DTYPE_Y> yGm;                       // 输出数据对应的GM内存空间
uint32_t tileLength;                             // 每个逻辑核需要知道分片数据个数
```

接下来要做的是完善算子类的初始化函数`Init()`，在该函数中我们需要为`GlobalTensor`分配内存，并初始化相应的`TQue`，同时需要针对某些变量做合法性判断。

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                            uint32_t formerNum, uint32_t tailNum,
                            uint32_t formerLength, uint32_t tailLength,
                            uint32_t alignNum)
{
    if (GetBlockIdx() < formerNum)
    {
        // 处理整块逻辑
        this->tileLength = formerLength;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + formerLength * GetBlockIdx(), formerLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + formerLength * GetBlockIdx(), formerLength);
    }
    else
    {
        // 处理尾块逻辑
        this->tileLength = tailLength;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
    }

    ASSERT(alignNum != 0 && "align num can not be zero!");
    pipe.InitBuffer(inQueueX, BUFFER_NUM, (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
}
```

再然后就是算子最核心的部分——计算逻辑，分别实现矢量编程范式的三步骤。

```cpp
__aicore__ inline void CopyIn()
{
    LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    DataCopy(xLocal, xGm, this->tileLength); // GM -> LM
    inQueueX.EnQue<DTYPE_X>(xLocal);
}
__aicore__ inline void Compute()
{
    LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
    pipe.InitBuffer(tempBuf, this->tileLength * sizeof(DTYPE_X));
    LocalTensor<DTYPE_X> tempLocal = tempBuf.Get<DTYPE_X>(this->tileLength);
    // 计算exp(x)
    Exp(yLocal, xLocal, this->tileLength);
    // 计算-x
    half nagOne(-1.0);
    Muls(tempLocal, xLocal, nagOne, this->tileLength);
    // 计算exp(-x)
    Exp(tempLocal, tempLocal, this->tileLength);
    // 计算exp(x)-exp(-x)
    Sub(yLocal, yLocal, tempLocal, this->tileLength);
    // 计算最终结果
    half denominator(0.5);
    Muls(yLocal, yLocal, denominator, this->tileLength);
    outQueueY.EnQue<DTYPE_Y>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
__aicore__ inline void CopyOut()
{
    LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
    DataCopy(yGm, yLocal, this->tileLength); // LM -> GM
    outQueueY.FreeTensor(yLocal);
}
```

实现的具体细节与接口可以参考官方文档。

最后再将`Process()`函数补全，并完善核函数。

```cpp
__aicore__ inline void Process()
{
    CopyIn();
    Compute();
    CopyOut();
}
```

```cpp
extern "C" __global__ __aicore__ void
sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    op.Init(x, y,
            tiling_data.formerNum, tiling_data.tailNum,
            tiling_data.formerLength, tiling_data.tailLength,
            tiling_data.alignNum);
    if (TILING_KEY_IS(1))
    {
        op.Process();
    }
}
```

至此就完成了`kernel`侧的实现。

#### `host`侧实现

我们回到`op_host/sinh_custom.cpp`，关于类型推导函数，这个算子输入输出的形状一致。`msopgen`生成的算子工程中，默认即为输入输出形状一致，所以无须改动。如果在写其他复杂算子的时候，需要仔细分析数据形状的变化。关于算子原型注册，也无须改动。

现在就完成了整个算子的逻辑，可以执行`build.sh`来验证有没有编译时错误，若没有错误则可以进行运行时验证。

# 核函数调用

笔者直接将官方的核函数调用样例拿来做了一些修改，需要修改的地方如下。

```sh
kernel_invocation
├── cmake
├── CMakeLists.txt
├── data_utils.h
├── input
├── main.cpp # 需要修改
├── output
├── run.sh # 需要修改
├── add_custom.cpp # 替换为自己的算子实现
├── add_custom.py # 需要修改
└── verify_result.py # 添加的代码，用于验证结果
```

将官方样例中的`add_custom.cpp`替换为自己实现的`kernel`侧算子，笔者这里的名称为`sinh_custom.cpp`。同时为了CPU侧调试，需要添加一个核函数的包装函数，代码如下。

```cpp
#ifndef __CCE_KT_TEST__
void sinh_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y)
{
    sinh_custom<<<blockDim, l2ctrl, stream>>>(x, y);
}
#endif
```

> 注意：为了快速验证逻辑，在核函数验证过程中未使用动态`tiling`，所以没有之前提到的那些`tiling`参数。

然后是`sinh_custom.py`，官方样例中是`add_custom.py`，这里修改文件名称，因为后面的`run.sh`中是通过算子文件名来调用这一python脚本的。

由于本算子只需要一个输入向量，所以只生成一个`input`数据，然后修改`golden`数据的生成方式，调用`numpy`中与算子功能相同的函数来计算，注意数据类型，代码如下。

```python
import numpy as np


def gen_golden_data_simple():
    np.random.seed(42)
    input_x = np.random.randn(8, 2048).astype(np.float16)
    golden = np.sinh(input_x).astype(np.float16)
    print(f'-----------------------{input_x[0][0]}')
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
```

`main.cpp`中要调整相应的内存申请等操作，只需要一个`input`，CPU侧调试和NPU侧调试的代码都需要修改，具体如下。

```cpp
#include <stdio.h>

#include "data_utils.h"
#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void sinh_custom_do(uint32_t coreDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y);
#endif

int32_t main(int32_t argc, char *argv[])
{
  size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
  size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
  uint32_t blockDim = 8;

#ifdef __CCE_KT_TEST__
  uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
  uint8_t *y = (uint8_t *)AscendC::GmAlloc(outputByteSize);

  ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);

  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  ICPU_RUN_KF(sinh_custom, blockDim, x, y);

  WriteFile("./output/output_y.bin", y, outputByteSize);

  AscendC::GmFree((void *)x);
  AscendC::GmFree((void *)y);
#else
  CHECK_ACL(aclInit(nullptr));
  aclrtContext context;
  int32_t deviceId = 0;
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  uint8_t *xHost, *yHost;
  uint8_t *xDevice, *yDevice;
  CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
  CHECK_ACL(aclrtMallocHost((void **)(&yHost), outputByteSize));
  CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&yDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
  CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

  sinh_custom_do(blockDim, nullptr, stream, xDevice, yDevice);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(yHost, outputByteSize, yDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./output/output_y.bin", yHost, outputByteSize);

  CHECK_ACL(aclrtFree(xDevice));
  CHECK_ACL(aclrtFree(yDevice));
  CHECK_ACL(aclrtFreeHost(xHost));
  CHECK_ACL(aclrtFreeHost(yHost));

  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
#endif
  return 0;
}
```

原样例中的验证方式是求md5和，但由于核函数中调用了`Exp`、`Muls`等API，所以精度可能会有损失，不适合用md5sum的方式来验证。这里就需要引入新的文件`verify_result.py`，这里使用了`numpy.isclose`函数来进行验证，这也是官方单算子API调用的结果验证方式。

```python
import sys
import math
import numpy as np


def data_compare(file1, file2,file3):
    input1 = np.fromfile(file1, dtype=np.float16)
    print("input1: ", input1)
    golden = np.fromfile(file2, dtype=np.float16)
    output = np.fromfile(file3, dtype=np.float16)
    print("output: ", output)
    print("-------------golden is :")
    print("golden: ", golden)

    different_element_results = np.isclose(
        output, golden,
        rtol=5e-2,
        atol=1e-3,
        equal_nan=True)
    different_element_indexes = np.where(
        different_element_results != np.array((True,)))[0]
    if different_element_indexes.size == 0:
        print("result correct!")
    else:
        print("result error!")
    return 0 if different_element_indexes.size == 0 else 1


if __name__ == '__main__':
    intput_file1 = sys.argv[1]
    golden_file = sys.argv[2]
    output_file = sys.argv[3]
    cmp_result = data_compare(intput_file1, golden_file, output_file)

    if (cmp_result == 0):
        sys.exit(0)
    else:
        sys.exit(1)
```

最后是修改`run.sh`脚本，需要修改的只有最后验证结果的部分。原样例的验证方式是md5sum。

```sh
echo "md5sum: ";md5sum output/*.bin
```

修改为调用脚本判断。

```sh
echo "result verification: "
python3 verify_result.py ./input/input_x.bin ./output/golden.bin ./output/output_y.bin
```

# 单算子API调用

单算子调用是通过自动生成的两段式API来执行的，为了快速验证，同样是将官方样例中的单算子API调用样例拿来做了一些修改。需要修改的几处关键代码如下。

```sh
aclnn_online_model
├── build
├── inc
├── README.md
├── run
│   └── out
│       ├── execute_sinh_op
│       ├── result_files
│       └── test_data
│           ├── config
│           └── data
│               ├── generate_data.py # 生成测试数据脚本，需要修改
├── run.sh # 需要修改
├── scripts
│   └── verify_result.py # 调整验证方式，例如相对和绝对误差参数等
└── src
    ├── CMakeLists.txt # 需要修改
    ├── common.cpp
    ├── main.cpp # 需要修改
    ├── operator_desc.cpp
    └── op_runner.cpp # 需要修改
```

具体细节如下。

`generate_data.py`中，按照算子来修改测试数据生成方式。本算子需要`half`类型的测试数据，故代码改为：

```python
import numpy as np

a = np.random.randn(8, 2048).astype(np.float16)

a.tofile('input_0.bin')
```

`verify_result.py`中，根据实际读取的输入和输出，利用`np.isclose`来进行比较，该函数详细用法参考`numpy`官方文档。

```python
import sys
import math
import numpy as np


def data_compare(file1, file2):
    input1 = np.fromfile(file1, dtype=np.float16)
    print("input1: ", input1)
    golden = np.sinh(input1).astype(np.float16)
    output = np.fromfile(file2, dtype=np.float16)
    print("output: ", output)
    print("-------------golden is :")
    print("golden: ", golden)

    different_element_results = np.isclose(
        output, golden,
        rtol=5e-2,
        atol=1e-3,
        equal_nan=True)
    different_element_indexes = np.where(
        different_element_results != np.array((True,)))[0]
    return 0 if different_element_indexes.size == 0 else 1


if __name__ == '__main__':
    intput_file1 = sys.argv[1]
    output_file = sys.argv[2]
    cmp_result = data_compare(intput_file1, output_file)

    if (cmp_result == 0):
        sys.exit(0)
    else:
        sys.exit(1)
```

`main.cpp`中，需要将`CreateOpDesc()`函数根据具体的输入输出来做修改。

```cpp
OperatorDesc CreateOpDesc()
{
    std::vector<int64_t> shape{8, 2048};
    aclDataType dataType = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    return opDesc;
}
```

`op_runner.cpp`中将两段式API修改为自己算子的API，请善用`Ctrl + F`搜索关键代码进行修改，具体的API名称可以查看算子目录下的`build_out/autogen`目录。

```cpp
...
auto ret = aclnnSinhCustomGetWorkspaceSize(inputTensor_[0], outputTensor_[0], &workspaceSize, &handle);
...
INFO_LOG("Execute aclnnSinhCustomGetWorkspaceSize success, workspace size %lu", workspaceSize);
...
if (aclnnSinhCustom(workspace, workspaceSize, handle, stream) != ACL_SUCCESS)
{
    ...
}
INFO_LOG("Execute aclnnSinhCustom success");
...
```

接着修改`src/CMakeLists.txt`。

```cmake
set(AUTO_GEN_PATH "../SinhCustom/build_out/autogen") # 16行

# 50行以后，修改可执行文件的名称
add_executable(execute_sinh_op
    ${AUTO_GEN_PATH}/aclnn_sinh_custom.cpp
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    op_runner.cpp
    common.cpp
)

target_link_libraries(execute_sinh_op
    ascendcl
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_sinh_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
```

最后修改`run.sh`脚本中关于路径的部分。修改完成后，就可以执行`run.sh`脚本进行单算子API调用了。

```sh
INFO: acl executable run success!
input1:  [ 0.468  -0.2585 -3.066  ...  0.9136 -1.117  -1.368 ]
output:  [  0.485   -0.2615 -10.71   ...   1.047   -1.365   -1.837 ]
-------------golden is :
golden:  [  0.4854  -0.2615 -10.71   ...   1.046   -1.364   -1.837 ]
INFO: compare golden data success!
```

出现上述提示证明算子通过验证。
