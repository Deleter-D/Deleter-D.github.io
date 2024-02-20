---
title: CUDA编程——GPU加速库和OpenACC
toc: true
mathjax: true
tags:
  - CUDA
  - 高性能计算
  - 异构计算
categories:
  - 高性能计算
  - CUDA
abbrlink: 9782
date: 2024-02-20 16:49:53
---

很多人是参考《Professional CUDA C Programming》一书来入门CUDA的，这本书本身是很好的入门材料，但由于CUDA版本迭代非常快，导致书中的一些内容已经是过时的了。这也是笔者撰写本系列博客的初衷之一，这个系列参考了本书以及CUDA 12.x的官方文档，并在每个章节都附有详细的代码参考，并且代码是基于CUDA 12.x的，可以解决一些由于版本迭代带来的问题。本系列的博客由《Professional CUDA C Programming》一书、CUDA官方文档、互联网上的一些资料以及笔者自己的理解构成，希望能对你有一些帮助，若有错误也请大胆指出。

<!-- more -->

## CUDA库概述

CUDA支持的库及其作用域如下表所示。

| 库名                            | 作用域                       | 官方文档                                                     |
| ------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| NVIDIA CUDA Math Library        | 数学运算                     | https://docs.nvidia.com/cuda/cuda-math-api/index.html        |
| NVIDIA cuBLAS                   | 线性代数                     | https://docs.nvidia.com/cuda/cublas/index.html               |
| NVIDIA cuSPARSE                 | 稀疏线性代数                 | https://docs.nvidia.com/cuda/cusparse/index.html             |
| NVIDIA CUSP                     | 稀疏线性代数和图形计算       | https://cusplibrary.github.io/index.html                     |
| NVIDIA cuFFT                    | 快速傅里叶变换               | https://docs.nvidia.com/cuda/cufft/index.html                |
| NVIDIA cuRAND                   | 随机数生成                   | https://docs.nvidia.com/cuda/curand/index.html               |
| NVIDIA NPP                      | 图像和信号处理               | https://docs.nvidia.com/cuda/npp/index.html                  |
|                                 |                              |                                                              |
| MAGMA                           | 新一代线性代数               | https://icl.utk.edu/magma/                                   |
| IMSL Fortran Numerical Library  | 数学与统计学                 | https://www.imsl.com/products/imsl-fortran-libraries         |
| AccelerEyes ArrayFire           | 数学，信号和图像处理，统计学 | https://arrayfire.com/                                       |
| Thrust                          | 并行算法和数据结构           | https://docs.nvidia.com/cuda/thrust/index.html               |
| Geometry Performance Primitives | 计算几何                     | https://developer.nvidia.com/geometric-performance-primitives-gpp |
| Paralution                      | 稀疏迭代方法                 | https://www.paralution.com/                                  |
| AmgX                            | 核心求解                     | https://github.com/NVIDIA/AMGX                               |

CUDA的库有一些通用的工作流：

- 在库操作中创建一个特定的库句柄来管理上下文信息；
- 为库函数的输入输出分配设备内存；
- 如果输入格式不是函数库支持的格式则需要进行转换；
- 将输入以支持的格式填入预先分配的设备内存中；
- 配置要执行的库运算；
- 执行一个将计算部分交付给GPU的库函数调用；
- 取回设备内存中的计算结果（结果可能是库设定的格式）；
- 如有必要，将取回的数据转换成应用程序的原始格式；
- 释放CUDA资源；
- 继续完成应用程序的其他工作。

## cuSPARSE库

较新版本的cuSPARSE将API分为了两大类：

- Legacy：这部分接口是为了兼容旧版本所保留的，在未来的版本也不会改进；
- Generic：这部分是cuSPARSE的标准接口。

下面的讨论都基于Generic系列接口。

Generic系列接口大体分为几类：

- 稀疏向量与稠密向量之间的操作（`Axpby`、`Gather`、`Scatter`、求和、点积）；
- 稀疏向量与稠密矩阵之间的操作（乘积）；
- 稀疏矩阵与稠密向量之间的操作（乘积、三角线性方程求解、三对角、五对角线性方程求解）；
- 稀疏矩阵与稠密矩阵之间的操作（乘积、三角线性方程求解、三对角、五对角线性方程求解）；
- 稀疏矩阵与稀疏矩阵之间的操作（求和、乘积）；
- 稠密矩阵与稠密矩阵之间的操作，输出一个稀疏矩阵（乘积）；
- 稀疏矩阵预处理（不完全Cholesky分解、不完全LU分解）；
- 不同稀疏矩阵存储格式的相互转换。

### cuSPARSE数据存储格式

cuSPARSE的索引有两种，从零开始和从一开始的，这是为了兼容`C/C++`和`Fortran`。

#### 向量存储格式

稠密向量不过多介绍，与`C/C++`的数组存储方式是一致的。

稀疏向量是借助两个数组表示的：

- 值数组`values`：存储向量中的非零值；
- 索引数组`indices`：存储向量中非零值在等价稠密向量中的索引。

官方文档中的图片很好的解释了这种存储方式。

<img src="https://github.com/Deleter-D/Images/assets/56388518/5356cf8f-d236-4e1f-b18e-2348c6d0943a" style="zoom:25%;" />

#### 矩阵存储格式

##### 稠密矩阵

稠密矩阵有行优先和列优先两种组织方式，通过几个参数来表示：

- 矩阵行数`rows`；
- 矩阵列数`columns`；
- 主维度`leading_dimension`：主维度在行优先模式下必须大于等于列数，在列优先模式下必须大于等于行数；
- 值数组指针：该数组的长度在行优先模式下为`rows * leading_dimension`，在列优先模式下为`columns * leading_dimension`。

下图是一个$5\times 2$的稠密矩阵在两种模式下的内存布局。

<img src="https://github.com/Deleter-D/Images/assets/56388518/c0295ca3-e6c7-4f35-a6eb-b7ec254b0df4" style="zoom: 25%;" />

这里比较特殊的一个参数就是主维度`leading_dimension`，这个参数的存在是为了更好的表示子矩阵。下图是官方文档中的示例。

<img src="https://github.com/Deleter-D/Images/assets/56388518/405673bc-5fd7-4208-ae44-a0094fe79547" style="zoom:25%;" />

我们推广这个示例，将一个`rows * columns`的矩阵以行优先存储，并令其`leading_dimension`为`columns`。此时取它的一个`m * n`的子矩阵，起始元素指针为`sub`，想要得到其`(i, j)`位置的元素，只需要利用如下计算公式：

```
sub_ij = sub + j * leading_dimension + i;
```

列优先存储同理。

##### 稀疏矩阵

###### 坐标存储Coordinate (COO)

COO是一种利用非零元素及其坐标来存储稀疏矩阵的方式，主要有如下参数表示：

- 矩阵行数`rows`；
- 矩阵列数`columns`；
- 非零元素个数`nnz`；
- 行索引数组指针`row_indices`：其长度为`nnz`，存放了非零元素在等价稠密矩阵中的行索引；
- 列索引数组指针`column_indices`：其长度为`nnz`，存放了非零元素在等价稠密矩阵中的列索引；
- 值数组指针`values`：其长度为`nnz`，存放了矩阵的非零元素，按照等价稠密矩阵行优先的顺序排列。

COO的每一项由一个`<row, column>`的二元组表示，COO默认是按照行的顺序排序的。

![](https://github.com/Deleter-D/Images/assets/56388518/5bb89fb8-5c28-4ad2-9eff-742ee6966220)

若想计算COO格式下的元素在等价稠密矩阵中的位置，可以通过如下公式：

```c++
// 行优先
rows_indices[i] * leading_dimension + column_indices[i];

// 列优先
column_indices[i] * leading_dimension + row_indices[i];
```

###### 压缩稀疏行Compressed Sparse Row (CSR)

CSR和COO非常类似，只是将行索引数组进行了压缩，用一个行偏移数组来代替了。

- 行偏移数组`row_offsets`：其长度为`rows + 1`，存储了每一行起始元素在列索引数组和值索引数组中的位置；
- 其余参数与COO一致。

![](https://github.com/Deleter-D/Images/assets/56388518/b1e1c0e8-83a0-4b8c-a689-803396446ac9)

若想计算CSR格式下的元素在等价稠密矩阵中的位置，可以通过如下公式：

```cpp
// 行优先
row * leading_dimension + column_indices[row_offsets[row] + k]

// 列优先
column_indices[row_offsets[row] + k] * leading_dimension + row
```

> 其中，`row`表示稠密矩阵的第几行，`k`的范围是`k = 0; k < row_offsets[row + 1] - row_offsets[row]`。

此外还有CSC、SELL、BSR、BLOCKED-ELL等稀疏矩阵的存储方式，详细参考官方文档的介绍。

### 具体示例

我们来实现一个比较常见的操作${\bf Y}=\alpha{\bf A\cdot X}+\beta{\bf Y}$，也就是`SpMV`函数。接下来的内容重点在于cuSPARSE库一些通用操作。

首先需要创建一个句柄。

```cpp
cusparseHandle_t handle;
ERROR_CHECK_CUSPARSE(cusparseCreate(&handle));
```

由于我们的数据大部分是在主机端准备的，所以生成的数据自然而然地是以稠密的形式存储的。所以需要进行稠密矩阵到稀疏矩阵的转换。这一部分的工作可能有一些复杂，总体步骤包含：

- 创建cuSPARSE的稠密和稀疏矩阵；
- 判断是否需要额外的buffer；
- 分析非零元素个数；
- 准备特定稀疏矩阵格式所需要的空间；
- 执行转换。

```cpp
// 创建稠密矩阵
cusparseDnMatDescr_t dn_mat;
ERROR_CHECK_CUSPARSE(cusparseCreateDnMat(&dn_mat, rows, columns, ld, d_A, CUDA_R_32F, CUSPARSE_ORDER_ROW));

// 创建稀疏矩阵
cusparseSpMatDescr_t sp_mat;
ERROR_CHECK_CUSPARSE(cusparseCreateCsr(&sp_mat, rows, columns, 0, d_row_offsets_A, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

// 若有必要，为转换工作申请额外的buffer
size_t buffer_size = 0;
void *d_buffer     = NULL;
ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size)); // 该函数返回所需的buffer大小
ERROR_CHECK(cudaMalloc(&d_buffer, buffer_size));

// 分析矩阵中的非零元素个数
ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));
// 获取非零元素个数
int64_t rows_tmp, cols_tmp, nnz;
ERROR_CHECK_CUSPARSE(cusparseSpMatGetSize(sp_mat, &rows_tmp, &cols_tmp, &nnz));

// 申请CSR中的列索引数组和值数组
int *d_column_indices_A;
float *d_values_A;
ERROR_CHECK(cudaMalloc((void **)&d_column_indices_A, nnz * sizeof(int)));
ERROR_CHECK(cudaMalloc((void **)&d_values_A, nnz * sizeof(float)));

// 为稀疏矩阵设置各个数组指针
ERROR_CHECK_CUSPARSE(cusparseCsrSetPointers(sp_mat, d_row_offsets_A, d_column_indices_A, d_values_A));

// 执行稠密矩阵到稀疏矩阵的转换
ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));
```

准备好矩阵后，接着准备参与运算的两个稠密向量。

```cpp
cusparseDnVecDescr_t dn_vec_X, dn_vec_Y;
ERROR_CHECK_CUSPARSE(cusparseCreateDnVec(&dn_vec_X, columns, d_X, CUDA_R_32F));
ERROR_CHECK_CUSPARSE(cusparseCreateDnVec(&dn_vec_Y, rows, d_Y, CUDA_R_32F));
```

最后就是执行计算，但在执行真正的计算之前，依然需要判断是否需要额外的buffer。

```cpp
// 若有必要，为SpMV计算申请额外的buffer
float alpha = 3.0f;
float beta  = 4.0f;
size_t spmv_buffer_size;
void *d_spmv_buffer;
ERROR_CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sp_mat, dn_vec_X, &beta, dn_vec_Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size));
ERROR_CHECK(cudaMalloc(&d_spmv_buffer, spmv_buffer_size));

// 执行SpMV计算
ERROR_CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sp_mat, dn_vec_X, &beta, dn_vec_Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer));
```

> 注意，无论是计算之前，还是前面提到的稠密转稀疏之前，它们判断是否需要额外buffer的操作，全部交给库来自行判断，不需要人为干预。程序员需要做的只是写一个像上面一样较为通用的代码，使其能够在需要buffer的时候申请得到即可。

> 示例完整代码参考[cusparse.cu](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/01_cusparse.cu)。关于其他API就不过多阐述了，用到的时候查官方文档即可。

## cuBLAS库

cuBLAS库与cuSPARSE库最大的不同在于，cuBLAS库并不支持多种稀疏矩阵类型，它更擅长处理稠密矩阵和稠密向量的运算。

当前的cuBLAS库将接口分为了四类：

- cuBLAS API（CUDA 6.0开始）；
- cuBLASXt API（CUDA 6.0开始）；
- cuBLASLt API（CUDA 10.1开始）；
- cuBLASDx API（未包含在CUDA Toolkit中）。

上面四套API的主要区别在于：

- cuBLAS API需要将数据搬移到设备上进行运算；
- cuBLASXt API可以将数据放在主机或参与运算的任何设备上，库会承担运算和数据分发的责任；
- cuBLASLt API是一套专注于通用矩阵乘（GEMM）的灵活的轻量级API，该API可以通过参数来灵活指定矩阵数据布局、输入类型、计算类型以及算法的实现。用户指定了一组预期的GEMM操作后，这组操作可以根据不同的输入来复用；
- cuBLASDx API则是一个设备端API扩展，可以在核函数中执行BLAS计算。通过融合数值运算，可以减少延迟并进一步提高性能。目前该组API还在preview阶段。

同时，cuBLAS API存在新旧两套API，后面的所有讨论都基于定义在`cublas_v2.h`头文件中的新版API，旧版API定义在`cublas.h`中。具体的区别这里不过多讨论，有兴趣可以查看官方文档[New and Legacy cuBLAS API](https://docs.nvidia.com/cuda/cublas/index.html#new-and-legacy-cublas-api)。

### cuBLAS数据存储格式

cuBLAS有两套数据排布方式，一套为了兼容Fortran从1开始的索引，另一套是兼容C从0开始的索引。可以通过以下两个宏来计算全局索引。

```cpp
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
```

要记住最核心的一点，cuBLAS是以**列主序**的形式存储矩阵的。

### 具体示例

在熟悉了cuSPARSE的使用之后，你会发现cuBLAS要简洁很多，因为少了很多配置稀疏矩阵的过程，下面以通用矩阵乘法为例说明。

同样地，首先创建句柄。

```cpp
cublasHandle_t handle;
ERROR_CHECK_CUBLAS(cublasCreate(&handle));
```

然后就可以直接进行计算了，如果需要的话，可以使用`cublasSetStream()`来绑定一个流。

```cpp
ERROR_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
```

> 详细代码参考[cublas.cu](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/02_cublas.cu)。

## cuFFT库

cuFFT库由两部分组成：

- cuFFT：提供GPU上的高性能快速傅里叶变换操作，它需要提前将数据搬移到设备端；
- cuFFTW：为FFTW的用户提供快速移植到GPU的能力，为了这种快速移植能力，cuFFTW支持主机端的数据传入，它将自动为用户处理如`cudaMalloc`、`cudaMemcpy`等操作。

下面的内容重点介绍cuFFT。cuFFT提供一种简单易用的配置机制称为`plan`，它使用内部构建的block来优化给定配置和特定GPU硬件之间的转化。一旦创建了一个`plan`，库将自动保存多次执行该`plan`所需的所有状态，无需重新配置。不同类型的FFT需要不同的线程配置和GPU资源，`plan`接口提供了这样一种简单的配置重用方式。

cuFFT支持多种类型的变换，如复数-复数变换（C2C）、复数-实数变换（C2R）、实数-复数变换（R2C）。由于变换的不同，需要的输入输出数据布局也就不同。

### cuFFT数据存储格式

在cuFFT中，数据布局严格取决于配置和变换的类型。一般地，C2C变换情况下，输入输出数据应为`cufftComplex`或`cufftDoubleComplex`，具体取决于计算精度。而C2R变换，只需要非冗余的复数元素组成的向量作为输入，输出则是由`cufftReal`或`cufftDouble`元素组成的向量。对于R2C变换，需要一个实数向量作为输入，输出一个非冗余复数元素组成的向量。

在C2R和R2C变换中，输入输出的大小是不同的。对于非就地变换，创建一个大小合适的输出数组即可。但对于就地变换，程序员应当使用填充的数据布局，这种布局与FFTW兼容。无论是就地C2R还是R2C变换，输出的起始地址都与输入的起始地址一致，所有应当填充R2C中的输入或C2R中的输出数据。

以一维变换为例，期望的输入输出大小以及类型如下表所示。

| FFT类型 | 输入数据大小（类型）                            | 输出数据大小（类型）                            |
| ------- | ----------------------------------------------- | ----------------------------------------------- |
| C2C     | $x$（`cufftComplex`）                           | $x$（`cufftComplex`）                           |
| C2R     | $\lfloor\frac{x}{2}\rfloor+1$（`cufftComplex`） | $x$（`cufftReal`）                              |
| R2C     | $x$（`cufftReal`）                              | $\lfloor\frac{x}{2}\rfloor+1$（`cufftComplex`） |

对于多维的情况，参考官方文档[multidimensional-transforms](https://docs.nvidia.com/cuda/cufft/index.html#multidimensional-transforms)。

### 具体示例

关于傅里叶变换算法本身这里不过多展开，下面用一个比较基本的一维复数-复数FFT来说明。

首先依旧是创建句柄，这里之所以将句柄命名为`plan`是因为，后续创建cuFFT的`plan`时，是基于这个句柄的。

```cpp
cufftHandle plan;
ERROR_CHECK_CUFFT(cufftCreate(&plan));
```

创建`plan`，这个`plan`可以复用。如果需要的话，可以使用`cufftSetStream()`来绑定一个流。

```cpp
ERROR_CHECK_CUFFT(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
```

执行变换操作，这里进行了一次正向变换，归一化后又进行了逆向变换。

```cpp
// 执行正向变换
ERROR_CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
// 归一化
scaling_kernel<<<1, 128>>>(d_data, element_count, 1.f / fft_size);
// 执行逆向变换
ERROR_CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
```

> 详细代码参考[cufft.cu](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/03_cufft.cu)。

## cuRAND库

介绍cuRAND库之前要引入两个与随机数生成相关的概念：

- PRNG：伪随机数生成器；
- QRNG：拟随机数生成器。

PRNG和QRNG的最大区别就在于，生成每一个随机数的事件是否为独立事件。PRNG每次采样均为独立统计事件，这意味着每次采样，所得到某个数的概率是相同的。而QRNG每次采样并不是独立事件，它会尽可能的均匀填充输出类型的范围。一个更具体的例子是，假设第一个生成的随机数是2的概率为$P_0$，下一个生成的随机数同样是2的概率为$P_1$。在PRNG中$P_1$不会因为上一次取得的数是2就变小，它与$P_0$是完全相等的，但在QRNG中，$P_1$会由于$P_0$的成立而变小。

cuRAND库与其他库最大的不同就是，它提供了主机端和设备端两套API。

主机端API定义在头文件`curand.h`中。但要注意的是，主机端API允许两种生成方式：主机生成和设备生成。若生成时传入的数据指针是主机内存指针，则生成过程由CPU在主机端完成，结果也存储在主机内存中。若生成时传入的数据指针是设备内存指针，则生成过程由设备端完成，结果存储在设备的全局内存中。

设备端API定义在头文件`curand_kernel.h`中，可以在核函数中直接生成随机数并使用，而不需要将生成结果存入全局内存后再读取。

cuRAND库的RNG有9种，分别有5种PRNG和4种QRNG：

- PRNG：
  - CURAND_RNG_PSEUDO_XORWOW：使用XORWOW算法实现的，XORWOW算法是伪随机数生成器xor-shift系列的成员；
  - CURAND_RNG_PSEUDO_MRG32K3A：组合多重递归伪随机数生成器系列的成员；
  - CURAND_RNG_PSEUDO_MTGP32：Mersenne Twister伪随机数生成器系列的成员，具有为GPU定制的参数；
  - CURAND_RNG_PSEUDO_MT19937：Mersenne Twister伪随机数生成器系列的成员，参数与CPU版本相同，但顺序不同，仅支持主机API，并且只能在架构sm_35或更高版本上使用；
  - CURAND_RNG_PSEUDO_PHILOX4_32_10：Philox系列的成员，三大基于非加密计数器的随机数生成器之一。
- QRNG：
  - CURAND_RNG_QUASI_SOBOL32：32位序列的Sobol生成器；
  - CURAND_RNG_QUASI_SCRAMBLED_SOBOL32：添加扰乱的32位序列的Sobol生成器；
  - CURAND_RNG_QUASI_SOBOL64：64位序列的Sobol生成器；
  - CURAND_RNG_QUASI_SCRAMBLED_SOBOL64：添加扰乱的64位序列的Sobol生成器。

> cuRAND中的QRNG都是基于Sobol算法的，它以方向向量作为种子，上述的四种变体每种都能产生高达20000维的序列。

### 具体示例

#### 主机端API

主机端API调用流程如下：

- 创建生成器并指定RNG类型；
- 设置偏移量（`offset`）、排序方式（`ordering`）、种子（`seed`）；
- 从指定分布中执行生成任务；
- 若在设备端生成，根据需要确定是否将生成结果拷贝回主机端；

具体示例如下，首先创建生成器。

```cpp
ERROR_CHECK_CURAND(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_XORWOW)); // 主机端生成
ERROR_CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW)); // 设备端生成
```

设置`offset`、`ordering`、`seed`。

```cpp
// 设置偏移量
ERROR_CHECK_CURAND(curandSetGeneratorOffset(gen, 0ULL));
// 设置排序方式
ERROR_CHECK_CURAND(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST));
// 设置种子
ERROR_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
```

从指定分布中执行生成任务。

```cpp
// 以正态分布为例，此外还有均匀分布、对数正态分布和泊松分布
ERROR_CHECK_CURAND(curandGenerateNormal(gen, data, n, mean, stddev));
```

#### 设备端API

设备端API调用主要有以下几个步骤：

- 根据RNG算法创建状态；
- 初始化状态；
- 生成随机值；

每个支持的RNG算法都有对应的状态。

```cpp
curandStateXORWOW_t rand_state;
```

根据不同的RNG算法调用不同的`curand_init`重载。

```cpp
unsigned long long seed = threadIdx.x;
unsigned long long subsequence = 1ULL;
unsigned long long offset      = 0ULL;
curand_init(seed, subsequence, offset, &rand_state);
```

> 值得注意的是，由于线程的高度并发，所以应当避免在不同线程中使用相同的种子，也应当避免使用当前时间戳作为种子。
>
> 这里的`subsequence`会使得`curand_init()`返回的序列是调用了`(2^67 * subsequence + offset)`次`curand()`的结果。

最后生成随机值。

```cpp
float x = curand_normal(&rand_state);
```

> 详细代码参考[curand.cu](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/04_curand.cu)。

## OpenACC

### 基本概念

OpenACC是一个基于编译器指令的API，它的工作方式和OpenMP非常类似，都是使用`#pragma`开头的编译器指令作为指导。OpenACC的并行粒度分为`gang`、`worker`、`vector`，这些概念可以和CUDA一一对应。

| OpenACC  | CUDA                 |
| -------- | -------------------- |
| `gang`   | `block`              |
| `worker` | `warp`（未显式指出） |
| `vector` | `thread`             |

一个`gang`可以包含一个或多个执行线程，每个`gang`内部都包含一个或多个`worker`。每个`worker`都有一个向量宽度，由一个或多个同时执行相同指令的向量元素构成，简单理解就是一个`worker`可以包含一个或多个`vector`。每个`vector`都是一个单一的执行流，类似于CUDA线程。

OpenACC的目标是建立一个单线程的主机程序，该主机程序将核函数下放至多处理单元（PU），每个PU一次只运行一个`gang`，但可以同时执行多个独立的`worker`。在OpenACC中，`gang`并行使用多个PU，每个`gang`中的多线程并行即为`worker`并行。每个`worker`中的并行以及一个跨向量操作的并行称为`vector`并行。这里的PU有点类似于NVIDIA GPU中的SM。

### 并行模式

根据任务是否通过`gang`、`worker`、`vector`并行执行，OpenACC将执行分为几种模式。

假设一个OpenACC程序的并行计算区域创建了$G$个`gang`，每个`gang`包含$W$个`worker`，每个`worker`的向量宽度为`V`。此时总共有$G\times W\times V$个执行线程来处理这个并行区域。

#### `gang`冗余模式

开始执行并行区域时，`gang`以冗余模式执行，可以在并行执行前对`gang`的状态进行初始化。在该模式下，每个`gang`中只有一个活跃的`worker`和一个活跃的`vector`元素，其他`worker`和`vector`元素是闲置的，因此只有$G$个活跃执行线程。用CUDA伪核函数来表示如下。

```cpp
__global__ void kernel() {
    if (threadIdx.x == 0)
        foo();
}
```

#### `gang`分裂模式

在OpenACC并行区域的某些地方，程序可能通过`gang`转换为并行执行，这种情况下程序以`gang`分裂模式执行。该模式下每个`gang`中仍然只有一个活跃的`worker`和一个活跃的`vector`元素，因此同样只有$G$个活跃执行线程。但每个活跃的`vector`执行不同的并行区域，故计算任务被分散到各个`gang`中。以向量加法为例，CUDA伪核函数表达如下。

```cpp
__global__ void kernel(int* v1, int* v2, int* out, int N) {
    if (threadIdx.x == 0) {
        for (int i = blockIdx.x; i < N; i += gridDim.x) {
            out[i] = v1[i] + v2[i];
        }
    }
}
```

> 每个`gang`只有一个活跃`worker`时，程序处于单一`worker`模式，当`worker`中只有一个活跃`vector`时，程序处于单一`vector`模式。所以`gang`冗余模式和`gang`分裂模式也可以被称作单一`worker`模式和单一`vector`模式。

#### `worker`分裂模式

在该模式下，并行区域的工作被划分到多个`gang`的多个`worker`中，可以提供$G\times W$路并行。CUDA伪核函数表达如下。

```cpp
__global__ void kernel(int* v1, int* v2, int* out, int N) {
    if (threadIdx.x % warpSize == 0) {
        int warpId = threadIdx.x / warpSize;
        int warpsPerBlock = blockDim.x / warpSize;
        for (int i = blockIdx.x * warpsPerBlock + warpId; i < N; i += gridDim.x * warpsPerBlock) {
            out[i] = v1[i] + v2[i];
        }
    }
}
```

#### `vector`分裂模式

该模式将工作在`gang`、`worker`、`vector`通道上进行划分，提供$G\times W\times V$路并行。该模式最接近CUDA核函数的行为模式，CUDA伪核函数表达如下。

```cpp
__global__ void kernel(int* v1, int* v2, int* out, int N) {
    if (threadIdx.x < N)
        out[i] = v1[i] + v2[i];
}
```

### 基本用法

前面提到了OpenACC的工作方式与OpenMP的极为相似，在源代码中加入`#pragma acc`即可指导编译器对源代码进行翻译，使其能够在GPU上并行执行。

#### 计算指令

##### 核函数指令

`#pragma acc kernels`会自动分析代码块中的可并行循环。

```cpp
#pragma acc kernels
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}
```

核函数指令可以有条件子句来修饰，当条件为`false`时，代码块不会在设备上执行。

```cpp
#pragma acc kernels if(N > 128)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}
```

默认情况下，核函数指令结束时会有一个隐式同步，但可以通过添加`async`子句来使执行不被阻塞。

`async`子句接受一个可选的整型参数，若传入ID则可以使用指令来等待。

```cpp
#pragma acc kernels async(3)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

#pragma acc wait(3) // 或通过运行时API来等待 acc_async_wait(3)
// 或者使用空等待指令，等待所以异步任务完成
#pragma acc wait // 或通过运行时API来等待 acc_async_wait_all
```

也可以将`async`子句和`wait`子句结合起来，实现链式异步工作。

```cpp
#pragma acc kernels async(0)
{
    ...
}
#pragma acc kernels wait(0) async(1)
{
    ...
}
#pragma acc kernels wait(1) async(2)
{
    ...
}
#pragma acc wait(2)
```

而检查异步任务在没有阻塞的情况下是否完成只能通过运行时API来完成。

```cpp
acc_async_test(int); // 已结束返回非零值，否则返回零
```

> 目前想要编译OpenACC的代码，推荐使用PGI编译器。PGI被英伟达收购之后，编译器就纳入了NVIDIA HPC SDK中了。需要安装HPC SDK后使用`pgcc`命令来编译代码。详细代码参考[openacc_kernels.c](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/05_openacc_kernels.c)。

##### 并行指令

上面提到的核函数指令是一个强大的工具，编译器会自动分析代码并选择一个合适的并行策略，在这个过程中，程序员对程序的控制是较少的。但并行指令`#pragma acc parallel`则可以提供更多的控制选项。

并行指令同样支持核函数指令的一些子句，如`if`、`async`、`wait`。此外可以使用`num_gangs(int)`来设置`gang`数量，`num_workers(int)`设置`worker`数量，`vector_length(int)`设置每个`worker`的向量宽度。

```cpp
#pragma acc parallel num_gangs(32) num_workers(32) vector_length(64)
{
    ...
}
```

并行指令还支持`reduction`子句，格式为`#pragma acc parallel reduction(op:var1, var2, ...)`，支持的`op`有`+`、`*`、`max`、`min`、`&`、`|`、`^`、`&&`、`||`。

```cpp
#pragma acc parallel reduction(+ : result)
{
    for (int i = 0; i < N; i++)
    {
        result += A[i];
    }
}
```

并行指令还支持`private`和`firstprivate`子句。`private`会为每个`gang`创建一个`private`型复制变量，只有该`gang`可以使用该变量的拷贝，因此该值的改变对其他`gang`或主机程序是不可见的。`firstprivate`功能和`private`相同，只是会将每个`gang`中的`private`型变量的值初始化为主机上该变量当前的值。

```cpp
#pragma acc parallel private(a)
{
    ...
}

#pragma acc parallel firstprivate(a)
{
	...
}
```

> 详细代码参考[openacc_parallel.c](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/06_openacc_parallel.c)。

##### 循环指令

并行指令需要程序员为编译器明确标注并行性，并行区域总是以`gang`冗余模式开始的，执行并行模式之间的转换需要对编译器有明确的指示，这种指示可以通过循环指令`#pragma acc loop`来完成。

```cpp
#pragma acc parallel
{
#pragma acc loop
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
#pragma acc loop
    for (int i = 0; i < N; i++)
    {
        D[i] = C[i] * A[i];
    }
}
```

上面的代码并没有为循环指令添加子句，所以编译器可以自由使用它认为的最优循环调度。也可以通过`gang`、`worker`或`vector`子句来显式控制每一级的并行性。

```cpp
#pragma acc parallel
{
    int a = 1;

#pragma acc loop gang
    for (int i = 0; i < N; i++)
    {
        vec2[i] = a;
    }
}
```

上述代码中，并行区域以`gang`冗余模式开始，遇到带有`gang`子句的循环指令后，转换为了`gang`分裂模式。

考虑下列代码。

```cpp
#pragma acc parallel
{
#pragma acc loop
    for (int i = 0; i < N; i++)
    {
        ...
    }
}

#pragma acc kernels
{
#pragma acc loop
    for (int i = 0; i < N; i++)
    {
        ...
    }
}
```

可以简写为下列形式。

```cpp
#pragma acc parallel loop
for (int i = 0; i < N; i++)
{
    ...
}

#pragma acc kernels loop
for (int i = 0; i < N; i++)
{
    ...
}
```

> 详细代码参考[openacc_parallel.c](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/06_openacc_parallel.c)。

循环指令`loop`不仅可以和并行指令`paralle`结合，还可以与核函数指令`kernels`结合，但某些循环指令的子句在并行指令和核函数指令下会有所不同。

| 子句                | 并行指令下的行为                                             | 核函数指令下的行为                                           |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `collapse(int)`     | 指明循环指令适用于多重嵌套循环                               | 与并行指令下相同                                             |
| `gang(int)`         | 指明循环应通过`gang`划分到并行区域，`gang`数量由并行指令决定 | 说明循环应通过`gang`进行划分，`gang`有选择的使用整型参数     |
| `worker(int)`       | 指明循环应通过每个`gang`中的`worker`划分到并行区域，将每个`gang`由单一`worker`模式转换到`worker`分裂模式 | 说明循环应通过每个`gang`中的`worker`划分到并行区域，`worker`有选择的使用整型参数 |
| `vector(int)`       | 指明循环应通过`vector`通道进行分配，是一个`worker`由单一`vector`模式转换到`vector`分裂模式 | 说明循环应通过`vector`通道进行分配，`vector`有选择的使用整型参数 |
| `seq`               | 为了按序执行，使用`seq`对循环进行标记                        | 与并行指令下相同                                             |
| `auto`              | 指明编译器应为相关的循环选择`gang`、`worker`或`vector`并行   | 与并行指令下相同                                             |
| `tile(int, ...)`    | 指明编译器应将嵌套循环中的每个循环拆分为两个循环：外层的`tile`循环和内层的`element`循环。内层循环次数为`tile_size`，外层循环次数取决于串行代码。若附加到多个紧密的嵌套循环中，`tile`可以使用多个`tile_size`，并自动将所有外部循环放在内部循环之外 | 与并行指令下相同                                             |
| `device_type(type)` | `type`是一个逗号分隔的列表，分隔不同设备类型的子句。所有子句都遵循`device_type`的设定，只有当循环在指定设备类型上执行时，才可能有指令结束，或在下一个`type`的设备上执行的情况。 | 与并行指令下相同                                             |
| `independent`       | 该子句声称被标记的循环为并行的且编译器分析高于一切           | 与并行指令下相同                                             |

#### 数据指令

`#pragma acc data`可以在主机和设备之间进行显式的数据传输。

```cpp
#pragma acc data copyin(A[0 : N], B[0 : N]) copyout(C[0 : N], D[0 : N])
{
#pragma acc parallel
    {
#pragma acc loop
        for (int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }
#pragma acc loop
        for (int i = 0; i < N; i++)
        {
            D[i] = C[i] * A[i];
        }
    }
}
```

上面的代码告知编译器，只有`A`和`B`应该被拷贝到设备，只有`C`和`D`应该被拷贝回主机。同时指明了数组的范围，某些情况下，编译器能够推断要复制的数组大小，可以将代码简化为下面的样子。

```cpp
#pragma acc data copyin(A, B) copyout(C, D)
```

除了上述指定代码块的方法，还可以使用`enter data`指令和`exit data`指令来标记在任意节点传入和传出设备的数组。`enter data`指明的数据会持续保留在设备端，直到遇到将其传回的`exit data`指令。这两个指令可以与`async`和`wait`子句结合发挥最大作用。

> 注意，单纯的`data`指令不支持`async`和`wait`子句。

```cpp
    init(vec1);
    init(vec2);

#pragma acc enter data copyin(vec1[0 : N], vec2[0 : N]) async(0)

	process(vec3);

#pragma acc kernels wait(0) async(1)
    {
        for (int i = 0; i < N; i++)
        {
            vec1[i] = do_something(vec2[i]);
        }
    }

#pragma acc exit data copyout(vec1[0 : N]) wait(1) async(2)

    process(vec4);

#pragma acc wait(2)
```

考虑如下代码。

```cpp
#pragma acc data copyin(A[0 : N], B[0 : N]) copyout(C[0 : N], D[0 : N])
{
#pragma acc parallel
    {
		...
    }
}
```

可以简写为下列形式。

```cpp
#pragma acc parallel copyin(A[0 : N], B[0 : N]) copyout(C[0 : N], D[0 : N])
{
    ...
}
```

> 详细代码参考[openacc_data.c](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/07_openacc_data.c)。

数据指令支持的子句见下表。

| 子句                            | 行为                                                         | `data`支持 | `enter data`支持 | `exit data`支持 |
| ------------------------------- | ------------------------------------------------------------ | ---------- | ---------------- | --------------- |
| `if(cond)`                      | 若`cond`为`true`则执行数据搬移                               | Y          | Y                | Y               |
| `copy(var1, ...)`               | 在进入数据区域时将变量拷贝至设备端，离开数据区域时拷贝回主机端 | Y          | N                | N               |
| `copyin(var1, ...)`             | 指明变量只能被拷贝至设备端                                   | Y          | Y                | N               |
| `copyout(var1, ...)`            | 指明变量只能被拷贝回主机端                                   | Y          | N                | Y               |
| `create(var1, ...)`             | 指明列出的变量需要在设备端分配内存，但变量值不必传入或传出设备 | Y          | Y                | N               |
| `present(var1, ...)`            | 指明列出的变量已经在设备端了，不必再次传入。运行时，编译器会发现并使用这些已经存在于设备端的数据 | Y          | N                | N               |
| `present_or_copy(var1,...)`     | 若列出的变量已经在设备端了，则功能与`present`一致；若不在设备端，则功能与`copy`一致 | Y          | N                | N               |
| `present_or_copyin(var1, ...)`  | 若列出的变量已经在设备端了，则功能与`present`一致；若不在设备端，则功能与`copyin`一致 | Y          | Y                | N               |
| `present_or_copyout(var1, ...)` | 若列出的变量已经在设备端了，则功能与`present`一致；若不在设备端，则功能与`copyout`一致 | Y          | N                | N               |
| `present_or_create(var1, ...)`  | 若列出的变量已经在设备端了，则功能与`present`一致；若不在设备端，则功能与`create`一致 | Y          | Y                | N               |
| `deviceptr(var1, ...)`          | 指明列出的变量是设备内存指针，不必再为该指针指向的数据分配空间，也不必在主机和设备之间传输。 | Y          | N                | N               |
| `delete(var1, ...)`             | 可以与`exit data`结合使用，显式释放设备内存                  | N          | N                | Y               |

> 更多指令和运行时API可以参考官方文档[Specification | OpenACC](https://www.openacc.org/specification)。

### 与CUDA结合

要结合CUDA与OpenACC，需要通过`deviceptr`子句来实现CUDA和OpenACC之间的数据共享。核心代码如下。

```cpp
#pragma acc parallel loop gang deviceptr(d_A, d_B, d_C)
for (int i = 0; i < M; i++)
{
#pragma acc loop worker vector
    for (int j = 0; j < P; j++)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += d_A[i * N + k] * d_B[k * P + j];
        }
        d_C[i * P + j] = sum;
    }
}
```

> 详细代码参考[cuda_openacc.cu](https://github.com/Deleter-D/CUDA/blob/master/07_acceleration_library_and_OpenACC/08_cuda_openacc.cu)。
