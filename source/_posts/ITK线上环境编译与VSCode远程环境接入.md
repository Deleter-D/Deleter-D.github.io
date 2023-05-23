---
title: ITK线上环境编译与VSCode远程环境接入
toc: true
mathjax: true
tags:
  - ITK
  - VS Code远程开发
  - 毕昇编译器
categories: 项目
abbrlink: 51688
date: 2023-04-17 20:32:05
---

最近学校的一个项目需要在服务器上编译安装ITK，顺便记录以下VS Code接入远程环境的配置过程。

<!-- more -->

# ITK编译与安装

## 前期准备

首先将源码压缩包上传至线上环境，解压源码。

```shell
tar -xvf InsightToolkit-5.2.1.tar.gz
```

创建一个文件夹用于存放Cmake构建出来的文件。

```shell
mkdir ITK-build
```

再创建一个文件夹用于ITK的安装目标路径。

```shell
mkdir ITK-install
```

截止目前，文件结构应该为如下所示。

```shell
|--InsightToolkit-5.2.1
|--InsightToolkit-5.2.1.tar.gz
|--ITK-build
|--ITK-install
```

上述四个文件及文件夹应处在同一层级下。

## Cmake配置

进入`ITK-build`文件夹，并执行cmake命令。

```shell
cd ITK-build
cmake ../InsightToolkit-5.2.1
```

按下回车后cmake会开始进行配置，耐心等待配置完成。

完成后会发现当前文件夹下构建出来很多东西，此时需要执行ccmake进行配置。

```shell
ccmake ../InsightToolkit-5.2.1
```

按下回车后会进入ccmake的TUI界面。

![](https://user-images.githubusercontent.com/56388518/232485666-c6668030-9d61-4caf-a54e-db35624dcfa3.png)

此时按`t`进入高级模式。

![](https://user-images.githubusercontent.com/56388518/232485681-b9dda49e-940a-45ff-add0-07a74c028de7.png)

### 第一次配置

首先要配置的是编译器，更改如下两项为图中所示的路径。

![](https://user-images.githubusercontent.com/56388518/232485692-af5c88e6-253a-4a8a-abd6-333dd065b240.png)

此时按下`c`开始配置，耐心等待配置完成。

> 目前由于线上环境的cmake和ccmake版本不匹配，ccmake版本过低导致无法配置。
>
> 解决方案：正常按下c即可，ccmake会提示版本过低，此时按下e退出提示界面，再按下q退出ccmake的TUI界面，再去执行cmake命令。
>
> ```shell
> cmake ../InsightToolkit-5.2.1
> ```
>
> 在ccmake中的配置，会由于按下了c而得以保存，故执行cmake的时候会沿用在ccmake中更改后的配置项。
>
> 后面所有关于cmake的配置都需要如此执行，以下不再赘述。

### 第二次配置

第一次配置完成后再次执行ccmake，并按下`t`进入高级模式。

```shell
ccmake ../InsightToolkit-5.2.1
```

打开编译动态链接库的选项。

![](https://user-images.githubusercontent.com/56388518/232485701-07cc07a2-96cd-41b4-8994-63c15a86bc1f.png)

按下Page Down翻到下一页，或者使用方向键下键也可以翻页，改动如图。

![](https://user-images.githubusercontent.com/56388518/232485725-904f8550-d199-46db-b19b-8077cd1606d8.png)

这里的`CMAKE_INSTALL_PREFIX`指定到前期准备过程中创建的ITK安装目标路径中。

再次按下`c`进行配置。

### 第三次配置（视情况而定）

再次执行ccmake命令，按`t`进入高级模式，检查所有之前改动过的配置项。

如果发现所有的配置项都是已经更改过的状态，就可以按`q`退出，不进行配置。

如果发现有些配置被恢复为了默认状态，则需要再次更改后再进行配置，知道所有配置项都符合要求。

## Make构建

配置完成后，直接在`ITK-build`目录下执行`make`命令。

```shell
make -j8
```

加上参数`-jN`开启多线程以加速编译

## 安装

make构建结束后，在同一目录下执行安装命令。

```shell
make install
```

到此ITK就安装完成了。

安装完成后，需要将ITK的链接库文件导入环境变量中。

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/bisheng_tester2/Example/ITK-install/lib
```

执行该语句时，切记不要忘记加上`$LD_LIBRARY_PATH`，否则会将当前的环境变量覆盖掉，导致线上环境出现问题。

# VS Code接入远程环境

## 前期准备

下载Remote - SSH插件。

![](https://user-images.githubusercontent.com/56388518/232485745-2ebc937d-5163-4110-a3f9-9ed491217e44.png)

配置文件如下。

```
# Read more about SSH config files: https://linux.die.net/man/5/ssh_config
Host BiSheng
    HostName 10.8.3.1
    User bisheng_tester2
    Port 22
    IdentityFile "...\id_rsa"
```

最后一项`IdentityFile`指向配置Open VPN时用到的私钥。

提前在线上环境创建一个放置代码的文件夹，在VS Code中打开远程文件夹。

## 配置

### `c_cpp_properties.json`

按下`Ctrl + Shift + P`，在弹出的窗口中选择C/C++：编辑配置(UI)。

![](https://user-images.githubusercontent.com/56388518/232485750-b237c4a1-dc48-4e5a-a51a-a71cfd454c28.png)

选择编译器路径。

![](https://user-images.githubusercontent.com/56388518/232485757-47149fbd-c492-4a5c-8f9e-203e90bb3097.png)

往下翻页，找到包含路径配置项，配置到与图中一致。

![](https://user-images.githubusercontent.com/56388518/232485767-e1e5090f-8154-477b-8131-3473fb10cbc0.png)

**该路径请指向自己安装的ITK目录。**

此时在文件夹中生成了一个`.vscode`文件夹，其中有一个`c_cpp_properties.json`配置文件。

![](https://user-images.githubusercontent.com/56388518/232485782-ce75c096-5eef-4bea-86ea-434490b7bfc7.png)

内容与刚才的配置对应。

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/data/bisheng_tester2/Example/ITK-install/include/**"
            ],
            "defines": [],
            "compilerPath": "/home/bisheng_tester2/Ascend/ascend-toolkit/latest/aarch64-linux/bisheng_cpp/bin/clang++",
            "cStandard": "c17",
            "cppStandard": "c++14",
            "intelliSenseMode": "linux-clang-arm64"
        }
    ],
    "version": 4
}
```

### `tasks.json`

任意新建一个cpp文件并打开。

点击菜单栏`终端 -> 配置任务`，在弹出的窗口中选择clang++

![](https://user-images.githubusercontent.com/56388518/232485787-029c2a1b-adcd-4211-a217-f99db8045205.png)

此时在`.vscode`文件夹中会生成一个`tasks.json`配置文件，修改编译器的参数如下所示。

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"type": "cppbuild",
			"label": "C/C++: clang++ 生成活动文件",
			"command": "/home/bisheng_tester2/Ascend/ascend-toolkit/latest/aarch64-linux/bisheng_cpp/bin/clang++",
			"args": [
				"-fsycl",  // 添加
				"-fsycl-targets=ascend_910-cce",  // 添加
				"-fcolor-diagnostics",
				"-fansi-escape-codes",
				"-g",
				"-gdwarf",  // 添加
				"${file}",
				"-o",
				"${fileDirname}/${fileBasenameNoExtension}",
				"-I",  // 添加，指定包含文件路径
				"/data/bisheng_tester2/Example/ITK-install/include/ITK-5.2",  // 添加
				"-L",  // 添加，指定链接库路径
				"/data/bisheng_tester2/Example/ITK-install/lib",  // 添加
				"-lxxx"  // 添加，指定具体要链接的库文件，xxx代表链接库名称，需要依据实际情况更改
			],
			"options": {
				"cwd": "${fileDirname}"
			},
			"problemMatcher": [
				"$gcc"
			],
			"group": "build",
			"detail": "编译器: /home/bisheng_tester2/Ascend/ascend-toolkit/latest/aarch64-linux/bisheng_cpp/bin/clang++"
		}
	]
}
```

### `launch.json`

点击菜单栏`运行 -> 添加配置`，此时会在`.vscode`文件夹下生成一个`launch.json`，点击右下角的添加配置。

![](https://user-images.githubusercontent.com/56388518/232485794-f13ad526-b891-4888-bc32-33f1efcf6962.png)

选择(gdb) 启动。

![](https://user-images.githubusercontent.com/56388518/232485802-e11792de-d107-456f-97b0-6c05a3bf37fa.png)

修改配置文件如下。

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) 启动",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/${fileBasenameNoExtension}", // 主要修改此处
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "为 gdb 启用整齐打印",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "将反汇编风格设置为 Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

到此VS Code的配置就完成了。

## Demo

在刚才创建的cpp文件中，写一个小Demo。

```c++
#include <iostream>
#include <vnl/vnl_matrix.h>  // itk中的一个第三方库

int main()
{
    vnl_matrix<int> A(100, 100, 1);  // 实例化一个100*100的矩阵，元素均为1

    std::cout << A << std::endl;  // 输出该矩阵

    return 0;
}
```

点击菜单栏`终端 -> 运行生成文件`，或直接按快捷键`Ctrl + Shift + B`进行编译。终端输出如下。

![](https://user-images.githubusercontent.com/56388518/232485815-66c535e6-17e3-4f4c-91ad-f3fa271e45c7.png)

此时点击右上角的调试或运行，在弹出的窗口中选择刚才配置的`launch`任务，即可执行该Demo。

![](https://user-images.githubusercontent.com/56388518/232485822-f24d789b-a918-4906-be38-06a4104405c1.png)

如果提示如下报错信息。

```shell
error while loading shared libraries: libitkvnl-5.2.so.1: cannot open shared object file: No such file or directory
```

则需要在VS Code的终端中将ITK的链接库路径添加至环境变量中，参考ITK安装步骤的最后一步。

Demo执行结果如下。

![](https://user-images.githubusercontent.com/56388518/232485832-5fe0a4f6-6a3a-427f-a3a9-d24bbdd6533d.png)
