---
title: npm常用命令
tags:
  - npm
  - node.js
categories: [前端,nodejs]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991269-25bd57bf-12fb-41b1-bbde-1c6b56d3983d.png
abbrlink: 34123
date: 2022-03-14 17:54:57
---

npm换源命令，安装、更新模块命令等常用命令

<!--more-->

## 换源

换源命令

```shell
npm config set registry https://registry.npm.taobao.org
```

检查是否成功

```shell
npm config get registry
```

## 常用命令

安装

```shell
npm install <Module Name>
```

全局安装

```shell
npm install <Module Name> -g
```

查看所有全局安装的模块

```shell
npm list -g
```

查看某个模块的版本号

```shell
npm list <Module Name>
```

更新模块版本

```shell
npm install <Module Name>@<version> -g
```

在package文件的dependencies节点写入依赖，即运行时依赖

```shell
npm install -save <Module Name>
npm i <Module Name> -S # 简写
```

在package文件的devDependencies节点写入依赖，即开发时依赖

```shell
npm install -save-dev <Module Name>
npm i <Module Name> -D # 简写
```
