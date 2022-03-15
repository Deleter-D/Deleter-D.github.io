---
title: npm常用命令
tags:
  - npm
  - node.js
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2Fa6824d8dfdde8731187cb90d639ba72b1625b32e.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844124&t=519509d871e238b98b5be6df83639892
abbrlink: 34123
date: 2022-03-14 17:54:57
---

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
