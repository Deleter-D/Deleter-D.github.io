---
title: hexo集成LaTex数学公式解决方案
tags:
  - Hexo
  - LaTex
categories: Hexo
toc: true
cover: >-
  https://user-images.githubusercontent.com/56388518/194000349-474ca0eb-f3b8-4e5b-96ce-87d785831919.png
abbrlink: 47466
date: 2022-10-05 14:51:38
---

hexo集成LaTex数学公式所需要的依赖，还有一些注意事项。

<!--more-->

# hexo集成LaTex数学公式解决方案

在hexo博客根目录下打开Git Bash

## 步骤一

卸载默认渲染器

```shell
npm uninstall hexo-renderer-marked
```

卸载hexo-math（如果之前没有安装可以跳过这步）

```shell
npm uninstall hexo-math
```

## 步骤二

安装pandoc

```shell
npm install hexo-renderer-pandoc --save
```

利用npm安装完成后还需要在电脑本地安装pandoc

官方网站：:link:[Pandoc](https://pandoc.org/index.html)

**安装完成后记得重启电脑**，否则pandoc会在执行`hexo g`的时候报错

## 步骤三

安装mathjax

```shell
npm install hexo-filter-mathjax --save
```

## 步骤四

在想要渲染LaTex公式的md文件头部加入配置项：`mathjax: true`即可

LaTex公式语法参考：:link:[LaTex公式指导手册](https://www.zybuluo.com/codeep/note/163962)
