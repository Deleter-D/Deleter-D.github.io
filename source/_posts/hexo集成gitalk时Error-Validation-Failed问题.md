---
title: 'hexo集成gitalk时Error: Validation Failed问题'
tags:
  - Hexo
  - Gitalk
categories: Hexo
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991837-c0116f55-3ac1-4173-97ec-e63a584c66f8.png
abbrlink: 64660
date: 2022-03-14 19:08:09
---

Hexo集成Gitalk后，某些文章下方的评论显示`Error: Validation Failed`的原因及解决方案。

<!--more-->

## Hexo集成Gitalk的问题

集成Gitalk后进入博客发现有些页面下方的评论显示Error: Validation Failed.

原因：Gitalk会限制Label name的长度，有些文章生成的URL长度会超过限制，所以导致这个问题

## 解决方案

可以集成一个对文章生成唯一id的插件

### hexo-abbrlink

在博客根目录下安装

```shell
npm install --save hexo-abbrlink
```

并修改配置文件`_config.yml`

```shell
permalink: [EveryWordsYouWant]/:abbrlink/
```

生成的链接类似于：

```
https://deleter-d.github.io/[EveryWordsYouWant]/30201/
```

### hexo-uuid

这个插件也是同理的

在博客根目录下安装

```shell
npm install --save hexo-uuid
```

并修改配置文件`_config.yml`

```shell
permalink: [EveryWordsYouWant]/:uuid/
```

