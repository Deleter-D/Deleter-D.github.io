---
title: package.json配置文件详解
tags:
  - package.json
  - 配置文件
categories: [前端,nodejs]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991363-6b870c0a-7963-4c9a-b4da-51c65309efdd.png
abbrlink: 49919
date: 2022-03-14 17:56:23
---

有关package.json配置文件的简单解释，方便理解配置文件所配置的内容。

<!--more-->

scripts指定运行的脚本

```json
"scripts":{
    "test1":"命令1",
    "test2":"命令2"
    # 使用npm run test1 即可运行命令1
}
```

dependencies中指定运行时依赖

```json
"dependencies":{
    "jquery":"^3.5.1", # ^表示每次运行npm install时会更新后两位版本号
    "bootstrap":"~4.5.3", # ~表示每次运行npm install时会更新最后一位版本号
    "layui":"2.0.3" # 不加符号则固定为这个版本
}
```
