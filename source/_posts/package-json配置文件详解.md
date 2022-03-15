---
title: package.json配置文件详解
tags:
  - package.json
  - 配置文件
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2Fd8a2ee48505f2ac7445868079a2be54a1ab89529.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844158&t=4f2475995ff5dd8e422532214bdb7f28
abbrlink: 49919
date: 2022-03-14 17:56:23
---

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
