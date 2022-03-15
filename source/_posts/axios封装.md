---
title: axios封装
tags:
  - axios
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F5ddd9a0db56a3113a54a45cb8f311d9c2089812e.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844258&t=853be73f0faddbb966af18c46e86052e
abbrlink: 30201
date: 2022-03-14 18:00:45
---

## 创建js文件

创建一个js文件作为封装配置，引入axios

```js
import axios from "axios";
```

创建一个axios实例进行自定义配置

```js
const instance = axios.create({
    baseURL: 'http://xxx.xxx',
    timeout: 5000,
})
```

## 各类请求封装

封装get

```js
export function get(url, params) {
    return instance.get(url, {
        params
    })
}
```

封装post

```js
export function post(url, params) {
    return instance.post(url, params, {
        transformRequest: [
            function (data) {
                let str = '';
                for (let key in data) {
                    str += encodeURIComponent(key) + '=' + encodeURIComponent(data[key]) + '&';
                }
                return str;
            }
        ],
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        }
    })
}
```

封装delete

```js
export function del(url) {
    return instance.delete(url)
}
```

## 引用方式

在别的文件中引用

```js
import {方法名} from '封装js文件路径'
```
