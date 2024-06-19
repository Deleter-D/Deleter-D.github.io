---
title: axios封装
tags:
  - axios
categories: [Java,前端,axios]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991763-81023062-d085-4994-9c55-56e5e3d094ae.png
abbrlink: 30201
date: 2022-03-14 18:00:45
---

封装自定义axios的方法。

<!--more-->

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
