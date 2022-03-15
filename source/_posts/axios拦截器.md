---
title: axios拦截器
tags:
  - axios
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F210ee9c757219c6b5a819fc8279f0d5a034bfba0.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844243&t=ca67910d4bbbcb1ea2cdea08f6b82b13
abbrlink: 4512
date: 2022-03-14 18:00:16
---

## 请求拦截器

```js
// 可以为配置好的实例单独配置拦截器
axios.interceptors.request.use(config=>{
    console.log("请求拦截成功，处理，放行");
    return config;
},err=>{
    console.log(err);
});
```

## 响应拦截器

```js
// 可以为配置好的实例单独配置拦截器
axios.interceptors.response.use(config=>{
    console.log("响应拦截成功，处理，放行");
    return config;
},err=>{
    console.log(err);
});
```
