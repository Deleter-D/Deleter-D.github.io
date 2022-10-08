---
title: axios拦截器
tags:
  - axios
categories: [前端,axios]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991654-a9e2f427-cb5f-49b7-8859-f2c2f32889c0.png
abbrlink: 4512
date: 2022-03-14 18:00:16
---

axios的请求拦截器和响应拦截器。

<!--more-->

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
