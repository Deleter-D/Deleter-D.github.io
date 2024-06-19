---
title: axios全局配置
tags:
  - axios
  - 配置文件
categories: [Java,前端,axios]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991589-a181c95d-a206-4c02-b353-ab14c5f1846b.png
abbrlink: 65179
date: 2022-03-14 17:59:24
---

axios的两种配置方式，全局配置与按实例配置。

<!--more-->

## 全局配置

将配置写入一个js文件中，在需要的地方引入该js文件即可

```js
axios.defaults.baseURL="http://127.0.0.1";
axios.defaults.timeout=5000;
axios.defaults.headers.post['content-type']='application/x-www-form-urlencoded';
```

应用全局配置后url直接在baseURL基础上拼接即可

```js
axios.get('xxx/xxx?id=1').then(res=>{
    console.log(res);
}).catch(err=>{
    console.log(err);
})
```

## 按实例进行配置

若与全局配置共存，则优先使用实例配置

```js
let request1 = axios.create({
    baseURL:"http://www.xxx.com/xxx"
    timeout:5000
})


let request2 = axios.create({
    baseURL:"http://localhost/xxx"
    timeout:3000
})
```

使用方式

```js
request1.get('xxx/xxx?id=1').then(res=>{
    console.log(res);
}).catch(err=>{
    console.log(err);
})


request2.get('xxx/xxx?id=1').then(res=>{
    console.log(res);
}).catch(err=>{
    console.log(err);
})
```
