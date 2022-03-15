---
title: axios全局配置
tags:
  - axios
  - 配置文件
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F5f19b48f172a28685271b29d88accd468861c037.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844225&t=2b6f98a3b9c2e90d35fee05c8958ff7fs
abbrlink: 65179
date: 2022-03-14 17:59:24
---

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
