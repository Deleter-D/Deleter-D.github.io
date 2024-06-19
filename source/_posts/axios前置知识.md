---
title: axios前置知识
tags:
  - axios
  - promise
  - RESTFul
categories: [Java,前端,axios]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991448-54f2eff8-11e0-404e-b1f7-7104725bcb52.png
abbrlink: 4623
date: 2022-03-14 17:57:39
---

学习axios必须的前置知识，包括RESTFul API规范、Promise的基本用法。

<!--more-->

## RESTFul API规范

GET (SELECT)：从服务器取出资源（一个或多个）

POST (CREATE)：在服务器新建一个资源

PUT (UPDATE)：在服务器更新资源（客户端提供改变后的完整资源）

PATCH (UPDATE)：在服务器更新资源（客户端提供改变的属性）

DELETE (DELETE)：从服务器删除资源

## Promise

#### 基本原理

主要用于异步计算，可以将异步操作队列化，按照期望的顺序执行，返回符合预期的结果，可以在对象之间传递和操作promise，辅助我们处理队列

可以解决回调地狱的问题

#### 基本语法

可以嵌套很多层

```js
new Promise((resolve,reject)=>{ //第一层的Promise
    console.log("");
    if(statement){
        resolve("success")//成功会调用这个
    }else{
        reject("fail")//失败会调用这个
    }
}).then(res=>{//成功则进入这里
    console.log(res);
    return new Promise((resolve,reject)=>{//第二层的Promise
        resolve("success")
    })
},err=>{//失败则进入这个
    console.log(err);
})
```

并发请求

```js
Promise.all([
    new Promise((resolve,reject)=>{
        resolve('first request')
    }),
    new Promise((resolve,reject)=>{
        resolve('second request')
    })
]).then(res=>{
    console.log(res);
});
```
