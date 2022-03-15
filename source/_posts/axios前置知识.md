---
title: axios前置知识
tags:
  - axios
  - promise
  - RESTFul
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F9aff9d61ec0a5ebf7838dffcfb4c8ab73e33ae1f.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649844179&t=03edac0474f15bb73be0b501395b7d9as
abbrlink: 4623
date: 2022-03-14 17:57:39
---

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
