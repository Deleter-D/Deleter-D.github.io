---
title: axios基础
tags:
  - axios
categories: 前端
cover: https://user-images.githubusercontent.com/56388518/193991506-22eabcad-4648-47d5-aa9b-a928df18297e.png
abbrlink: 63440
date: 2022-03-14 17:58:40
---

## 安装axios

```shell
npm i axios -S
```

## 基础语法

默认使用get请求

```js
axios('http://localhost/xxx/xxx')
    .then(res=>{
    console.log(res);
});
```

传参

get方法

```js
axios({
    method:'get', // 指定请求方法
    url:'http://localhost/xxx/xxx',
    params:{
        username:'zhangsan',
        age:10,
        gender:'male'
    }
}).then(res=>{
    console.log(res);
});
```

post方法

```js
axios({
    method:'post', // 指定请求方法
    url:'http://localhost/xxx/xxx',
    headers:{ // 使用post请求必须改变content-type,否则将依然用url拼接方式传参
        'content-type':'application/x-www-form-urlencoded'
    }
    data:{ // 并且数据要用data传,而不能用params
        username:'zhangsan',
        age:10,
        gender:'male'
    }
}).then(res=>{
    console.log(res);
});
```

## 实际应用语法

get请求

```js
axios.get('http://localhost/xxx?username=zhangsan').then(res=>{
    console.log(res);
});
// 另一种传参方式
axios.get('http://localhost/xxx',config:{params:{username='zhangsan'}}).then(res=>{
    console.log(res);
});
```

post请求

```js
axios.post('http://localhost/xxx',data:"username=zhangsan").then(res=>{
    console.log(res);
});
```

## 并发请求

结果存入数组

```js
axios.all([
    axios.get('http://localhost/xxx?id=1'),
    axios.get('http://localhost/xxx?id=2'),
    axios.get('http://localhost/xxx?id=3'),
]).then(res=>{
    console.log(res[0]);
    console.log(res[1]);
    console.log(res[2]);
}).catch(err=>{
    console.log(err);
});
```

结果单独接收

```js
axios.all([
    axios.get(url:'http://localhost/xxx?id=1'),
    axios.get(url:'http://localhost/xxx?id=2'),
    axios.get(url:'http://localhost/xxx?id=3'),
]).then(
    axios.spread((res1,res2,res3)=>{
        console.log(res1);
        console.log(res2);
        console.log(res3);
    })    
).catch(err=>{
    console.log(err);
});
```
