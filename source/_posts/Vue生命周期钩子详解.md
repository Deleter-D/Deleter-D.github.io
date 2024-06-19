---
title: Vue生命周期钩子详解
tags:
  - Vue
categories: [Java,前端,Vue]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193992164-23bf98cc-5c31-4a37-a916-0121cf40b31e.png
abbrlink: 27624
date: 2022-03-18 00:07:56
---

Vue页面的生命周期及特殊方法`$nextTick`。

<!--more-->

## 生命周期

```js
  beforeCreate() {
    console.log("创建实例之前");
  },
  created() {
    console.log("实例创建完成");
  },
  beforeMount() {
    console.log("模板编译之前");
  },
  mounted() {
    console.log("模板编译完成");
  },
  beforeUpdate() {
    console.log("数据更新之前");
  },
  updated() {
    console.log("模板内容更新完成");
  },
  beforeUnmount() {
    console.log("实例销毁之前");
  },
  unmounted() {
    console.log("实例销毁完成");
  }
```

用`<keep-alive>`标签使组件缓存

```html
<keep-alive>
  <MyConn></MyConn>
</keep-alive>
```

与之相关的两个生命周期函数

```js
  activated() {
    console.log("缓存的组件激活时");
  },
  deactivated() {
    console.log("缓存的组件停用时");
  }
```

## 特殊方法`$nextTick`

该函数将回调延迟到下次DOM更新循环之后执行，可用在任何option及生命周期钩子中

```js
activated() {   
    this.$nextTick(() => {
      this.$refs.username.focus();
    })
  },
```

在修改数据之后立即使用它，然后等待DOM更新
