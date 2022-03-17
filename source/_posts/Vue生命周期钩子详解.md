---
title: Vue生命周期钩子详解
tags:
  - Vue
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2Fefe0ab1540ebbf56916339c304dfc18cff304403.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650125391&t=81c961cfc8d9b938f96db666d90ccafc
abbrlink: 27624
date: 2022-03-18 00:07:56
---

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
