---
title: Vue-router路由（下）
tags:
  - Vue
  - Vue-Router
categories: [Java,前端,Vue]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193992311-d603f3db-8eed-4298-9cc3-cbc801d21b08.png
abbrlink: 52120
date: 2022-03-18 00:13:10
---

Vue生态中的一员大将Vue-router，负责管理页面路由，书接上回，本节介绍路由重定向、别名及导航守卫。

<!--more-->

## 重定向

重定向在routes配置中完成，要从/a重定向到/b

```js
const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/home',
        redirect: '/', // 访问/home时重定向到根
        component: Home
    }
]
```

也可以使用对象的形式

```js
const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/home',
        redirect: {name:'HomeRoot'}, // 访问/home时重定向到根
        component: Home
    }
]
```

还支持将params类型的参数传递转换为query类型的，但一般不这样做

```js
const routes = [
    {
        path: 'page/:id',
        redirect: to => {
            return {path: '/about/article', query: {name: 'li', age: to.params.id}}
        },
        component: () => import('../views/Page')
    },
]
```

## 别名

```js
const routes = [
    {
        path: '/about',
        name: 'About',
        alias: '/a', // 可以通过/a访问到/about       
        component: () => import('../views/About.vue'), 
    }
]
```

可以起多个别名，以数组的形式

```js
const routes = [
    {
        path: '/about',
        name: 'About',
        alias: ['/a', '/b', '/c'], // 可以通过/a访问到/about       
        component: () => import('../views/About.vue'), 
    }
]
```

若路由中有params类型的参数传递，起别名时也要带上参数

```js
const routes = [
    {
        path: 'page/:id',
        alias: 'p/:id',
        component: () => import('../views/Page')
    },
]
```

## 导航守卫

导航守卫主要用来通过跳转或取消的方式守卫导航

有多种机会植入路由导航过程中：

1. 全局导航守卫：在index.js中添加

   前置守卫

   ```js
   router.beforeEach((to, from) => {
       // 处理
       return true;
   })
   ```

   后置钩子

   ```js
   router.afterEach((to, from) => {
       // 处理
   })
   ```

2. 路由独享守卫：在单个路由下添加

   ```js
   const routes = [
     {
       path: '/users/:id',
       component: UserDetails,
       beforeEnter: (to, from) => {
         // reject the navigation
         return false
       },
     },
   ]
   ```

3. 组件内的守卫

   ```js
   const UserDetails = {
     template: `...`,
     beforeRouteEnter(to, from) {
       // 在渲染该组件的对应路由被验证前调用
       // 不能获取组件实例 `this` ！
       // 因为当守卫执行时，组件实例还没被创建！
     },
     beforeRouteUpdate(to, from) {
       // 在当前路由改变，但是该组件被复用时调用
       // 举例来说，对于一个带有动态参数的路径 `/users/:id`，在 `/users/1` 和 `/users/2` 之间跳转的时候，
       // 由于会渲染同样的 `UserDetails` 组件，因此组件实例会被复用。而这个钩子就会在这个情况下被调用。
       // 因为在这种情况发生的时候，组件已经挂载好了，导航守卫可以访问组件实例 `this`
     },
     beforeRouteLeave(to, from) {
       // 在导航离开渲染该组件的对应路由时调用
       // 与 `beforeRouteUpdate` 一样，它可以访问组件实例 `this`
     },
   }
   ```

## `<keep-alive>`与路由组合

在Vue3.x中不再支持如下形式

```html
<keep-alive>
    <router-view/>
</keep-alive>
```

而应该采用如下方式

```html
<router-view v-slot="{ Component }">
  <transition>
    <keep-alive>
      <component :is="Component"/>
    </keep-alive>
  </transition>
</router-view>
```

`<keep-alive>`有两个可选属性

`include`选择缓存的页面，可填字符串或正则表达式

`exclude`选择不缓存的页面，可填字符串或正则表达式
