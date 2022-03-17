---
title: Vue-router路由（上）
tags:
  - Vue
  - Vue-Router
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2F736b3d2c3ee9832291342e0c2c800a5f9d1c39a1.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650125576&t=4d8d05d4ea1715ee8317f07cc62bdf2a
abbrlink: 14233
date: 2022-03-18 00:11:10
---

## 路由

创建index.js，引入createRouter和createWebHistory方法

```js
import { createRouter, createWebHistory } from 'vue-router'
```

声明一个routes变量

```js
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    // 这种写法需要在上面import该组件
    // 使用这种方式引入，在打包时会将所有页面打包到同一个js中
    component: Home
  },
  {
    path: '/about',
    name: 'About',
    // 这种写法是在该处直接import，故上面不需要引入
    // 使用这种方式引入，可将各页面打包成多个chunk，可以实现懒加载，尽量使用该方法
    component: () => import('../views/About.vue')
  }
]
```

创建router实例

```js
const router = createRouter({
  history: createWebHistory(process.env.BASE_URL), // 历史模式
  routes // 相当于routes:routes，JSON的简化写法
})
```

暴露接口

```js
export default router
```

在main.js中引入

```js
import router from './router'

createApp(App).use(router).mount('#app') // 利用use将router作为参数传入
```

在页面中利用`<router-link>`标签创建路由，会自动渲染成`<a>`标签

```html
<!--to指明路径-->
<router-link to="/">Home</router-link>
<!--激活的页面将填充到如下标签中-->
<router-view/>
```

激活路由的样式，默认为该样式

```css
#nav a.router-link-exact-active {
  color: #42b983;
}
```

可通过`active-class`属性改变

```html
<router-link active-class="active" to="/">Home</router-link>
```

## 路由模式

`Hash`：vue-router默认使用该模式，该模式使用URL的hash值来作为路由，用来指导浏览器动作，对服务端完全无用，支持所有浏览器

引入

```js
import { createRouter, createWebHashHistory } from 'vue-router'

const router = createRouter({
  history: createWebHashHistory(process.env.BASE_URL), // 历史模式
  routes // 相当于routes:routes，JSON的简化写法
})
```

`History`：创建一个 HTML5 历史，即单页面应用程序中最常见的历史记录。应用程序必须通过 http 协议被提供服务。

`Abstract`：支持所有JavaScript运行模式，若发现没有浏览器的API，路由会自动强制进入这个模式

## 嵌套路由

在路由的index.js中

```js
const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/about',
        name: 'About',
        component: () => import('../views/About.vue'),
        children: [ // 利用children字段声明子路由
            {
                // 当进入about页面时子路径为空，会匹配到该处，从而实现默认显示
                path: '', 
                component: () => import('../views/Order')
            },
            {
                path: 'order',
                name: 'Order',
                component: () => import('../views/Order')
            },
            {
                path: 'setting',
                name: 'Setting',
                component: () => import('../views/Setting')
            }
        ]
    }
]
```

在页面中

```html
<div class="menu">
  <li><router-link to="/about/order">我的订单</router-link></li>
  <li><router-link to="/about/setting">个人设置</router-link></li>
</div>
<div class="content">
  <router-view/>
</div>
```

## 路由参数传递

### params类型

带参数的路由定义

```js
const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/about',
        name: 'About',
        component: () => import('../views/About.vue'),
        children: [            
            {
                // 通过:id的方式可以绑定一个参数
                // 从而实现用同一个组件模板显示不同内容
                // id这个参数名是任意起
                path: 'page/:id',
                component: () => import('../views/Page')
            }
        ]
    }
]
```

传递参数

```html
<ul>
  <li v-for="item in articles">
    <router-link :to="'/about/page/'+item.id">{{ item.title }}</router-link>
  </li>
</ul>
```

获取参数

```html
<!--params.id中的id要和子路由中声明的参数名称一致-->
文章ID:{{ $route.params.id }}
```

也可以用计算属性的方式获取

```html
文章ID:{{ pageid }}
```

```js
export default {
  name: "Page",
  computed: {
    pageid() {
      // params.id中的id要和子路由中声明的参数名称一致
      return this.$route.params.id
    }
  }
}
```

### query类型

路由定义用传统方式

```js
const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/about',
        name: 'About',
        component: () => import('../views/About.vue'),
        children: [            
            {
                path: 'article',
                component: () => import('../views/Article')
            }
        ]
    }
]
```

传递参数

```html
<ul>
  <li>
    <!--以对象的形式传递-->
    <router-link :to="{path:'/about/article', query:{name:'zhang',age:20}}">文章二</router-link>
  </li>
</ul>
```

```html
<!--自定义方式传递参数-->
<button @click="$router.push({path:'/about/article',query:{name:'sun',age:19}})">文章三</button>
```

获取参数

```html
姓名:{{ $route.query.name }}
年龄:{{ $route.query.age }}
```
