---
title: Vue模板语法
tags:
  - Vue
  - v-指令
categories: [前端,Vue]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193991999-72eab280-1d5f-4681-8811-c22a7f97277b.png
abbrlink: 37641
date: 2022-03-15 19:06:25
---

Vue的模板语法及v-指令的基本用法。

<!--more-->

## options

```html
data <!--数据-->
methods <!--定义方法-->
computed <!--计算属性-->
```

计算属性的两种写法

```js
computed: {
    prop: {
      get() {
        // 操作
      },
      set() {
        // 操作
      },
    },
  },
```

当属性只需要获取时简写为

```js
computed: {
    prop() {
      // 操作
    }
  },
```

## 插值

```html
{{msg}}
```

## v-指令

#### v-pre：将不再渲染msg的内容，页面直接显示{{msg}}

```html
<h1 v-pre>{{msg}}</h1>
```

#### v-once：只引用一次变量的值，变量值的后续改变不会影响该引用

```html
<h1 v-once>{{num}}</h1>
```

#### v-text：指定显示的文本信息，使用该指令时标签中不可以有内容

```html
<h1 v-text="Test"></h1>
<h1 v-text="Test">content</h1> content的位置不能有内容
```

#### v-html：若返回的变量中有html标签等，可以用该指令指定，能够自动渲染其中的html

```html
<h1 v-html="url"></h1>
```

#### v-bind：动态绑定属性，任何属性都可以绑定

```html
<h2 v-bind:title="msg">{{ msg }}</h2>
<h2 :title="msg">{{ msg }}</h2> <!--语法糖写法-->
```

#### v-on：绑定事件监听器

```html
<button v-on:click="sub()">-</button>
<input type="text" size="2" v-model="num">
<button @click="add">+</button> <!--语法糖写法-->
```

绑定事件调用函数时，若函数声明时有一个参数，调用时未传入参数，则该参数默认为事件监听对象；若函数声明时有多个参数，调用时需要用`$event`显式的将事件监听对象传入

v-on事件修饰符：

`.stop`：阻止事件冒泡

`.self`：当事件在该元素本身触发时才触发事件

`.capture`：添加事件侦听器时，使用事件捕获模式，即优先捕获使用该修饰符的事件

`.prevent`：阻止默认事件

`.once`：事件只触发一次

#### v-if：条件分支

```html
<button @click="card=1">①</button>
<button @click="card=2">②</button>
<button @click="card=3">③</button>
<button @click="card=4">④</button>
<div v-if="card==1">
  111 <br>
</div>
<div v-else-if="card==2">
  222 <br>
</div>
<div v-else-if="card==3">
  333 <br>
</div>
<div v-else>
  ### <br>
</div>
```

`v-if`与`v-show`：

`v-if`：是真正的条件渲染，它会确保在切换过程中条件块内的事件监听器和子组件适当被销毁和重建

`v-show`：不管初始条件是什么，元素总是会被渲染，只是简单的基于CSS进行切换

#### v-for：循环

注意要显式的绑定`:key`

```html
<ul>
  <li v-for="item in list" :key="item">{{item}}</li>
</ul>
```

可以利用`slice()`限制循环区间：

```html
<ul>
  <li v-for="item in list.slice(0,3)" :key="item">{{item}}</li>
</ul>
```

若需要数组下标：

```html
<ul>
  <li v-for="(item,index) in list" :key="item">{{index}}-{{ item }}</li>
</ul>
```

遍历对象时，键、值、下标都可以遍历

```html
<ul>
  <li v-for="(item,key,index) in obj" :key="item">{{key}}-{{item}}-{{index}}</li>
</ul>
```

遍历对象数组

```html
<ul>
  <li v-for="(item,index) in books" :key="item.id">{{index}}-{{item.name}}-{{item.price}}</li>
</ul>
```

#### v-model：双向绑定，常与表单一起使用

```html
<input type="text" v-model="msg">
```

`v-model`修饰符：

`.lazy`：懒加载

`.number`：让其转换为number类型

`.trim`自动过滤掉输入框的首尾空格
