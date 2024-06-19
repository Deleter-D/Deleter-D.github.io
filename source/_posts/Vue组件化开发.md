---
title: Vue组件化开发
tags:
  - Vue
  - Vue组件
categories: [Java,前端,Vue]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193992093-afdfc04a-59e9-4f6e-8849-f56062508265.png
abbrlink: 61428
date: 2022-03-15 19:08:07
---

Vue组件化开发的思想及父子组件之间的通信方式，以及Vue中插槽的基本用法。

<!--more-->

## Vue组件

### 创建组件

```js
// 创建一个Vue 应用
const app = Vue.createApp({})

// 定义一个名为 button-counter 的新全局组件
app.component('button-counter', {
  data() {
    return {
      count: 0
    }
  },
  template: `
    <button @click="count++">
      You clicked me {{ count }} times.
    </button>`
})
```

### 样式的传递

```html
<style>
    这里的内容可以传递到子孙组件
</style>
<style scoped>
    这里的内容仅能在本组件中使用
</style>
```

## 组件数据通信

### 父传子：Pass props

传递数据的父模块需要在标签中将数据以属性的形式传递

```html
<MyMain msg="hello"></MyMain>
```

接收数据的子模块需利用`props`来接收

```js
export default {
  name: "MyMain",
  props: ['msg'],
  components: {MyConn}
}
```

引用时利用插值引用即可

关于props：

如果传递的变量类型不同，需要以如下方式指定

```js
props: {
    msg: {type: String},
    article: {type: Array},
  },
```

用如下方式指定缺省值，即当没有数据传过来时的默认值

```js
props: {
    msg: {
      type: String,
      default: '###',
    },
    article: {
      type: Array,
      default: ['aaa', 'bbb', 'ccc'],
    },
  },
```

用如下方式要求属性必须传递

```js
props: {
    msg: {type: String,},
    title: {
      type: String,
      required: true,
    },
  },
```

**已经传递给子组件的数据，子组件还可以继续传给孙组件**

### 子传父：$emit Event

在要传数据的子组件中绑定一个事件

```html
<button @click="changenum(2)">+</button>
```

在方法中调用`$emit`方法，第一个参数为自定义的事件名，第二个参数为待传的数据

```js
methods: {
    changenum(num) {
      this.$emit('mycountevent', num);
    },
  },
```

在接收数据的父模块中绑定自定义的事件

```html
<MyConn @mycountevent="mydemo"></MyConn>
```

再利用自定义事件触发的方法使用数据

```js
methods: {
    mydemo(data) {
      this.count += data;
    },
  }
```

## 父子组件之间的访问

#### 子组件调用父组件的方法：`$parent`或`$root`

父组件中的方法和数据

```js
data() {
    return {
      msg1: 'hello1',
      count: 0,
    }
  },
methods: {
    changen() {
      this.count++;
    }
  },
```

子组件中调用

```js
methods: {
    one() {
      console.log(this.$parent.count); // 访问数据
      this.$parent.changen(); // 访问方法
    }
  },
```

同样可利用该方法访问爷组件的方法和数据

```js
methods: {
    one() {
      console.log(this.$parent.$parent.msg); // 访问数据
    }
  },
```

利用`$root`直接访问根组件

```js
methods: {
    one() {
      console.log(this.$root.msg); // 访问数据
    }
  },
```

#### 父组件调用子组件的方法：`$children`或`$refs`

子组件中的方法和数据

```js
data() {
    return {
      msg: 'This is a test.',
      num: 0
    }
  },
methods: {   
    changeone() {
      this.num++;
    }
  },
```

在父组件中给子组件标签定义`ref`属性

```html
<MyConn ref="aaa" @mycountevent="mydemo"></MyConn>
<button @click="two">子加一</button>
```

然后在父组件中利用`$refs`调用子组件方法和数据

```js
methods: {   
    two() {
      this.$refs.aaa.changeone(); // 方法
      console.log(this.$refs.aaa.msg); // 访问数据
    }
  }
```

**在Vue3.x中删除了`$children`方法**

`$children`的使用方式与`$parent`类似，只不过`$children`是利用数组的形式访问子组件，但数组的形式容易搞混子组件的顺序，且在插入新的子组件后数组序列会改变，所以不推荐使用该方式

## 插槽

### 实现组件的扩展性，抽取共性设计组件

子组件中添加`<slot>`标签占位

```html
<template>
  <div class="mybar">
    <slot></slot>
  </div>
</template>
```

父组件调用子组件时可以在子组件内添加标签，多个标签也可以

```html
<template>
  SideBar
  <MyBar>
    <button>提交</button>
  </MyBar>
  <MyBar>
    <a href="#">提交</a>
  </MyBar>
  <MyBar>
    <p>
      <span>11</span>
      <span>22</span>
    </p>
  </MyBar>
</template></template>
```

还可以为设置插槽的缺省值，当父组件调用子组件且未在子组件标签内部添加标签时就会利用缺省值

```html
<template>
  <div class="mybar">
    <slot>
      <button>提交</button>
    </slot>
  </div>
</template>
```

### 子组件中可以有多个插槽

```html
<template>
  <div>
    <slot></slot>
    <slot name="one"></slot>
  </div>
</template>
```

父组件调用时，若指定插槽名称，则会替换相同名称的插槽，若未指定名称，则只会替换掉没有名称的插槽

```html
<!--指定名称替换-->
<MyBar>
  <template v-slot:one>
    <a href="#">提交</a>
  </template>
</MyBar>
<!--使用缺省值-->
<MyBar>
  <template v-slot:default>
    <a href="#">提交</a>
  </template>
</MyBar>
```

### 利用插槽给父组件传递数据

在子组件的插槽中，自定义一个属性

```html
<slot :user="user"></slot>
```

父组件中获取，test接收了子组件中的自定义属性

```html
<template v-slot:default="test">
  <a href="#">{{ test.user.name }}</a>
</template>
```
