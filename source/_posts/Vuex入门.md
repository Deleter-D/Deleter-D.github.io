---
title: Vuex入门
tags:
  - Vue
  - Vuex
categories: [前端,Vue]
toc: true
cover: https://user-images.githubusercontent.com/56388518/193992384-179d45b4-705d-4a45-842a-512ad67d7b3d.png
abbrlink: 33216
date: 2022-03-18 00:15:19
---

Vue生态中的又一员猛将Vuex，负责管理全局状态。

<!--more-->

# Vuex

Vuex就相当于一个全局的状态管理，项目不复杂的情况下不建议引入Vuex

![vuex](https://user-images.githubusercontent.com/56388518/193992464-b6c699e2-eacd-4340-96ce-8591888b995d.png)

由Vue Components分发一个Action，然后与后端进行交互，由Actions提交到Mutations，通过Mutations修改状态，这样修改可以通过Devtools记录下操作，再由State的改变渲染Vue Components

**和路由类似，Vuex有一个全局的变量`$store`，利用该变量进行一系列操作**

## 核心概念

### `state`

定义全局变量

```js
export default createStore({
    state: {
        num: 0,
    },
    mutations: {},
    actions: {},
    modules: {}
})
```

### `mutations`

定义对`state`的操作

```js
export default createStore({
    state: {
        dnum: 0,
    },
    mutations: {
        sub(state) {
            state.dnum--;
        },
        add(state) {
            state.dnum++;
        }
    },
    actions: {},
    modules: {}
})
```

调用`mutations`的方法时，应在组件内定义方法调用

```js
export default {
  name: 'Home',
  methods: {
    add1() {
      this.$store.commit('add');
    },
    sub1() {
      this.$store.commit('sub');
    }
  }
}
```

传参问题：组件向`mutations`传递参数

`mutations`的参数除`state`外只支持一个参数，所以：

传递单个参数直接传递

```js
export default {
  name: 'Home',
  methods: {
    add2() {
      let count = 2;
      this.$store.commit('add2',count);
    },
    sub2() {
      let count = 2;
      this.$store.commit('sub2',count);
    }
  }
}
```

传递多个参数以对象的形式传递

```js
export default {
  name: 'Home',
  methods: {
    add3() {
      let payload = {
        count: 2,
        num: 1,
      }
      this.$store.commit('add3', payload);
    },
    sub3() {
      let payload = {
        count: 2,
        num: 1,
      }
      this.$store.commit('sub3', payload);
    },
  }
}
```

### `Getter`

全局的计算属性

```js
export default createStore({
    state: {
        num: 0,
    },
    mutations: {},
    getters: {
        vxnum(state) {
            return state.num * state.num;
        },
    },
    actions: {},
    modules: {}
})
```

若想要在`getters`中定义的计算属性想要复用另一个`getters`中的计算属性，则：

```js
export default createStore({
    state: {
        cartlist: [
            {name: '《忒修斯之船》', price: 129},
            {name: '《忒修斯之船2》', price: 139},
            {name: '《忒修斯之船3》', price: 149},
        ],
    },
    mutations: {},
    getters: {
        goodsnum(state) {
            return state.cartlist.filter(n => n.price > 130);
        },
        goodsprice(state, getters) { // 第二个参数传入getters
            return getters.goodsnum.reduce((s, n) => s + n.price, 0);
        }
    },
    actions: {},
    modules: {}
})
```

若想实现带参数的计算属性，则：

```js
export default createStore({
    state: {
        cartlist: [
            {name: '《忒修斯之船》', price: 129},
            {name: '《忒修斯之船2》', price: 139},
            {name: '《忒修斯之船3》', price: 149},
        ],
    },
    mutations: {},
    getters: {
        goodsfilter(state) {
            // price处可以传多个参数
            return function (price) {
                return state.cartlist.filter(n => n.price > price);
            }

            // 简写
            // return price => state.cartlist.filter(n => n.price > price);
        }
    },
    actions: {},
    modules: {}
})
```

### `Actions`

处理异步请求

```js
import {createStore} from 'vuex'

export default createStore({
    state: {
        num: 0,
    },
    mutations: {
        cnum(state) {
            state.num = 99;
        }
    },
    getters: {},
    actions: {       
        demo(context) {
            // 操作
        },
        // 以解构的方式直接获取到context中的state等方法和属性
        fun({state, commit, getters},payload) {
            // 操作
        }
    },
    modules: {}
})
```

Action 函数接受一个与 store 实例具有相同方法和属性的 context 对象，因此你可以调用 `context.commit` 提交一个 mutation，或者通过 `context.state` 和 `context.getters` 来获取 state 和 getters。

### `Mudules`

每个模块拥有自己的 `state`、`mutation`、`action`、`getter`、甚至是嵌套子模块

```js
export default createStore({
    state: {},
    mutations: {},
    getters: {},
    actions: {},
    modules: {
        user: {
            state: () => ({
                name: 'zhangsan',
                age: 100,
            }),
            getters: {},
            mutations: {},
            actions: {},
        },
        article: {},
    }
})
```

也可以在外部定义

```js
const user = {
    state: () => ({
        name: 'zhangsan',
        age: 100
    }),
    mutations: {},
    getters: {},
    actions: {},
    modules: {}
}

export default createStore({
    state: {},
    mutations: {},
    getters: {},
    actions: {},
    modules: {
        user,
    }
})
```

注意调用子模块中的`state`时应为如下形式，模块相当于放在了state里面，而`mutation`、`action`、`getter`、不需要加模块名，所以这些中的方法名称不能重复

```html
{{ $store.state.user.name }}
```

子模块调用根模块的`state`如下，还可以通过`rootGetters`参数调用根模块的getters

```js
const user = {
    state: () => ({
        name: 'zhangsan',
        age: 100
    }),
    mutations: {
        setname(state, payload) {
            state.name = payload;
        }
    },
    getters: {
        fullname(state) {
            return state.name + state.age;
        },
        // 调用自身的其他getters
        fullname2(state, getters) {
            return getters.fullname + '222';
        },
        // 利用rootState参数调用根模块的state
        fullname3(state, getters, rootState) {
            return getters.fullname2 + rootState.num;
        }
    },
    actions: {},
    modules: {}
}
```

---

**可以根据需要将各个属性拆分成不同的js文件**
