---
title: Vue组合式API——CompositionAPI
tags:
  - Vue
  - Composition API
  - 组合式API
categories: 前端
cover: https://user-images.githubusercontent.com/56388518/193992571-90879f64-0a1a-41d7-b5bf-40c47ed459a2.png
abbrlink: 12922
date: 2022-03-18 00:17:36
---

# Composition API（组合式API）

使用传统option配置方法写组件的时候，随着业务复杂度越来越高，代码量会不断加大，由于相关业务的代码需要遵循option的配置写到特定的区域，导致后续维护非常复杂，同时代码可复用性不高，而composition-api就是为了解决这个问题而生的。

Composition API是为了实现基于函数的逻辑复用机制而产生的，主要思想是，我们将它们定义为从新的setup函数返回的JavaScript变量，而不是将组件的功能（如state、method、computed等）定义为对象属性。

## 基本示例

原生Vue的实现形式

```html
<template>
  <div class="home">
    <h3>Count:{{ count }}</h3>
    <h3>Double count:{{ double }}</h3>
    <button @click="add">+</button>
  </div>
</template>
```

```js
<script>
export default {
  name: 'Home',
  data() {
    return {
      count: 0,
    }
  },
  computed: {
    double() {
      return this.count * 2;
    }
  },
  methods: {
    add() {
      this.count++;
    }
  }
}
</script>
```

使用Composition API的实现形式

```html
<template>
  <div class="about">
    <h3>Count:{{ data.count }}</h3>
    <h3>Double count:{{ data.double }}</h3>
    <button @click="add">+</button>

  </div>
</template>
```

```js
<script>
import {reactive, computed} from 'vue'

export default {
  setup() {
    const data = reactive({
      count: 0,
      double: computed(() => data.count * 2),
    });

    function add() {
      data.count++;
    }

    return {data, add};
  }
}
</script>
```

## `setup()`：Composition API的入口函数

`setup()`在`beforeCreate()`之前执行

```js
export default {
  name: "SubComp",
  data() {
    return {
      msg: 'Test message.'
    }
  },
  props: {
    one: {
      type: String
    },
    two: {
      type: String
    }
  },
  // props是父组件传过来的数据，context是上下文
  setup(props,context) {
    console.log('setup is called.');
    console.log(this);
    console.log(props.one + props.two);
  },
}
```

context对象包含以下属性

```js
const MyComponent = {
  setup(props,context){
    context.attrs // 父组件传过来的属性，即使没有用props声明也可以访问
    context.slots // 获取父组件传进来的插槽内容
    context.parent
    context.root
    context.emit // 向父组件传递数据
    context.refs  
  }
}
```

## 常用API

`ref()`函数用来为给定的值创建一个响应式的数据对象，`ref()`的返回值是一个对象，这个对象只包含一个`.value`属性

```js
export default {
  name: "CommonAPI",
  setup() {
    let num2 = ref(22);
    let myfun2 = (newvalue) => {
      num2.value = newvalue;
    }
    return {
      num2,
      myfun2
    }
  }
}
```

`reactive()`声明对象类型的响应式数据

```js
export default {
  name: "CommonAPI",
  setup() {
    let user = reactive({
      name: 'zhangsan',
      age: 100,
      sex: '男'
    });

    return {
      user
    }
  }
}
```

`toRefs()`将对象的数据成员拆解成单独的变量，并变成响应式的

```js
export default {
  name: "CommonAPI",
  setup() {
    let user = reactive({
      name: 'zhangsan',
      age: 100,
      sex: '男'
    });

    return {
      ...toRefs(user), // 三个点表示将对象拆解成单独的变量
    }
  }
}
```

`readonly()`将响应式数据变为原生数据

```js
export default {
  name: "CommonAPI",
  setup() {
    let user = reactive({
      name: 'zhangsan',
      age: 100,
      sex: '男'
    });

    let user2 = readonly(user);

    return {
      user2
    }
  }
}
```

`isRef()`判断该变量是否为响应式变量

```js
export default {
  name: "CommonAPI",
  setup() {
    let num2 = 1;
    let num3 = isRef(num2) ? num2.value : num2 ;

    return {
      num2,
      num3
    }
  }
}
```

## 计算属性

```js
export default {
  name: "Computed",
  setup() {
    const user = reactive({
      firstname: 'san',
      lastname: 'zhang'
    });

    let fullname = computed(() => {
      return user.firstname + '·' + user.lastname;
    });

    return {
      ...toRefs(user),
      fullname
    }
  }
}
```

## 侦听器watch

```js
export default {
  name: "Watch",
  setup() {
    let a = ref(1);
    let b = ref(2);
    // 只写回调会初始化执行
    // 函数内部用到哪个变量就会监听哪个变量
    watch(() => {
      console.log(a.value + '---' + b.value);
    });

    //写一个参数则会监听指定的变量
    // 不会初始化执行
    watch(a, () => {
      console.log('a changed');
    });

    //若想初始化执行，则加入第三个参数
    watch(a, () => {
      console.log('a changed');
    }, {immediate: true});

    // 还可以传入两个参数获取变化前后的
    watch(a, (newA, oldA) => {
      console.log('A changed from ' + oldA + ' to ' + newA);
    }, {immediate: true});        

    // 也可以监听多个值及其变化
    watch([a, b], ([newA, newB], [oldA, oldB]) => {
      console.log('A changed from ' + oldA + ' to ' + newA);
      console.log('B changed from ' + oldB + ' to ' + newB);
    }, {immediate: true});            

    // 立即执行传入的函数，并响应式追踪其依赖，并在其依赖变更时重新运行该函数
    watchEffect(() => {
      console.log(a.value + '###' + b.value);
    });

    return {
      a,
      b
    }
  }
}
```

若要监听对象类型的变量

```js
export default {
  name: "Watch",
  setup() {
    const user = reactive({
      a: 1,
      b: 2
    });

    watch(user, () => {
      console.log('User changed');
    });

    watch(() => user.a, (newx, oldx) => {
      console.log('User.a changed from ' + newx + ' to ' + oldx);
    });

    watch([() => user.a, () => user.b], ([newA, newB], [oldA, oldB]) => {
      console.log('User.a changed from ' + newA + ' to ' + oldA);
      console.log('User.b changed from ' + newB + ' to ' + oldB);
    });

    watchEffect(() => {
      console.log(user.a)
    });

    return {
      ...toRefs(user),
    }
  }
}
```

## Composition API中的生命周期函数

原生生命周期函数与Composition API之间的映射关系

`beforeCreate`→`setup()`

`created`→`setup()`

`beforeMount`→`onBeforeMount()`

`mounted`→`onMounted()`

`beforeUpdate`→`onBeforeUpdate()`

`updated`→`onUpdated()`

`beforeUnmount`→`onBeforeUnmount()`

`Unmounted`→`onUnmounted()`

使用时要注意，是以回调方法的形式调用的

```js
export default {
  name: "LifeHook",
  setup() {
    onBeforeMount(() => {
      console.log('onBeforeMount');
    });
  },
}
```

## `provide`和`inject`的使用

`provide`和`inject`这对选项允许祖先组件向其所有子孙后代组件注入一个依赖，不论组件层次有多深，在其上下游关系成立的时间内始终生效

`provide`提供变量，相当于加强版父组件prop，可以跨越中间件

`inject`注入变量，相当于加强版子组件props

### 原生Vue方式

祖先组件

```js
export default {
  name: "RootApp",
  components: {SecondApp},
  data() {
    return {
      title: "The information is provided by root component."
    }
  },
  provide() {
    return {
      title: this.title,
    }
  },
}
```

子孙组件

```js
export default {
  name: "ThirdApp",
  inject: ['title'],
}
```

这种方式传递的变量不是响应式的

### Composition API方式

祖先组件

```js
export default {
  name: "RootApp",
  components: {SecondApp},
  setup() {
    let info = ref('The information is provided by root component through setup.');

    provide('info', info);

    return {
      info,
    }
  }
}
```

子孙组件

```js
export default {
  name: "ThirdApp",
  setup() {
    let info = inject('info');

    return {
      info,
    }
  }
}
```

这种方式传递的变量是响应式的，而且是双向的响应式，即在祖先组件中更改变量，子孙组件中会发生变化；在子孙组件中更改变量，祖先组件中也会发生变化

## Composition API处理路由

因为无权访问`setup()`中的`this`，故使用`useRouter()`和`useRoute()`函数

```js
export default {
  name: "Page",
  setup() {
    const route = useRoute();
    const router = useRouter();

    let id = ref();

    watch(() => route.params, (newId) => {
      id.value = newId.id;
    }, {immediate: true})

    return {
      id,
    }
  }
}
```

组件级的导航守卫则替换为：

`onBeforeRouteLeave((to,from)=>{})`

`onBeforeRouteUpdate(async(to,from)=>{})`

```js
export default {
  name: "Page",
  setup() {
    onBeforeRouteLeave((to, from) => {
      return window.confirm(`确定要从${from.fullPath}到${to.fullPath}`);
    });
  }
}
```

## Composition API结合Vuex

使用`useStore()`函数

```js
export default {
  name: "Vuex",
  setup() {
    const store = useStore();

    return {
      num2: computed(() => store.state.num2),
      double2: computed(() => store.getters.double2),
      cnum2: (newnum) => store.commit('changenum2', newnum),
      cnum22: () => store.dispatch('timecnum2'),
    }
  }
}
```
