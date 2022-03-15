---
title: Vue_CLI安装及项目搭建
tags:
  - Vue
  - Vue_CLI
categories: 前端
cover: >-
  https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fi0.hdslb.com%2Fbfs%2Farticle%2Fae3f2cf7a3a026e00e18d12ef1a27dbb6e8d5d42.jpg&refer=http%3A%2F%2Fi0.hdslb.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1649934350&t=e7df522ff25d35cbc6f964ce2b6f32f8
abbrlink: 8559
date: 2022-03-15 19:02:57
---

## Vue-CLI (Command Line Interface)

安装

```shell
npm install -g @vue/cli
```

检查版本

```shell
vue --version
```

创建项目

```shell
vue create <Project Name>
```



## 创建过程配置

```shell
Vue CLI v4.5.15
? Please pick a preset:
  Default ([Vue 2] babel, eslint)
> Default (Vue 3) ([Vue 3] babel, eslint)
  Manually select features
```

选择Vue 3项目，默认自带babel和eslint

babel：将ES6语法转换为ES5语法

eslint：检查语法，修复不规范代码

若选择第三项手动配置

```shell
? Check the features needed for your project: (Press <space> to select, <a> to toggle all, <i> to invert selection)
>(*) Choose Vue version
 (*) Babel # 将ES6语法转换为ES5语法
 ( ) TypeScript # JS的超集
 ( ) Progressive Web App (PWA) Support # App支持
 (*) Router # 路由
 (*) Vuex # 状态管理
 (*) CSS Pre-processors # sass，less转css
 (*) Linter / Formatter # 检查语法规范
 ( ) Unit Testing # 单元测试
 ( ) E2E Testing # 端对端测试
```

路由是否使用历史模式，选择Y

```shell
? Use history mode for router? (Requires proper server setup for index fallback in production) (Y/n)
```

处理Sass还是Less，根据需求选择即可

```shell
? Pick a CSS pre-processor (PostCSS, Autoprefixer and CSS Modules are supported by default): (Use arrow keys)
> Sass/SCSS (with dart-sass)
  Sass/SCSS (with node-sass)
  Less
  Stylus
```

选择eslint标准，选择Airbnb

```shell
? Pick a linter / formatter config: (Use arrow keys)
  ESLint with error prevention only
> ESLint + Airbnb config
  ESLint + Standard config
  ESLint + Prettier
```

保存时修复与提交git时修复，可全选

```shell
? Pick additional lint features: (Press <space> to select, <a> to toggle all, <i> to invert selection)
>(*) Lint on save
 (*) Lint and fix on commit (requires Git)
```

各插件的配置文件独立还是全部放入package.json

```shell
? Where do you prefer placing config for Babel, ESLint, etc.? (Use arrow keys)
> In dedicated config files
  In package.json
```

是否为下次手动配置记录此次选择

```shell
? Save this as a preset for future projects? (y/N)
```

项目名称

```shell
? Save preset as:
```



## 基本操作

build

```shell
npm run build
```

运行

```shell
npm run serve
```
