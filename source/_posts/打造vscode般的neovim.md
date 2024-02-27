---
title: 打造vscode般的neovim
toc: true
mathjax: true
tags:
  - Neovim
  - 编辑器
  - VSCode
categories: 折腾
abbrlink: 22702
date: 2024-02-02 16:56:36
---

偶然在Youtube上看到一个大佬的Neovim配置教程，深入浅出，逻辑条理，加了一点自己的内容，就产生了这么一个授人以渔的教程。

<!-- more -->

# 打造VS Code般的Neovim

## 一点废话

Neovim是什么就不过多介绍了，有兴趣可以自行Google，没兴趣那我就一句话介绍Neovim：

- ~~一个可以变得很好看很好用的基于TUI的文本编辑器。~~
- ~~由于Vim拒绝了两个大补丁，作者一怒之下开的新分支。~~
- Neovim是一个积极重构Vim的项目，旨在简化维护和鼓励贡献，在多个开发人员之间分配工作，在不修改核心的情况下创建高级UI，最大化可扩展性。

就是单纯用vim觉得丑，所以跑路，尝试配置一个媲美VS Code体验的Neovim。教程来自于油管一个大佬，讲的很不错，不像某些教程，只告诉你怎么做，不告诉你为什么。这里附上链接，有兴趣可以自行观看。[Neovim for Newbs. FREE NEOVIM COURSE - YouTube](https://www.youtube.com/playlist?list=PLsz00TDipIffreIaUNk64KxTIkQaGguqn)

后面的内容有一个很重要的宗旨，你不是非得按照这个文档描述的步骤一步一步来做，这个文档最重要的目的是告诉你Neovim配置的方法。当你学会这个方法之后，可以完全自定义想要的内容。当然，这个时代不一定非要自己造轮子，github上有很多别人配置好的预制菜，你只需要clone下来就可以用。但本文档可以让你了解整个流程，在抄别人作业过程中遇到问题可以自己解决掉。

## 依赖

安装之前，你大致需要以下这些依赖工具，请确保环境中已经安装。

```
git, gcc, g++, python, python-venv, unzip
```

## 安装Neovim

> 官方仓库：[neovim/neovim](https://github.com/neovim/neovim)

Windows系统可以直接下载zip包，解压到任何位置，然后运行`nvim.exe`。

Linux下可以下载`nvim.appimage`，然后执行`chmod u+x nvim.appimage && ./nvim.appimage`。

安装好后在终端里运行`nvim`，就能得到下图。

![](https://github.com/Deleter-D/Images/assets/56388518/6ef7b27d-58e8-413c-b416-8ce95856db90)

## 安装Nerd fonts

我建议在安装完Neovim后先把字体安装好，后面用到的一些美化插件是比较依赖于`Nerd fonts`的。打开官方网站[Nerd Fonts](https://www.nerdfonts.com/#home)，选一个喜欢的字体下载。这里推荐`Hack Nerd Font`。

Windows下只需要双击ttf文件，点击安装即可。

Linux下需要执行以下步骤。

```shell
sudo unzip Hack -d /usr/share/fonts/Hack
cd /usr/share/fonts/Hack
sudo mkfontscale  # 生成核心字体信息
sudo mkfontdir    # 生成字体文件夹
sudo fc-cache -fv # 刷新系统字体缓存
```

## 初始化配置文件

Neovim的配置全部是基于`lua`的，所以后面创建的所以脚本均为`lua`脚本。

创建初始化配置文件`init.lua`，进行一些基础配置。

> Windows中配置文件放在`C:\Users\username\AppData\Local\nvim`下；
>
> Linux中放在`~/.config/nvim`下

```lua
vim.cmd("set expandtab") 	 -- 使用空格代替制表符，确保在不同的编辑环境中保持一致
vim.cmd("set tabstop=2")     -- 设置制表符的宽度为2个空格
vim.cmd("set softtabstop=2") -- 设置软制表符的宽度也为2个空格，软制表符指的是按下Tab的时候插入的空格数
vim.cmd("set shiftwidth=2")  -- 设置每级缩进为2个空格
```

然后在命令模式下执行`source %`，激活当前配置文件。

## 插件管理器

目前主流的插件管理器有两个：`lazy`和`packer`。

> 官方仓库：
>
> - [folke/lazy.nvim](https://github.com/folke/lazy.nvim)
> - [wbthomason/packer.nvim](https://github.com/wbthomason/packer.nvim)

两个包管理的差别不大，~~为了贯彻好看的宗旨~~，这里我们选择`lazy`作为包管理器。

安装`lazy`非常简单，只需要将README中Installation部分的语句复制粘贴到`init.lua`中即可。下面这些语句实现的大致功能为，为`lazy`指定一个存放数据的目录，然后检查`lazy`是否被安装，若没有则从`github`上拉取安装。

> 这里注意要修改一下链接，将仓库链接从`https`协议改为`ssh`协议。如果你的网络条件支持稳定的通过`https`访问`github`，那当我这句话没说过。

```lua
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git",
    "clone",
    "--filter=blob:none",
    "ssh://git@github.com/folke/lazy.nvim.git", -- 注意此处需要魔改一下
    "--branch=stable", -- latest stable release
    lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)
```

然后我们创建两个变量来存放插件和选项，并调用`setup()`函数来配置它。这个`setup()`函数将伴随整个插件配置过程，具体细节无需关心，毕竟我们也不写`lua`，我们只是`lua`的搬运工。

```lua
local plugins = {}
local opts = {}

require("lazy").setup(plugins, opts)
```

完成上面的步骤后，同样`source %`激活一下，这里可能会卡顿一会儿，因为此时后台在从`github`上拉取并安装`lazy`。

安装好后，命令模式下执行`: Lazy`就可以打开`lazy`的管理界面了。

![](https://github.com/Deleter-D/Images/assets/56388518/b76255d1-a1ee-4f02-b10a-9c259b4127b6)

## 主题

主题这里，大佬推荐`catppuccin`，确实挺好看，无脑跟！

> 官方仓库：[catppuccin/nvim](https://github.com/catppuccin/nvim)

有了`lazy`，安装这些插件及其简单，只需要将下面这行放入我们上一步中创建的`plugins`变量中。

```lua
local plugins = {
  { "catppuccin/nvim", name = "catppuccin", priority = 1000 },
}
```

此时`:q`退出Neovim，然后重新打开，你会发现`lazy`管理界面跳了出来，然后提示你正在安装。等待一会儿就可以看到，主题颜色已经变了。但还没结束，现在主题颜色变了只是因为`lazy`帮你安装完成后自动加载了该插件。如果此时你再退出重开一次，会发现主题又变回了默认状态。我们需要再添加两行代码，将默认的主题配置成`catppuccin`。

```lua
require("catppuccin").setup()
vim.cmd.colorscheme "catppuccin"
```

到此，`catppuccin`就配置完成了，不出意外你将得到下面这样的状态。

![](https://github.com/Deleter-D/Images/assets/56388518/27b03968-3582-458e-91a6-438a3ce64739)

## 核心插件

### Telescope

`telescope`是一个基于列表模糊查找器，可以用来查找文件。其他更高级的功能可以参考官方仓库。

> 官方仓库：[nvim-telescope/telescope.nvim](https://github.com/nvim-telescope/telescope.nvim)

安装插件同样及其简单，在`plugins`变量中增加一项即可。

```lua
local plugins = {
  { "catppuccin/nvim", name = "catppuccin", priority = 1000 },
  {
    'nvim-telescope/telescope.nvim', tag = '0.1.5',
    dependencies = { 'nvim-lua/plenary.nvim' }
  },
}
```

退出Neovim再重新打开，你又会看到`lazy`在帮你安装插件。

![](https://github.com/Deleter-D/Images/assets/56388518/d415dabc-08d6-412c-b0b6-42c8b4ccef40)

上图是已经安装好的状态。然后在`init.lua`中对`telescope`进行一些配置。这里主要介绍两个功能，基本能够满足需要，其余的功能可以参考官方仓库自行配置。

#### 文件搜索

在`init.lua`中添加如下两行。

```lua
local builtin = require("telescope.builtin") -- 用来加载telescope内置的内容
vim.keymap.set('n', '<C-p>', builtin.find_files, {}) -- 设置快捷键，Ctrl + P为模糊搜索文件
```

再重开Neovim，然后按`Ctrl + P`就可以打开`telescope`，输入文件名即可搜索文件。

![](https://github.com/Deleter-D/Images/assets/56388518/12f2dcb4-e1be-4903-9607-686929de3c0e)

#### 字符搜索

同样在`init.lua`中添加语句。

```lua
vim.g.mapleader = " " -- 设置leader键为空格
-- 快捷键为 <leader> + fg，这里f和g按顺序按就好，但是空格不要松开
vim.keymap.set('n', '<leader>fg', builtin.live_grep, {}) 
```

配置完后`:source %`一下，按上面配置好的快捷键即可打开字符搜索。下图是一个搜索`lazy`字符串的例子。

![](https://github.com/Deleter-D/Images/assets/56388518/127233e7-4338-4b5f-9818-9d2c82e97443)

> 到这一步可能会出现一个问题，搜索框可以正常打开，但是搜什么都是空白的。原因是缺少一个依赖工具`ripgrep`，只需要按官方仓库的提示安装即可。
>
> `ripgrep`官方仓库：[BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)
>
> 安装Installation章节的提示安装即可。
>
> - Windows下执行`winget install BurntSushi.ripgrep.MSVC`。
> - Linux下，以Ubuntu为例执行`sudo apt-get install ripgrep`。
>
> 其余操作系统官方仓库中写的都很清楚，安装完成该工具，重启终端并打开Neovim就可以正常使用了。

### Treesitter

`treesitter`是一个基于语法树的语法高亮插件。

> 官方仓库：[nvim-treesitter/nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter)

安装过程类似，还是在`plugins`中加入一行。

```lua
local plugins = {
  { "catppuccin/nvim", name = "catppuccin", priority = 1000 },
  {
    'nvim-telescope/telescope.nvim', tag = '0.1.5',
    dependencies = { 'nvim-lua/plenary.nvim' }
  },
  { "nvim-treesitter/nvim-treesitter", build = ":TSUpdate" },
}
```

重启Neovim，`lazy`会帮你安装，然后对其进行配置。

> 在配置`treesitter`之前，请确保你的系统路径中有可用的C编译器，`gcc`、`clang`、`MSVC`都可以。
>
> 如果你是Windows用户，要确保开启开发者模式。打开`设置 -> 系统 -> 开发者选项 -> 开发人员模式`。

在`init.lua`中添加如下代码。

```lua
local config = require("nvim-treesitter.configs")
config.setup({
  ensure_installed = { "lua", "c", "cpp" }, -- 确保安装lua，c和cpp相关的内容
  highlight = { enable = true }, -- 开启高亮
  indent = { enable = true },    -- 开启缩进
})
```

此时`:source %`一下，`treesitter`就会开始安装语言相关的内容。安装完后会发现，当前打开的这个`lua`脚本已经拥有了语法高亮。

![](https://github.com/Deleter-D/Images/assets/56388518/4a834eb2-f842-465a-8961-27be541d7e92)

你可以用`:TSInstall`来安装新的语言支持，用`:TSModuleInfo`来查看已经支持的语言。

### Neo-tree

`neo-tree`，就如同该插件的名称一样，是一个文件`tree`。同样在`plugins`中添加。

> 官方仓库：[nvim-neo-tree/neo-tree.nvim](https://github.com/nvim-neo-tree/neo-tree.nvim)

```lua
local plugins = {
  { "catppuccin/nvim", name = "catppuccin", priority = 1000 },
  {
    'nvim-telescope/telescope.nvim', tag = '0.1.5',
    dependencies = { 'nvim-lua/plenary.nvim' }
  },
  { "nvim-treesitter/nvim-treesitter", build = ":TSUpdate" },
  {
    "nvim-neo-tree/neo-tree.nvim",
    branch = "v3.x",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-tree/nvim-web-devicons",
      "MunifTanjim/nui.nvim",
    }
  }
}
```

重启Neovim，等待`lazy`安装插件，安装完成后即可使用`neo-tree`。

输入命令`:Neotree filesystem reveal left`就可以在左侧显示文件树了。当然，为这条命令设置一个快捷键会更方便。

```lua
vim.keymap.set('n', '<C-b>', ':Neotree filesystem reveal left<CR>', {})
```

这里为了保持VS Code的习惯，设置成了`Ctrl + b`，你可以根据自己的习惯来配置。

![](https://github.com/Deleter-D/Images/assets/56388518/b938e2af-50fb-48a3-9adb-1df39d998003)

## 模块化插件配置

配置到这里聪明的你可能已经发现一个问题了，我们所有的配置都集中在同一个文件`init.lua`中，这显然是不合理的。所以我们接下来将之前所做过的所有配置，拆分开来，同时体验一下`neo-tree`带来的便利。

按下`Ctrl + b`打开文件树，按`a`创建文件夹及文件`lua/plugins.lua`。将我们之前的`plugins`变量丢进`plugins.lua`脚本中，然后返回该变量，脚本内容如下。

```lua
return {
  { "catppuccin/nvim", name = "catppuccin", priority = 1000 },
  {
    'nvim-telescope/telescope.nvim', tag = '0.1.5',
    dependencies = { 'nvim-lua/plenary.nvim' }
  },
  { "nvim-treesitter/nvim-treesitter", build = ":TSUpdate" },
  {
    "nvim-neo-tree/neo-tree.nvim",
    branch = "v3.x",
    dependencies = {
      "nvim-lua/plenary.nvim",
      "nvim-tree/nvim-web-devicons",
      "MunifTanjim/nui.nvim",
    }
  }
}
```

然后将`init.lua`中的`plugins`变量删除，并修改下面的语句。

```lua
require("lazy").setup("plugins") -- 原本是 require("lazy").setup(plugins, opts)
```

我们创建的`plugins.lua`脚本相对`init.lua`的路径为`./lua/plugins.lua`，这里只需要写`"plugins"`即可，`lazy`会自动找到这个路径。重启Neovim确保工作正常。

但是这样的模块化依然不够，这只是将依托答辩从旱厕搬进了马桶，并没有好多少，所以我们需要对插件配置进一步拆分。

在`lua`文件夹下创建`plugins`文件夹，在`plugins`文件夹下创建我们第一个配置的插件`catppuccin`对应的脚本，命名为`catppuccin.lua`。

![](https://github.com/Deleter-D/Images/assets/56388518/9108a6d9-0b6b-4308-b71e-f9529e44ea08)

> 此时左下角提示`Config Change Detected. Reloading...`，这都要归功于`lazy`插件管理器，它会实时的检测插件配置的变换，自动帮你重新加载。

将`plugins.lua`中关于`catppuccin`的配置都转移到`catppuccin.lua`中，同样的方式进行返回。

```lua
return {
  "catppuccin/nvim", 
  name = "catppuccin", 
  priority = 1000
}
```

同时`init.lua`中还有一些关于`catppuccin`的配置。

```lua
require("catppuccin").setup()
vim.cmd.colorscheme "catppuccin"
```

想要将这些配置也搬入`catppuccin.lua`脚本，就需要用到`lazy`提供的一个配置项`config`。这个`config`需要我们定义一个函数，在函数体中对插件进行配置。同时，使用`config`就相当于自动调用了`require("...").setup(opts)`。所以我们对`catppuccin.lua`进行修改。

```lua
return {
  "catppuccin/nvim", 
  name = "catppuccin", 
  priority = 1000,
  config = function()
    vim.cmd.colorscheme "catppuccin"
  end
}
```

至此就实现了`catppuccin`插件的模块化，其他的所有插件都是同理的。我们将所有的插件都模块化，最终得到这样的文件结构。

![](https://github.com/Deleter-D/Images/assets/56388518/5a03004f-6add-4b1c-b2a6-29a1ee8b077d)

现在我们的`init.lua`已经比之前简洁多了，这种模式可以更方便的管理插件，并且可以随时添加新的插件，只需要添加一个新的`lua`脚本即可。

> 到这一步，我们之前用作过渡的`plugins.lua`脚本就可以删掉了，因为`lazy`会去找`plugins`文件夹下的子模块。

但是！`init.lua`中还有一些关于Vim的基本配置，这些配置如果慢慢增多，还是会导致`init.lua`混乱不堪。所以我们将关于Vim的配置一并模块化。

在`lua`文件夹下创建一个名为`vim-options.lua`的脚本，将`init.lua`中关于Vim配置的语句全部丢进去。

![](https://github.com/Deleter-D/Images/assets/56388518/ec10b4b0-ab2e-4b45-b7d3-e9014fa2594b)

同时在`init.lua`中添加一句。

```lua
require("vim-options") -- 本句需要在require("lazy")之前
```

## 代码相关插件

### LSP

先来说说什么是LSP，下面是wiki对于LSP的定义：

> The Language Server Protocol (LSP) is an open, JSON-RPC-based protocol for use between source code editors or integrated development environments (IDEs) and servers that provide "language intelligence tools": programming language-specific features like code completion, syntax highlighting and marking of warnings and errors, as well as refactoring routines. The goal of the protocol is to allow programming language support to be implemented and distributed independently of any given editor or IDE.

感觉还是有点抽象，我觉得视频中的大佬解释的非常形象，这里转述一下。

将编辑器想象成客户端，LSP想象成服务器。编辑器打开一个文件的时候，会向LSP发送一个`DID OPEN`信号，告知LSP目前打开了一个文件。当文件发生任何改动的时候，会向LSP发送一个`DID UPDATE`信号，告知LSP代码被更改过。此时LSP会返回一个JSON文件，其中包含了类似下面这样的信息，告知编辑器哪里有错误、警告等等。

```json
{
    "ERRORS": "...",
    "WARNINGS": "..."
}
```

所以，编写代码的过程中，能够有这样一个LSP提示错误和警告是非常必要的功能。

#### LSP管理

创建一个`lsp-config.lua`脚本来配置LSP。首先需要一个插件`Mason`，可以帮助我们很方便的安装各种语言的LSP、Linter、Formatter等。

> Mason官方仓库：[williamboman/mason.nvim](https://github.com/williamboman/mason.nvim)

脚本中添加如下内容。

```lua
return {
  "williamboman/mason.nvim",
  config = function()
    require("mason").setup()
  end
}
```

重启Neovim等待安装完成，输入`:Mason`即可打开`mason`的管理界面，在这里可以看到各种LSP、Linter、Formatter等。

![](https://github.com/Deleter-D/Images/assets/56388518/c3d90e3f-617c-4157-a6dc-16051ec65eb2)

#### LSP配置

除了`mason`我们还需要另一个工具`mason-lspconfig`来辅助配置`mason`的LSP。修改`lsp-config.lua`脚本如下。

> 官方仓库：[williamboman/mason-lspconfig.nvim](https://github.com/williamboman/mason-lspconfig.nvim)

```lua
return {
  {
    "williamboman/mason.nvim",
    config = function()
      require("mason").setup()
    end
  },
  {
    "williamboman/mason-lspconfig.nvim",
    config = function()
      require("mason-lspconfig").setup({
        ensure_installed = { "lua_ls", "clangd", "pyright" } -- 配置预安装的LSP服务
      })
    end
  }
}
```

> 各种编程语言对应的LSP服务名称可以参考官方仓库的README[available-lsp-servers](https://github.com/williamboman/mason-lspconfig.nvim?tab=readme-ov-file#available-lsp-servers)。这里选择预装了`lua`、`C/C++`、`python`的LSP服务。

重启Neovim等待安装，输入`:Mason`检查是否安装成果，不出意外你会得到如下所示的状态。

> 这里有一个需要注意的地方，有些LSP是基于`node.js`和`npm`包管理器的，所以请确保你的环境已经安装了`node.js`。
>
> 同时，如果你使用了代理，请给`npm`也配置代理，具体方法自行Google。

![](https://github.com/Deleter-D/Images/assets/56388518/e9fa2622-c281-41b3-bfad-08c35fe9bcad)

#### LSP客户端

接下来我们还需要用到一个叫做`nvim-lspconfig`的插件，使得Neovim能够和LSP进行通信

> 官方仓库：[neovim/nvim-lspconfig](https://github.com/neovim/nvim-lspconfig)

继续修改`lsp-config.lua`脚本如下。

```lua
return {
  {
    "williamboman/mason.nvim",
    config = function()
      require("mason").setup()
    end
  },
  {
    "williamboman/mason-lspconfig.nvim",
    config = function()
      require("mason-lspconfig").setup({
        ensure_installed = { "lua_ls", "clangd", "pyright" }
      })
    end
  },
  {
    "neovim/nvim-lspconfig",
    config = function()
      local lspconfig = require("lspconfig")
      lspconfig.lua_ls.setup({})
      lspconfig.clangd.setup({})
      lspconfig.pyright.setup({})
    end
  }
}
```

老样子，重启Neovim等待安装，然后输入`:LspInfo`查看当前LSP服务的状态。

![](https://github.com/Deleter-D/Images/assets/56388518/1e010c1b-ba90-4236-aab7-8012b90c4524)

可以看到检测到一个客户端，所配置的LSP服务叫做`lua_ls`。

然后配置一些快捷键，以便进行LSP相关的操作。

```lua
{
    "neovim/nvim-lspconfig",
    config = function()
      local lspconfig = require("lspconfig")
      lspconfig.lua_ls.setup({})
      lspconfig.clangd.setup({})
      lspconfig.pyright.setup({})

      vim.keymap.set('n', 'K', vim.lsp.buf.hover, {})
      vim.keymap.set('n', 'gD', vim.lsp.buf.declaration, {})
      vim.keymap.set('n', 'gd', vim.lsp.buf.definition, {})
      vim.keymap.set('n', 'gi', vim.lsp.buf.implementation, {})
      vim.keymap.set('n', '<C-k>', vim.lsp.buf.signature_help, {})
      vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, {})
      vim.keymap.set('n', '<leader>ca', vim.lsp.buf.code_action, {})
    end
}
```

### 代码格式化

这里用到一个名为`null-ls`的插件，通过和特定语言的formatter配合进行操作，这里的formatter通过`mason`来安装。

> 官方仓库：[nvimtools/none-ls.nvim](https://github.com/nvimtools/none-ls.nvim)

添加`none-ls.lua`配置脚本。

```lua
return {
	"nvimtools/none-ls.nvim",
	config = function()
		local null_ls = require("null-ls")
		null_ls.setup({
			sources = {
				null_ls.builtins.formatting.stylua,
				null_ls.builtins.formatting.clang_format,
				null_ls.builtins.formatting.black,
				null_ls.builtins.formatting.isort,
			},
		})

		vim.keymap.set("n", "<leader>gf", vim.lsp.buf.format, {})
	end,
}
```

这里配置了`lua`、`C/C++`和`Python`的formatter。

### 代码自动补全

关于代码自动补全，有一系列的插件，这里推荐先捋清它们之间的关系。我们先列出可能用到的插件并附上仓库链接：

1. `nvim-cmp`（[hrsh7th/nvim-cmp](https://github.com/hrsh7th/nvim-cmp)）；
2. `LuaSnip`（[L3MON4D3/LuaSnip](https://github.com/L3MON4D3/LuaSnip)）；
3. `cmp.luasnip`（[saadparwaiz1/cmp_luasnip](https://github.com/saadparwaiz1/cmp_luasnip)）；
4. `friendly-snippets`（[rafamadriz/friendly-snippets](https://github.com/rafamadriz/friendly-snippets)）；
5. `cmp.nvim.lsp`（[hrsh7th/cmp-nvim-lsp](https://github.com/hrsh7th/cmp-nvim-lsp)）。

`nvim-cmp`是一个代码补全引擎，可以根据输入来显示补全信息。`nvim-cmp`这个补全引擎只能提供代码补全的能力，它本身并没有补全的”素材“。这时就需要一些第三方插件来提高代码补全的来源，也就是“素材”（`Snippets`），并赋予这些Snippets展开的能力。

`LuaSnip`是一个`lua`写的`Snippets`引擎，它属于扩展`Snippets`的工具。是为`nvim-cmp`提供服务的。

`cmp.luasnip`则是`LuaSnip`的“素材”来源，它为`nvim-cmp`提供一系列可能的`Snippets`，然后`LuaSnip`会将这些`Snippets`展开。

`friendly-snippets`是一个针对不同编程语言的`Snippets`集合，可以将不同语言的“素材”集中在一起，并使得`LuaSnip`可以加载。

`cmp.nvim.lsp`也是一个`Snippets`来源，但可以从任何缓存中存在的LSP来获取“素材”。

解释清楚上述几个插件的关系后，就可以开始配置了。先来进行`nvim-cmp`的基本配置，初步配置文件如下所示。

```lua
return {
  "hrsh7th/nvim-cmp",
  config = function()
    local cmp = require("cmp")
    cmp.setup({
      snippet = {
        expand = function(args)
          require("luasnip").lsp_expand(args.body)
        end,
      },
      window = {
        completion = cmp.config.window.bordered(),
        documentation = cmp.config.window.bordered(),
      },
      mapping = cmp.mapping.preset.insert({
        -- ["<C-b>"] = cmp.mapping.scroll_docs(-4),
        ["<C-f>"] = cmp.mapping.scroll_docs(4),
        ["<C-Space>"] = cmp.mapping.complete(),
        ["<C-e>"] = cmp.mapping.abort(),
        ["<CR>"] = cmp.mapping.confirm({ select = true }),
      }),
      sources = cmp.config.sources({
        { name = "nvim_lsp" },
        { name = "luasnip" },
      }, {
        { name = "buffer" },
      }),
    })
  end,
}
```

接下来配置`LuaSnip`及其`Snippets`来源：`cmp.luasnip`、`friendly-snippets`。将上面对`nvim-cmp`的配置放入一个`{}`中，然后在它的上面再创建一个新的`{}`，内容如下。

```lua
{
  "L3MON4D3/LuaSnip",
  dependencies = {
    "saadparwaiz1/cmp_luasnip",
    "rafamadriz/friendly-snippets",
  },
},
```

然后需要在`nvim-cmp`的配置中添加一行。

```lua
require("luasnip.loaders.from_vscode").lazy_load()
```

重启Neovim就发现，现在已经有了一定程度的代码补全能力。

![](https://github.com/Deleter-D/Images/assets/56388518/84dee0d6-1053-4c17-ae90-a4f3e18ec6fb)

最后安装`cpm.nvim.lsp`来提高补全的质量，使得引擎能从LSP中获取`Snippets`。同样地，在上面脚本的基础上再添加一个`{}`。

```lua
{
  "hrsh7th/cmp-nvim-lsp",
},
```

同时，在`lsp-config.lua`脚本中关于`nvim-lspconfig`插件的配置中增加一项，并在所配置的每一项LSP中增加语句，内容如下。

```lua
{
    "neovim/nvim-lspconfig",
    config = function()
        local capabilities = require("cmp_nvim_lsp").default_capabilities()

        local lspconfig = require("lspconfig")
        lspconfig.lua_ls.setup({
            capabilities = capabilities,
        })
        lspconfig.clangd.setup({
            capabilities = capabilities,
        })
        lspconfig.pyright.setup({
            capabilities = capabilities,
        })

        vim.keymap.set("n", "K", vim.lsp.buf.hover, {})
        vim.keymap.set("n", "gD", vim.lsp.buf.declaration, {})
        vim.keymap.set("n", "gd", vim.lsp.buf.definition, {})
        vim.keymap.set("n", "gi", vim.lsp.buf.implementation, {})
        vim.keymap.set("n", "<C-k>", vim.lsp.buf.signature_help, {})
        vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename, {})
        vim.keymap.set("n", "<leader>ca", vim.lsp.buf.code_action, {})
    end,
},
```

再次尝试代码补全，发现补全的来源中已经多了来自LSP的内容。

![](https://github.com/Deleter-D/Images/assets/56388518/8727d832-bb7c-44dc-9c5b-e35d6e3ef1d5)

### Debugger

在安装Debugger之前，我们需要介绍一个概念叫做Debug Adapter Protocol (DAP)。也许你在之前配置LSP的时候就已经在`Mason`中见过这个缩写了。DAP是微软为VS Code开发的，是为了使得编辑器和Debugger能够顺畅的交流。视频中的大佬描述的非常形象，编辑器和Debugger就像酷酷的西部牛仔一样，每个人都有自己交流的方式，而DAP站出来对Debugger说“Let's make things easier.”。DAP提供了一套通用的API，使得不同语言的不同Debugger能够使用这一套API为编辑器提供统一的格式，而编辑器要做的只是利用DAP提供的这套统一格式而已。

其实这套概念和LSP有点类似。我们需要一个服务端，也需要一个客户端。服务端我们可以借助`Mason`来安装，客户端我们这里用到的是`nvim-dap`，是一个为Neovim实现的DAP客户端。

> 官方仓库：[mfussenegger/nvim-dap](https://github.com/mfussenegger/nvim-dap)

同时我们还需要一个叫做`nvim-dap-ui`的UI插件来辅助Debugger工作。然后进行一些配置使得UI能够根据DAP自动打开或关闭。

> 官方仓库：[rcarriga/nvim-dap-ui](https://github.com/rcarriga/nvim-dap-ui)

新建一个`debugger.lua`的配置文件。

```lua
return {
  "mfussenegger/nvim-dap",
  dependencies = {
    "rcarriga/nvim-dap-ui",
  },
  config = function()
    local dap = require("dap")
    local dapui = require("dapui")
    dapui.setup()

    dap.listeners.before.attach.dapui_config = function()
      dapui.open()
    end
    dap.listeners.before.launch.dapui_config = function()
      dapui.open()
    end
    dap.listeners.before.event_terminated.dapui_config = function()
      dapui.close()
    end
    dap.listeners.before.event_exited.dapui_config = function()
      dapui.close()
    end

    vim.keymap.set("n", "<leader>b", dap.toggle_breakpoint, {})
    vim.keymap.set("n", "<F5>", dap.continue, {})
    vim.keymap.set("n", "<F6>", dap.terminate, {})
    vim.keymap.set("n", "<F7>", dap.restart, {})
    vim.keymap.set("n", "<F9>", dap.step_into, {})
    vim.keymap.set("n", "<F10>", dap.step_out, {})
    vim.keymap.set("n", "<F12>", dap.step_over, {})
  end,
}
```

除了上面的配置，我们还需要针对你要debug的语言安装对应的DAP，并在`nvim-dap`中配置。这里以C/C++为例，使用codelldb作为DAP。首先在`Mason`中安装codelldb，然后对`nvim-dap`进行如下配置。

```lua
local install_root_dir = vim.fn.stdpath("data") .. "/mason"
local extension_path = install_root_dir .. "/packages/codelldb/extension/"
local codelldb_path = extension_path .. "adapter/codelldb"
dap.adapters.codelldb = {
  type = "server",
  port = "${port}",
  executable = {
    command = codelldb_path,
    args = { "--port", "${port}" },
  },
}
dap.configurations.c = {
  {
    name = "Launch file",
    type = "codelldb",
    request = "launch",
    program = function()
      return vim.fn.input("Path to executable: ", vim.fn.getcwd() .. "/", "file")
    end,
    cwd = "${workspaceFolder}",
  },
}
dap.configurations.cpp = dap.configurations.c
```

用一个简单的C程序测试一下。

![](https://github.com/Deleter-D/Images/assets/56388518/343250c6-3db5-474f-870d-1d1d0565089c)

## 美化插件

### lualine

`lualine`是一个非常美观的底部状态栏。安装它只需要在`plugins`下添加一个`lua`脚本即可。

> 官方仓库：[nvim-lualine/lualine.nvim](https://github.com/nvim-lualine/lualine.nvim)

```lua
return {
  'nvim-lualine/lualine.nvim',
  dependencies = { 'nvim-tree/nvim-web-devicons' },
  config = function()
    require('lualine').setup({
      options = {
        theme = 'dracula'
      }
    })
  end
}
```

重启Neovim，然后就得到了如下图所示的效果。

![](https://github.com/Deleter-D/Images/assets/56388518/86ec9d67-d04f-4226-8c98-31a075397739)

~~好好好，越来越像VS Code了。~~

### Telescope-ui-select

该插件是`telescope`插件的一个扩展，可以配合`vim.lsp.buf.code_action()`使用，使其以一种悬浮窗的形式显示，而不是底部命令栏。

> 官方仓库：[nvim-telescope/telescope-ui-select.nvim](https://github.com/nvim-telescope/telescope-ui-select.nvim)

![](https://github.com/Deleter-D/Images/assets/56388518/a702981b-3793-424a-97d9-c13717634583)

### alpha-nvim

该插件可以提供一个类似于VS Code的欢迎界面，配置脚本如下。

> 官方仓库：[goolord/alpha-nvim](https://github.com/goolord/alpha-nvim)

```lua
return {
	"goolord/alpha-nvim",
	dependencies = {
		"nvim-tree/nvim-web-devicons",
	},
	config = function()
		local alpha = require("alpha")
		local dashboard = require("alpha.themes.dashboard")

		dashboard.section.header.val = {
			[[                               __                ]],
			[[  ___     ___    ___   __  __ /\_\    ___ ___    ]],
			[[ / _ `\  / __`\ / __`\/\ \/\ \\/\ \  / __` __`\  ]],
			[[/\ \/\ \/\  __//\ \_\ \ \ \_/ |\ \ \/\ \/\ \/\ \ ]],
			[[\ \_\ \_\ \____\ \____/\ \___/  \ \_\ \_\ \_\ \_\]],
			[[ \/_/\/_/\/____/\/___/  \/__/    \/_/\/_/\/_/\/_/]],
		}

		dashboard.section.buttons.val = {
			dashboard.button("e", "  New file", ":ene <BAR> startinsert <CR>"),
			dashboard.button("q", "󰅚  Quit NVIM", ":qa<CR>"),
		}

		alpha.setup(dashboard.opts)
	end,
}
```

`header`和`buttons`两项配置是最具有可玩性的，官方仓库中有许多用户提供了他们自定义的配置文件。上面的最简配置文件效果如下图所示。

![](https://github.com/Deleter-D/Images/assets/56388518/5beb0355-dae8-4dc0-aa08-b43e4fb349cc)

## 快捷键列表

经过上面的一系列配置，我们就得到了下表中的功能。

### 编辑器相关

| 快捷键       | 功能         |
| ------------ | ------------ |
| `<C-p>`      | 文件模糊搜索 |
| `<leader>fg` | 字符串搜索   |
| `<C-b>`      | 打开文件树   |

### LSP相关

| 快捷键       | 功能               |
| ------------ | ------------------ |
| `K`          | 悬浮窗显示函数信息 |
| `gD`         | 转到声明           |
| `gd`         | 转到定义           |
| `gi`         | 转到实现           |
| `<C-k>`      | 函数签名帮助信息   |
| `<leader>rn` | 重命名             |
| `<leader>ca` | 代码动作           |
| `<leader>gf` | 代码格式化         |

### DAP相关

| 快捷键      | 功能              |
| ----------- | ----------------- |
| `<leader>b` | 添加断点          |
| `<F5>`      | 开始/继续执行     |
| `<F6>`      | 终止debug会话     |
| `<F7>`      | 重新开启debug会话 |
| `<F9>`      | 单步执行          |
| `<F10>`     | 单步跳出          |
| `<F12>`     | 逐过程执行        |

# 写在最后

上面讲了这么多插件，但依然希望你不要教条的按照这个教程来做，重要的是学会配置的方法，同时善用Google寻找更加优秀的插件。当然，最推荐的还是直接抄作业，在巨人的肩膀上进行自定义。
