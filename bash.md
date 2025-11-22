## show git branch in bash

把颜色加回来即可，下面给出两种常用做法，你挑一条粘到 `~/.bashrc` 里（放在 `source ~/.git-prompt.sh` 之后）再 `source ~/.bashrc` 就好了。

------------------------------------------------
1. 最简单：把原来的彩色 PS1 捡回来  
先注释掉或删掉我那行纯文本的 `export PS1=...`，然后在 `~/.bashrc` 末尾加：
```bash
curl -Lo ~/.git-prompt.sh \
  https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
```
```bash
# 保留系统原来的彩色提示符，只在末尾追加分支名
source ~/.git-prompt.sh
export PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]$(__git_ps1 " \[\033[01;31m\](%s)\[\033[00m\]")\$ '
```

（这是 Debian/Ubuntu 默认的彩色模板，红括号里放分支名。）

------------------------------------------------
2. 自己拼一个彩色 PS1  
如果你想完全自定义，可写成：

```bash
source ~/.git-prompt.sh
# 绿色用户@主机 : 蓝色路径 红色(分支) 白色$
export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[1;31m\]$(__git_ps1 " (%s)")\[\e[0m\]\$ '
```

------------------------------------------------
3. 让 `__git_ps1` 自己带颜色  
`git-prompt.sh` 还提供了变量，可让分支名高亮而不用手动写转义：

```bash
source ~/.git-prompt.sh
GIT_PS1_SHOWCOLORHINTS=1        # 启用内置颜色
# 必须选用 \w 这类提示串，并且把 __git_ps1 放在最后
export PS1='\u@\h:\w$(__git_ps1 " (%s)")\$ '
```

（注意：这个变量**只在** `PROMPT_COMMAND` 模式或 `__git_ps1` 放在末尾时生效，详情看 `git-prompt.sh` 头部注释。）

------------------------------------------------
一句话  
**“没颜色”是因为 PS1 被覆盖成纯文本了；把颜色转义序列加回去，或者打开 `GIT_PS1_SHOWCOLORHINTS`，就又有颜色了。**


`set -euxo pipefail` 是 Bash 脚本中常用的组合命令，用于增强脚本的健壮性和调试能力。具体作用如下：

---

### 1. `-e`（`errexit`）
- **作用**：当任意命令执行失败（返回非零状态码）时，立即终止脚本。
- **示例**：
  ```bash
  # 脚本会在此处终止，因为 ls 一个不存在的文件会失败
  ls non_existent_file
  echo "这行不会执行"
  ```

---

### 2. `-u`（`nounset`）
- **作用**：当使用未定义的变量时，报错并终止脚本。
- **示例**：
  ```bash
  # 若未定义 $MY_VAR，脚本会在此处终止
  echo $MY_VAR
  ```

---

### 3. `-x`（`xtrace`）
- **作用**：打印每个执行的命令及其参数，便于调试。
- **示例**：
  ```bash
  # 终端会输出：+ echo 'Hello World'
  echo "Hello World"
  ```

---

### 4. `-o pipefail`
- **作用**：确保管道命令的退出状态是最后一个非零状态码（默认仅返回最后一个命令的退出状态）。
- **示例**：
  ```bash
  # 假设 command1 失败，command2 成功
  command1 | command2
  # 默认返回 command2 的退出状态（0），但启用 pipefail 后会返回 command1 的退出状态（非零）
  ```

---

### 组合效果
- **增强安全性**：避免脚本在错误状态下继续执行。
- **调试友好**：通过 `-x` 显示详细执行过程。
- **管道健壮性**：严格捕获管道中的任何错误。

---

### 常见用法
```bash
#!/bin/bash
set -euxo pipefail

# 脚本内容
echo "Start"
your_command_here
```

---

### 注意事项
- 若需忽略某条命令的失败，可使用 `|| true`：
  ```bash
  # 即使 command 失败，脚本仍会继续执行
  command || true
  ```
- 临时禁用选项：使用 `set +<option>`（如 `set +x` 关闭调试）。
