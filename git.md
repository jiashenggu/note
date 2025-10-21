### show git branch in bash

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


# [Use SSH, not HTTPS](https://mkyong.com/git/github-keep-asking-for-username-password-when-git-push/])
# git config
```
git config --global user.name "jiashenggu"
git config --global user.email "jiashengguwen@gmail.com"
```
```
git config --list
```

# generate ssh key
```
ssh-keygen -t ed25519 -C "your_email@example.com"
```
# Compare two repos
```bash
git remote add -f b path/to/repo_b.git
git remote update
git diff master remotes/b/master
git remote rm b
```

# add ssh key
```
ssh -v -T git@github.com
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

~/.ssh/config:

Host github.com
  HostName github.com
  User git
  IdentityFile  ~/.ssh/id_ed25519
  IdentitiesOnly yes
```
## 显示远程分支
https://www.freecodecamp.org/chinese/news/git-list-remote-branches/

git branch -r
git ls-remote --heads
git ls-remote
git show-branch -r
git branch -vv
git branch -a


# [git commit 撤销的三种方法](https://blog.csdn.net/weixin_45678402/article/details/134663161)


# Github进行fork后如何与原仓库同步：重新fork很省事，但不如反复练习版本合并
https://github.com/selfteaching/the-craft-of-selfteaching/issues/67

```bash
git remote add upstream $url
```

```bash
git fetch upstream  
git merge upstream/*
```

# 如何将上游 (upstream) 新建分支 (origin 没有) 导入到 origin 中？

可以使用以下命令将上游分支导入到本地和 origin:
## 0. 添加上游仓库
如果需要权限，url应该是ssh的git@github.com
```bash
git remote add upstream $url
```

## 1. 从上游仓库拉取最新的代码并创建一个新的本地分支：

```bash
git fetch upstream  
git checkout -b new_branch upstream/new_branch
```

## 2. 然后，将本地分支推送到远程仓库：

```bash
git push upstream new_branch
```

这样，你就可以在本地和远程仓库中创建一个新的分支，并将上游分支的代码导入到该分支中。请确保在执行这些命令之前，你已经将上游仓库添加为远程仓库，并且你有权限访问它。

## 相关问题

### git 设置上游分支

在 Git 中设置上游分支的方法如下：

1. 使用命令 `git remote -v` 检查您的远程仓库的 URL。
2. 使用命令 `git branch -a` 查看所有的远程分支。
3. 使用命令 `git branch --set-upstream-to=<远程分支>` 设置当前分支的上游分支。

例如，如果您想将当前的本地分支设置为 `upstream/main` 的远程分支，可以使用以下命令：
```bash
git branch --set-upstream-to=upstream/main
```
这样，你就成功设置当前分支的上游分支。
```bash
git push --set-upstream upstream dev
```
这个命令会将本地的 dev 分支推送到远程仓库的 dev 分支，并将该 dev 分支设置为该本地 dev 分支的上游分支。此后，只需执行简便的 `git push` 或 `git pull` 命令即可推送或更新。


### 通过环境变量临时指定ssh key
```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/custom_id_rsa' git clone git@github.com:user/repo.git
```
