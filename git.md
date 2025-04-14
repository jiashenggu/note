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
