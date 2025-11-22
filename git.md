## 在 Ubuntu 下把 **Meld 配置成 Git 的默认 merge 工具**只需 3 步
以后执行 `git merge` 出现冲突时，一条 `git mergetool` 就能弹出 Meld 的三向合并界面，边看图边点按钮解决冲突。

---

### 1. 安装 Meld（如已装可跳过）
```bash
sudo apt update
sudo apt install meld
```
安装完确认路径  
```bash
which meld          # 应返回 /usr/bin/meld
```

---

### 2. 一次性写入 Git 全局配置
```bash
git config --global merge.tool meld
git config --global mergetool.meld.path /usr/bin/meld
git config --global mergetool.meld.cmd \
  'meld $LOCAL $BASE $REMOTE --output=$MERGED --auto-merge'
```
说明  
- `$LOCAL`  当前分支版本  
- `$BASE`   双方共同祖先  
- `$REMOTE` 要合并进来的版本  
- `$MERGED` 解决后保存的结果文件  
- `--auto-merge` 让 Meld 先自动合并非冲突部分，你只处理剩余冲突 。

---

### 3. 正常使用流程
```bash
git merge feature-x        # 或 pull、cherry-pick 等
# 若有冲突，执行：
git mergetool
```
- Meld 会依次弹出每个冲突文件的三向面板。  
- 你在界面里点箭头或手动改完，保存并关闭窗口 → Git 自动标记为“已解决”。  
- 全部结束后  
```bash
git commit                 # 完成合并提交
```

---

### 可选增强
- 避免每次询问  
```bash
git config --global mergetool.prompt false
```
- 保留/丢弃 `.orig` 备份  
```bash
git config --global mergetool.keepBackup false
```

至此，Ubuntu 下 Git + Meld 可视化合并就配置好了，与 IDE 里图形化解决冲突的体验一致 。


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


7315b-5c8cd
25ab2-f9386
01c35-c467c
37a83-be5f6
d5ff5-27dd9
a07d5-c8814
bbf2e-d980f
67f20-cc013
fca32-bb910
1398e-09088
a88a5-39a9f
0b083-1ca35
e895c-0082f
7298e-02d48
a8c2e-1e2a8
967d3-4fb16
github
