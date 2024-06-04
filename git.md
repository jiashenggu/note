# git config
```
git config --global user.name "jiashenggu"
git config --global user.email "jiashengguwen@gmail.com"
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
