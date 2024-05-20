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

