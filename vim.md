:s（substitute）命令用来查找和替换字符串。语法如下：
```
:{作用范围}s/{目标}/{替换}/{替换标志}
```
例如 :%s/foo/bar/g 会在全局范围(%)查找 foo 并替换为 bar，所有出现都会被替换（g）。

https://harttle.land/2016/08/08/vim-search-in-file.html
