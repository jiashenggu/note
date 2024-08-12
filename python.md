```python
import re

# 文件名列表示例
filenames = [f"file{num}" for num in range(1, 101)]

# 自然排序的辅助函数
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# 对文件名进行自然排序
sorted_filenames = sorted(filenames, key=natural_sort_key)

# 输出排序后的文件名
for filename in sorted_filenames:
    print(filename)
```
