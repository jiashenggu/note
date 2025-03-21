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
