# 假设数据盘已挂载到 /mnt/data
chmod +x add-dev-user.sh del-dev-user.sh

# 把 /root 也迁走，并创建两个开发账号
./add-dev-user.sh alice 2001 "ssh-rsa AAAAB3NzaC1yc2EA..." /mnt/data/home "Alice@2025!" true
./add-dev-user.sh bob   2002 "ssh-rsa AAAAB3NzaC1yc2EA..." /mnt/data/home "Bob@2025!" false

# 验证
su - alice -c 'echo $HOME'   # → /mnt/data/home/alice
su - root -c 'echo $HOME'    # → /mnt/data/home/root

# users.conf
alice:2001:ssh-rsa AAAAB3NzaC1yc2EA...:Alice@2025!
bob:2002:ssh-rsa AAAAB3NzaC1yc2EA...:Bob@2025!

# batch script
```shell
#!/bin/bash
MIGRATE=true
while IFS=: read -r u uid key pwd; do
    ./add-dev-user.sh "$u" "$uid" "$key" /mnt/data/home "$pwd" "$MIGRATE"
    MIGRATE=false   # 只有第一行顺带迁 /root
done < users.conf
```
