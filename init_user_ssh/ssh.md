
adduser.sh
```
#!/usr/bin/env bash

set -e
set -x

username=${1:-""}
uid=${2:-""}
ssh_key_pub=${3:-""}
if [ "${username}" == "" ]; then
  echo "need username"
  exit 1
fi
adduser "${username}" --uid ${uid}  --disabled-password
mkdir -p /home/${username}/.ssh
echo "${ssh_key_pub}" >> /home/${username}/.ssh/authorized_keys
chown -R ${username}:${username} /home/${username}
chmod 700 /home/${username}/.ssh
chmod 600 /home/${username}/.ssh/authorized_keys
usermod -aG sudo ${username}
```


use an already existed directory as home
```
useradd <username> --home-dir .../<username> --no-create-home --uid 1234 --non-unique --shell /bin/bash
passwd <username> 

vi /etc/group  # add user to group
sudo visudo  # add user as sudoers
sudo chown -R <newowner>:<newgroup> <directory>
```

```
userdel $USER_NAME       #只是删除了这个用户，没删home目录
```

```
ssh -v -T git@github.com
eval "$(ssh-agent -s)"
ssh-add {$HOME}/.ssh/my_id_rsa


~/.ssh/config:

Host github.com
  HostName github.com
  User git
  IdentityFile {$HOME}/.ssh/my_id_rsa
  IdentitiesOnly yes
```
