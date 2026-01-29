## Proxy will affect many web applications!!!
https://www.bilibili.com/read/cv23352904/

# ~/.bashrc （设置环境，负责一般意义上的联网...）
```
export http_proxy=http://输入你的网络代理地址/
export https_proxy=https://输入你的网络代理地址/
export HTTP_PROXY=http://输入你的网络代理地址/
export HTTPS_PROXY=https://输入你的网络代理地址/
```
```
alias proxy_on="export http_proxy=http://;export https_proxy=https://"
alias proxy_off="unset http_proxy https_proxy"
```

# ~/.condarc （负责conda install之类的联网...一般来说在这个位置）
```
proxy_servers:
  http: http://你的网络代理地址
  https: https://你的网络代理地址
```

# git联网
```
git config --global http.proxy 你的网络代理地址
```

还有就是下载各种包需要用到的pip install也需要联网，他们在文件launch.py里：

第一个需要改的地方在def run_pip里面， 129行左右
```
return run(f'"{python}" -m pip --proxy=你的代理地址 {args} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")
```
第二个需要改的地方是def prepare_environment里下载torch这里，228行左右
```
torch_command = os.environ.get('TORCH_COMMAND', "pip --proxy=你的代理地址 install torch=`
```

# docker
```json
{
 "proxies": {
   "default": {
     "httpProxy": "http://",
     "httpsProxy": "http://"
   }
 }
}
```
配置 Docker proxy：


**步骤 1: 创建代理配置**
```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/proxy.conf << 'EOF'
[Service]
Environment="HTTP_PROXY=http://10.18.66.131:3128"
Environment="HTTPS_PROXY=http://10.18.66.131:3128"
Environment="NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
EOF
```

**步骤 2: 重启 Docker**
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

**步骤 3: 重新构建**
```bash
cd /home/gear/Documents/workspace/code/label-studio
bash quick_build_push.sh
```

# use os to set proxy
```python
import os
os.environ["http_proxy"] = '100.66.28.72:3128'
os.environ["https_proxy"] = '100.66.28.72:3128'
```
