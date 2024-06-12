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
