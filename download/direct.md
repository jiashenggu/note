# magnet
Downloading Magnet Links with aria2
aria2 is a lightweight multi-protocol and multi-source command-line download utility. It supports HTTP/HTTPS, FTP, SFTP, BitTorrent, and Metalink.

Installing aria2
First, open your terminal and run the following commands to update your package list and install aria2:
```
sudo apt update
sudo apt install aria2
```
Using aria2c to Download Magnet Links
After the installation is complete, you can use the aria2c command to download a magnet link. Here is an example command:
```
aria2c 'magnet:?xt=urn:btih:c0e342ae56775'
```
Downloading Magnet Links with transmission-cli
transmission-cli is the command-line version of the BitTorrent client Transmission. It can also be used to download magnet links.

Installing transmission-cli
Similarly, open your terminal and run the following commands to update your package list and install transmission-cli:
```
sudo apt update
sudo apt install transmission-cli
```
Using transmission-cli to Download Magnet Links
Once the installation is complete, you can use the transmission-cli command to download a magnet link. Here is an example command:
```
transmission-cli 'magnet:?xt=urn:btih:c0e342ae56775'
```
[Download Magnet Links on Linux Command Line](http://eadst.com/blog/247)

# wget
https://www.jinhang.work/tech/download-shared-files-using-wget-or-curl/

此处的文件是指公开的文件，不需要输入密码也不需要登录Google drive即可获取的文件。

## default:
-b, –background 启动后转入后台执行

-t, –tries=NUMBER 设定最大尝试链接次数(0 表示无限制)

-c, –continue 接着下载没下载完的文件


```bash
wget -t 0 -c -b
```


1. 下载小文件
```
FILEID=
FILENAME=
wget --no-check-certificate ‘https://docs.google.com/uc?export=download&id=${FILEID}’ -O ${FILENAME}
``` 

替换对应的FILEID即可，FILENAME自己命名。
 FILEID是Google drive公开分享的链接中ID后面的，例如：
```
https://drive.google.com/open?id=ThisIsFileID
```

如果下载中断了，想要继续下载，可以在wget后面添加 -c 参数

2. 下载大文件
因为Google drive的大文件，无法通过安全查杀

```bash
FILEID=
FILENAME=
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=${FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${FILENAME} && rm -rf /tmp/cookies.txt
```
同样替换其中的FILEID和FILENAME即可。注意FILEID有两处。

### Download SAM dataset
```bash
while read file_name cdn_link; do wget -O "$file_name" "$cdn_link"; done < links.txt
```

Using aria2 supports multi-threaded downloading and resuming.
Remove the header line of the links file provided by meta, then run:

```bash
while read file_name cdn_link; do aria2c -x4 -c -o "$file_name" "$cdn_link"; done <  file_list.txt
```
