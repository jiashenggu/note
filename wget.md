# civitai
```bash
wget -O .safetensors "https://civitai.com/api/download/models/350144?&token=959e323291458f321bbcb0ac94a15edd" --content-disposition
```
https://www.jinhang.work/tech/download-shared-files-using-wget-or-curl/

此处的文件是指公开的文件，不需要输入密码也不需要登录Google drive即可获取的文件。

# default:
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
