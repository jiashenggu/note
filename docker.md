上传 Docker 镜像（即“上传 Docker”）的核心流程只有三步：先给本地镜像打上“仓库地址/用户名/镜像名:标签”格式的新标签，再登录目标镜像仓库，最后执行 docker push。下面给出最常见的两种场景——Docker Hub 与阿里云 ACR——的完整命令示例，照着复制即可用。

-------------------------------------------------
一、上传至 Docker Hub（官方公共仓库）
1. 登录  
   docker login -u <DockerHub 用户名>  
   输入密码或访问令牌后看到 Login Succeeded 即可 。

2. 打标签  
   # 如果本地镜像叫 myapp:v1  
   docker tag myapp:v1 <用户名>/myapp:v1  
   # 示例：docker tag myapp:v1 jack/myapp:v1 

3. 推送  
   docker push <用户名>/myapp:v1  
   成功后任何人都能 docker pull <用户名>/myapp:v1 拉取 。

-------------------------------------------------
二、上传至阿里云容器镜像服务（ACR 个人版）
1. 登录  
   docker login --username=<阿里云UID> crpi-<实例ID>.<地域>.personal.cr.aliyuncs.com  
   输入 Registry 登录密码，显示 Login Succeeded 。

2. 打标签  
   docker tag <镜像ID> crpi-<实例ID>.<地域>.personal.cr.aliyuncs.com/<命名空间>/<仓库名>:<版本>  
   示例：  
   docker tag 3a8f crpi-abc123.cn-hangzhou.personal.cr.aliyuncs.com/myns/myapp:1.0 

3. 推送  
   docker push crpi-abc123.cn-hangzhou.personal.cr.aliyuncs.com/myns/myapp:1.0  
   推送完成可在阿里云控制台“镜像版本”页面看到 。

-------------------------------------------------
三、常见注意事项
- 镜像名称必须含仓库地址（Docker Hub 可省略 hub.docker.com，其余私有库必须写全）。  
- 标签写错会导致 push 失败或推到陌生仓库，务必核对。  
- 大镜像首次上传较慢，Docker 会自动分层、断点续传 。  
- CI/CD 里用 --password-stdin 或 AccessToken 避免明文密码，例如：  
  echo $TOKEN | docker login -u <用户名> --password-stdin

照以上步骤操作即可完成“上传 Docker”。

## docker run template for using gpu
```bash
docker rm -f foundationpose
DIR=$(pwd)/../
xhost +  && docker run --runtime nvidia --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE shingarey/foundationpose_custom_cuda121 bash -c "cd $DIR && bash"
```
