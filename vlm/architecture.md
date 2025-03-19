只有qwen1用了cross attention在merger/adapter里

qwen2/2.5都用的MLP

qwen2的vision encoder用了2d rope

qwen2.5的vision encoder用了window attention

qwen2.5的merger，将空间上相邻的四个patch特征拼接在一起通过双层MLP

## [qwenvl 以及qwenvl 2 模型架构理解](https://blog.csdn.net/Sansipi/article/details/144402848)
# [24年下半年较新的VLM架构](https://zhuanlan.zhihu.com/p/11503653276)

## AnyRes 
![image](https://github.com/user-attachments/assets/f4aa3c55-58ad-4cc4-96b1-1b35045e2ad1)
AnyRes 的具体步骤如下：

将高分辨率的图像分割成块，块的大小取决于视觉编码器能够处理的大小（例如 CLIP-ViT-L/14 可以处理的分辨率为 224*224）。视觉编码器单独处理每一块。
同时，将高分辨率的图像 resize 成视觉编码器能够处理的大小并使用视觉编码器进行编码
将上面两步的结果拼接在一起作为视觉特征
https://zhuanlan.zhihu.com/p/696402890

