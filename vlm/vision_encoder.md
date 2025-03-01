# DINO和CLIP可能的互补关系
DINO擅长图像之间的联系

CLIP擅长图文之间的联系


SigLIP-2来了：

SigLIP是CLIP最重要的升级版之一，之前Sam和大家分享过，它采用的Sigmoid二分类loss使每个样本的loss计算和batch内其他样本解耦，方便多机大batch训练，已经成为诸多多模态大模型选用的视觉encoder


SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

主要的改进在loss方面，添加了3个辅助loss以增进相应能力：

1️⃣文本生成能力：LooCa中的captioning loss，即接一个轻量级decoder，训captioning、dense captioning和ref expressing等图生文任务

2️⃣局部表示能力/dense prediction性能：用截断梯度的EMA模型当teacher，在训练过程最后20%的steps中进行SLIC的local-global自蒸馏和TIPS的MAE训练；

3️⃣轻量级模型能力：咱们上一期分享的ACED主动学习蒸馏，即只在部分“学生不会 &教师轻松完成 ”的高质量样本上进行训练。🚀效果：分类、检索等整体任务和detection、segmention、locaization等dense/local任务上均有显著提升，base尺寸下ImageNet-1k val比SigLIP第一代增长约2个百分点之多，相当亮眼

https://www.zhihu.com/pin/1877855780885123072
