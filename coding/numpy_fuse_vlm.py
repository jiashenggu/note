# 题目描述：
# 实现函数 fuse_vlm(image: np.ndarray, prompt: str, seed: int = 42) -> np.ndarray，要求完成以下步骤：
# 图像特征提取：
# 输入图像为 NumPy 数组，形状为 (H, W, C)，像素值范围为 [0, 255]。
# 计算图像的平均颜色，即对高度和宽度求均值，得到一个长度为 C 的特征向量。
# 对该向量进行 L2 归一化（若范数为 0，则保持原值）。
# 文本特征提取：
# 输入文本 prompt 按空格拆分成 token 列表。
# 对每个 token，计算其字符数；然后计算所有 token 字符数的平均值，得到一个标量特征。
# 将该标量转换为 NumPy 数组（形状 (1,)），并进行归一化（即除以其绝对值，若为 0 则不做归一化）。
# 多模态特征融合：
# 将图像特征（长度 C）与文本特征（长度 1）拼接成一个长度为 C+1 的联合特征向量。
# 使用给定随机种子 seed 初始化一个线性变换（权重矩阵 W 大小 (C+1) x (C+1) 和偏置向量 b 大小 (C+1,)），计算： final_feature=W×joint_feature+b
# 返回：
# 返回最终的融合特征向量 final_feature。
# 要求：
# 仅使用 NumPy 完成计算。
# 在 if name == "__main__": 代码块中提供测试代码，生成一个随机 4x4 的 RGB 图像（dtype=np.uint8）和示例文本 "OpenPI is amazing"，并打印最终的融合特征向量。

import numpy as np

def fuse_vlm(image: np.ndarray, prompt: str, seed: int = 42) -> np.ndarray:
    # 图像特征提取
    # 计算平均颜色 (C,)
    image_feature = np.mean(image, axis=(0, 1))
    
    # L2归一化
    norm = np.linalg.norm(image_feature)
    if norm > 0:
        image_feature = image_feature / norm
    
    # 文本特征提取
    tokens = prompt.split()
    if len(tokens) == 0:
        text_feature = np.array([0.0])
    else:
        # 计算每个token的字符数并取平均
        avg_chars = np.mean([len(token) for token in tokens])
        text_feature = np.array([avg_chars])
        # 归一化
        abs_val /= np.abs(text_feature)
        if abs_val > 0:
            text_feature = text_feature / abs_val
    
    # 多模态特征融合
    joint_feature = np.concatenate([image_feature, text_feature])
    
    # 初始化线性变换
    np.random.seed(seed)
    C_plus_1 = joint_feature.shape[0]
    W = np.random.randn(C_plus_1, C_plus_1)
    b = np.random.randn(C_plus_1)
    
    # 计算最终特征
    final_feature = W @ joint_feature + b
    
    return final_feature

if __name__ == "__main__":
    # 生成随机4x4 RGB图像
    np.random.seed(42)
    test_image = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
    test_prompt = "OpenPI is amazing"
    
    # 调用函数
    feature = fuse_vlm(test_image, test_prompt)
    
    # 打印结果
    print("Final fused feature vector:")
    print(feature)
    print("Shape:", feature.shape)
