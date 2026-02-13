# 单幅图像去雾的深度学习：综述与比较分析

## 摘要

单幅图像去雾仍然是计算机视觉和大气光学中一个具有挑战性的病态问题，旨在从有雾观测中恢复清晰场景。本文全面回顾了基于深度学习的去雾方法，涵盖了从早期的基于CNN的物理模型估计器到最近的生成扩散模型。我们系统地分析了大气散射模型、网络架构以及损失函数设计。我们在合成数据集和真实世界基准上的评估表明，物理引导的混合架构在保持物理合理性的同时，实现了优越的感知质量。我们讨论了关键挑战，包括非均匀雾、天空区域处理以及实时部署限制。未来方向强调了神经大气散射先验以及对未见雾浓度的零样本适应。

**关键词：** 图像去雾，大气散射，深度学习，GAN，扩散模型，感知质量

---

## 1. 引言

大气雾通过散射和吸收光线显著降低了户外图像质量，给自动驾驶、遥感和监控系统中的高层视觉任务带来了严峻挑战。与多图像或基于偏振的方法不同，单幅图像去雾必须从单一观测中解决场景辐射度与大气光之间的根本模糊性。

McCartney (1976) 提出并由 Narasimhan 和 Nayar (2000, 2002) 改进的大气散射模型将雾的形成过程形式化：

$$
I(x) = J(x)t(x) + A(1 - t(x))
$$

其中：
- $I(x)$：观测到的有雾图像
- $J(x)$：场景辐照度（待恢复的清晰图像）
- $A$：全球大气光
- $t(x)$：介质透射率图，$t(x) = e^{-\beta d(x)}$
- $\beta$：大气散射系数
- $d(x)$：场景深度

### 1.1 问题描述

给定 $I(x)$，去雾算法通过估计透射率图 $t(x)$ 和大气光 $A$ 来估算 $J(x)$，通常如下所示：

$$
J(x) = \frac{I(x) - A}{t(x)} + A
$$

这带来了三个核心挑战：
1. **病态性**：无穷多 $(J, t, A)$ 的组合会产生相同的 $I$
2. **非均匀雾**：空间变化的 $\beta$ 违反了均匀介质的假设
3. **颜色失真**：天空区域过度饱和，浓雾区域恢复不足

### 1.2 去雾方法论的演变

**基于先验的时代 (2009-2015)：** He 等人 (2009) 提出的暗通道先验（DCP）基于一种统计观察——即清晰图像块在至少一个颜色通道中包含低强度像素——占据了主导地位。其局限性包括天空区域的伪影和固定的先验假设。

**基于 CNN 的时代 (2016-2018)：** 透射率图的端到端学习（DehazeNet, MSCNN）用数据驱动的特征提取取代了手工设计的先验。

**基于 GAN 的时代 (2019-2021)：** 条件生成对抗网络（CycleGAN, Pix2Pix）实现了非配对训练和感知质量优化，尽管通常会牺牲物理一致性。

**Transformer/扩散模型时代 (2022-至今)：** 自注意力机制和扩散概率模型捕捉长距离依赖和生成先验，以实现逼真的细节恢复。

---

## 2. 相关工作

### 2.1 基于物理模型的方法

早期的深度学习方法集中于物理参数的估计：

| Method | Architecture | Target | Key Innovation |
|--------|-------------|---------|----------------|
| **DehazeNet** (2016) | CNN | $t(x)$ | 用于有界透射率的 BReLU 激活函数 |
| **MSCNN** (2016) | Multi-scale CNN | $t(x)$ | 由粗到细的透射率估计 |
| **AOD-Net** (2017) | Light CNN | $K(x)$ | 重构公式 $J(x) = K(x)I(x) - K(x) + b$ |
| **DCPDN** (2018) | Two-stream CNN | $t(x), A$ | 基于物理约束的联合估计 |

**局限性：** $t(x)$ 和 $A$ 估计中的累积误差通过除法运算传播至最终的 $J(x)$，放大了远距离区域的噪声。

### 2.2 端到端图像翻译

绕过显式的物理参数估计：

| Method | Framework | Training | Loss Functions |
|--------|-----------|----------|----------------|
| **Pix2Pix** (2017) | cGAN | Paired (成对) | $\mathcal{L}_{GAN} + \mathcal{L}_{L1}$ |
| **CycleGAN** (2017) | Cycle-consistent GAN | Unpaired (不成对) | $\mathcal{L}_{cycle} + \mathcal{L}_{identity}$ |
| **Pix2Pix-Turbo** (2024) | One-step GAN (单步 GAN) | Paired (成对) | Adversarial + Perceptual + Distillation (对抗 + 感知 + 蒸馏) |
| **GridDehazeNet** (2019) | Attention CNN | Paired (成对) | Multi-scale grid network (多尺度网格网络) |

**Pix2Pix-Turbo 架构：**
```python
class Pix2PixTurbo(nn.Module):
    """
    One-step image-to-image translation with encoder-decoder
    and skip connections for dehazing task
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder: Extract multi-scale features
        self.enc1 = ConvBlock(in_channels, 64, stride=1)
        self.enc2 = ConvBlock(64, 128, stride=2)
        self.enc3 = ConvBlock(128, 256, stride=2)
        self.enc4 = ConvBlock(256, 512, stride=2)
        
        # Bottleneck with residual blocks
```

```python
        self.bottleneck = nn.Sequential(
            *[ResBlock(512) for _ in range(4)]
        )
        
        # 带有跳跃连接的解码器
        self.dec4 = UpConvBlock(512, 256)
        self.dec3 = UpConvBlock(512, 128)  # 256+256 跳跃连接
        self.dec2 = UpConvBlock(256, 64)   # 128+128 跳跃连接
        self.dec1 = ConvBlock(128, out_channels)  # 64+64 跳跃连接
        
    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # 瓶颈层
        b = self.bottleneck(e4)
        
        # 带有跳跃连接的解码器
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        return torch.tanh(d1) + x  # 残差学习
```