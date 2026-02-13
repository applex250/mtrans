# Deep Learning for Single Image Dehazing: A Survey and Comparative Analysis

## Abstract

Single image dehazing remains a challenging ill-posed problem in computer vision and atmospheric optics, aiming to restore clear scenes from hazy observations. This paper presents a comprehensive review of deep learning-based dehazing approaches, spanning from early CNN-based physical model estimators to recent generative diffusion models. We systematically analyze atmospheric scattering models, network architectures (CNNs, GANs, Transformers), and loss function designs. Our evaluation on both synthetic (RESIDE, NH-HAZE) and real-world benchmarks demonstrates that physics-informed hybrid architectures achieve superior perceptual quality while maintaining physical plausibility. We discuss critical challenges including non-uniform haze, sky region handling, and real-time deployment constraints. Future directions emphasize neural atmospheric scattering priors and zero-shot adaptation to unseen haze densities.

**Keywords:** Image dehazing, atmospheric scattering, deep learning, GAN, diffusion models, perceptual quality

---

## 1. Introduction

Atmospheric haze significantly degrades outdoor image quality by scattering and absorbing light, posing severe challenges for high-level vision tasks in autonomous driving, remote sensing, and surveillance systems. Unlike multi-image or polarization-based methods, single image dehazing must resolve the fundamental ambiguity between scene radiance and atmospheric light from a single observation.

The atmospheric scattering model proposed by McCartney (1976) and refined by Narasimhan and Nayar (2000, 2002) formalizes the haze formation process:

$$
I(x) = J(x)t(x) + A(1 - t(x))
$$

Where:
- $I(x)$: Observed hazy image
- $J(x)$: Scene radiance (clear image to recover)
- $A$: Global atmospheric light
- $t(x)$: Medium transmission map, $t(x) = e^{-\beta d(x)}$
- $\beta$: Atmospheric scattering coefficient
- $d(x)$: Scene depth

### 1.1 Problem Formulation

Given $I(x)$, dehazing estimates $J(x)$, typically via intermediate estimation of transmission map $t(x)$ and atmospheric light $A$:

$$
J(x) = \frac{I(x) - A}{t(x)} + A
$$

This presents three core challenges:
1. **Ill-posedness**: Infinite $(J, t, A)$ combinations produce identical $I$
2. **Non-uniform haze**: Spatially varying $\beta$ violates homogeneous medium assumption
3. **Color distortion**: Over-saturation in sky regions, under-restoration in dense haze

### 1.2 Evolution of Dehazing Methodologies

**Prior-based Era (2009-2015):** Dark Channel Prior (DCP) by He et al. (2009) dominated through statistical observation that clear patches contain low-intensity pixels in at least one color channel. Limitations include sky region artifacts and fixed prior assumptions.

**CNN-based Era (2016-2018):** End-to-end learning of transmission maps (DehazeNet, MSCNN) replaced hand-crafted priors with data-driven feature extraction.

**GAN-based Era (2019-2021):** Conditional GANs (CycleGAN, Pix2Pix) enabled unpaired training and perceptual quality optimization, though often sacrificing physical consistency.

**Transformer/Diffusion Era (2022-present):** Self-attention mechanisms and diffusion probabilistic models capture long-range dependencies and generative priors for realistic detail recovery.

---

## 2. Related Work

### 2.1 Physical Model-Based Approaches

Early deep learning methods focused on estimating physical parameters:

| Method | Architecture | Target | Key Innovation |
|--------|-------------|---------|----------------|
| **DehazeNet** (2016) | CNN | $t(x)$ | BReLU activation for bounded transmission |
| **MSCNN** (2016) | Multi-scale CNN | $t(x)$ | Coarse-to-fine transmission estimation |
| **AOD-Net** (2017) | Light CNN | $K(x)$ | Reformulated $J(x) = K(x)I(x) - K(x) + b$ |
| **DCPDN** (2018) | Two-stream CNN | $t(x), A$ | Joint estimation with physical constraints |

**Limitation:** Cumulative errors in $t(x)$ and $A$ estimation propagate to final $J(x)$ via division operation, amplifying noise in distant regions.

### 2.2 End-to-End Image Translation

Bypassing explicit physical parameter estimation:

| Method | Framework | Training | Loss Functions |
|--------|-----------|----------|----------------|
| **Pix2Pix** (2017) | cGAN | Paired | $\mathcal{L}_{GAN} + \mathcal{L}_{L1}$ |
| **CycleGAN** (2017) | Cycle-consistent GAN | Unpaired | $\mathcal{L}_{cycle} + \mathcal{L}_{identity}$ |
| **Pix2Pix-Turbo** (2024) | One-step GAN | Paired | Adversarial + Perceptual + Distillation |
| **GridDehazeNet** (2019) | Attention CNN | Paired | Multi-scale grid network |

**Pix2Pix-Turbo Architecture:**
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
        self.bottleneck = nn.Sequential(
            *[ResBlock(512) for _ in range(4)]
        )
        
        # Decoder with skip connections
        self.dec4 = UpConvBlock(512, 256)
        self.dec3 = UpConvBlock(512, 128)  # 256+256 skip
        self.dec2 = UpConvBlock(256, 64)   # 128+128 skip
        self.dec1 = ConvBlock(128, out_channels)  # 64+64 skip
        
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.dec1(d2)
        return torch.tanh(d1) + x  # Residual learning
```