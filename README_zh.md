# 🎨 莫奈风格迁移 — Kaggle《I'm Something of a Painter Myself》
### CycleGAN · CUT · AttentionGAN 统一实现仓库

本项目为 Kaggle 图像生成竞赛  
**“I’m Something of a Painter Myself”**  
提供完整的风格迁移解决方案。

目标是利用 **无监督图像到图像转换（Unpaired Image-to-Image Translation）**，  
将普通照片转换为 Monet（莫奈）风格的艺术图像。

本仓库统一实现并对比三大经典模型：

- **CycleGAN（ICCV 2017）**
- **CUT – 对比式无监督翻译（ECCV 2020）**
- **AttentionGAN（WACV 2021）**

---

## 🏆 竞赛表现

在提交报告时，我们小组在排行榜上的成绩为：

# 🎉 **36.94468 分，排名第 4 / 170（前 2%）**

该成绩体现了模型在 FID/MI-FID 指标上的表现与稳定性。

![Score](imgs/score.png)

> *(注：竞赛榜单可能随时间更新，此处为当时成绩记录。)*

---

## 🏁 竞赛任务简介

数据集包含：

- `monet_jpg`, `monet_tfrec`：真实莫奈画作  
- `photo_jpg`, `photo_tfrec`：自然照片  

参赛者需要生成：

- **7000–10000 张**
- **256×256 分辨率**
- Monet 风格的图像  
并以 `images.zip` 形式提交。

评价指标为：

- **FID**  
- **MI-FID**

用于衡量生成图像与真实莫奈画作的分布差距。

---

## 📂 项目结构
```
painter/
├── data/                 # 仅包含结构，无真实数据
├── imgs/                 # 可放示例图片（可选）
├── notebooks/            # Kaggle 提交流程和探索
├── src/
│   ├── models/           # CycleGAN / CUT / AttentionGAN 模型
│   ├── datasets/         # 数据集处理
│   ├── util/             # 通用工具
│   ├── train_cyclegan.py
│   ├── train_cut.py
│   ├── train_attngan.py
│   ├── test_cyclegan.py
│   ├── test_cut.py
│   └── test_attngan.py
├── README.md
└── README_zh.md
```
---

## 🧠 模型介绍

### **1️⃣ CycleGAN**

特点：

- 使用 Cycle-Consistency Loss  
- 不需要配对训练数据  
- 适合风格迁移任务  

训练经验：

- Batch size = 1 效果最佳  
- 学习率衰减对于稳定性非常关键  

---

### **2️⃣ CUT（Contrastive Unpaired Translation）**

特点：

- 使用 **PatchNCE 对比损失**
- 保留更多图像结构  
- 收敛更快  

---

### **3️⃣ AttentionGAN**

特点：

- **空间注意力 + 通道注意力**  
- 更精细的局部风格学习  
- 有利于艺术纹理建模  

---

## 📊 最终结果展示

不同模型擅长不同风格：

| 模型 | 优势 | 劣势 |
|------|------|-------|
| **CycleGAN** | 强烈的整体风格迁移能力 | 不稳定，易波动 |
| **CUT** | 最佳结构保留 | 风格有时略弱 |
| **AttentionGAN** | 最细腻的局部纹理 | 训练更耗资源 |


---

## 🎯 CUT 训练过程分析

为了进一步提升效果，我们针对 CUT 尝试了 **四类训练策略**，并比较了
对应的 FID 分数与训练曲线。结果如下：

| 训练策略           | FID 分数 |
|--------------------|----------|
| 常规技巧调优       | ~1.8     |
| **Best-Restart**   | ~1.6（最佳） |
| 杜高等预训练       | ~4.0     |
| 莫奈画作微调       | ~1.8     |

以下展示四种策略的训练曲线：

### 📌 常规技巧调优
![CUT Standard Training](imgs/cut_standard.png)

### ⭐ Best-Restart（最佳策略）
从最佳 checkpoint 重新启动训练，可以显著改善收敛稳定性，
取得本组实验中最优的 ~1.6 FID。
![CUT Best-Restart](imgs/cut_best_restart.png)

### 🔁 杜高等预训练
长周期预训练导致效果变差（~4.0），可能存在过拟合或风格偏移。
![CUT High Epoch Pretraining](imgs/cut_pretrain.png)

### 🎨 莫奈画作微调
在真实 Monet 数据上进行微调，可恢复到 ~1.8 水平，风格更贴近真实画作。
![CUT Monet Finetune](imgs/cut_monet_finetune.png)

上述结果说明：  
**合理选择 checkpoint、采用分阶段训练策略，对于 CUT 在风格迁移任务中的性能提升非常关键。**

---

## 🛠️ 如何训练

### **CycleGAN**
```bash
python src/train_cyclegan.py
```
### **CUT**
```bash
python src/train_cut.py
```
### **AttentionGAN**
```bash
python src/train_attngan.py
```

---

🙏 致敬与引用

本项目基于以下优秀论文与开源实现：

- CycleGAN — Zhu et al., ICCV 2017
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- CUT — Park et al., ECCV 2020
https://github.com/taesungp/contrastive-unpaired-translation
- AttentionGAN — Tang et al., WACV 2021
https://github.com/Ha0Tang/AttentionGAN