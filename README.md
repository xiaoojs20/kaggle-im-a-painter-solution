# 🎨 I'm Something of a Painter Myself — Monet Style Transfer
### Unified Implementation of CycleGAN · CUT · AttentionGAN

This repository contains a complete solution for the Kaggle competition  
**“I’m Something of a Painter Myself”**, which focuses on generating Monet-style
paintings from natural photos using *unpaired image-to-image translation*.

We reconstruct and unify three representative GAN models:

- **CycleGAN** (ICCV 2017)
- **CUT – Contrastive Unpaired Translation** (ECCV 2020)
- **AttentionGAN** (WACV 2021)

---

## 🏆 Competition Performance

Our team achieved a **final score of 36.94468** on the leaderboard, ranking:

# 🎉 **4th out of 170 teams (Top 2%)**

This reflects strong model performance and stability across multiple architectures.

> *(Score and ranking were recorded during project reporting; leaderboard may later change depending on competition timeline.)*

---

## 🚀 Overview

The objective is to train models that generate 7,000–10,000 Monet-style images  
(256×256 RGB) for submission.

The dataset consists of:

- Monet paintings (`monet_jpg`, `monet_tfrec`)
- Natural photos (`photo_jpg`, `photo_tfrec`)

Evaluation uses **FID** and **MI-FID**, measuring the distance between the
distribution of generated images and real Monet images.

---

## 📂 Project Structure
```
painter/
├── data/                 # dataset placeholder; real data not included
├── imgs/                 # images for documentation (optional)
├── notebooks/            # Kaggle submission & analysis notebooks
├── src/
│   ├── models/           # unified CycleGAN / CUT / AttentionGAN models
│   ├── datasets/         # dataset loading utilities
│   ├── util/             # training utilities (logging, losses, etc.)
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

## 🧠 Models

### **1️⃣ CycleGAN**
CycleGAN performs unpaired translation using:

- Generator–Discriminator pairs  
- Cycle consistency loss  
- Least-squares GAN loss (LSGAN)

💡 *Observations during training*  
CycleGAN is sensitive to learning rate decay and requires batch size = 1,
consistent with official implementations.

📎 *(Insert training curves using external links if desired)*  
Example:  
![CycleGAN FID Curve](https://via.placeholder.com/600x300?text=CycleGAN+FID+Curve)

---

### **2️⃣ CUT (Contrastive Unpaired Translation)**

CUT replaces cycle-consistency with **PatchNCE contrastive loss**, allowing
stronger content preservation.

Key advantages:

- Faster convergence  
- Better structure retention  
- Works well on style transfer tasks with strong texture changes  

📎 *(Insert visual comparison via URL)*  
![CUT Architecture](https://via.placeholder.com/600x300?text=CUT+Architecture)

---

### **3️⃣ AttentionGAN**

AttentionGAN enhances translation using:

- Spatial attention  
- Channel attention  
- Region-level style learning  

This often yields more aesthetically pleasing Monet strokes and textures.

📎 Example placeholder:  
![AttentionGAN](https://via.placeholder.com/600x300?text=AttentionGAN+Example)

---

## 📊 Results Summary

| Model         | Strengths                               | Weaknesses                          |
|---------------|------------------------------------------|--------------------------------------|
| **CycleGAN**  | Strong global style transfer             | Unstable training, sensitive to LR   |
| **CUT**       | Best content preservation                | Sometimes less stylized              |
| **AttentionGAN** | Finest local texture & color details | Training more computationally heavy  |

Final generated images demonstrate that all three methods can produce Monet-style transformations with different emphases on texture, structure, and artistic abstraction.

📎 Insert comparison (optional):  
![Comparison Grid](https://via.placeholder.com/800x350?text=Method+Comparison)

---

## 🛠️ How to Train

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

⸻

🙏 Acknowledgements

This project builds on the outstanding work of these open-source implementations:
	•	CycleGAN – Zhu et al., ICCV 2017
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
	•	CUT – Park et al., ECCV 2020
https://github.com/taesungp/contrastive-unpaired-translation
	•	AttentionGAN – Tang et al., WACV 2021
https://github.com/Ha0Tang/AttentionGAN