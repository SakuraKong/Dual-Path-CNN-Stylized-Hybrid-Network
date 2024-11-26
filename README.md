Dual-Path CNN-Stylized Hybrid Network for Boundary-Aware Segmentation and Angle of Progression Measurement in Intrapartum Ultrasound

# CNN-Stylized Dual-Path CNN-Transformer Hybrid Model for Fetal Head and Pubic Symphysis Segmentation in Intrapartum Ultrasound Images


<img src="https://github.com/SakuraKong/Dual-Path-CNN-Stylized-Hybrid-Network/blob/master/fig2.jpg" alt="Medical Ultrasound Image Segmentation" width="600">
---

### 🚀 **Overview**

Accurate segmentation of the fetal head (FH) and pubic symphysis (PS) in intrapartum ultrasound (IU) images is essential for automatically measuring the angle of progression (AoP). This is crucial for predicting delivery outcomes and preventing related complications.

This repository presents a novel **CNN-stylized dual-path CNN-Transformer hybrid model** that addresses key challenges in IU image segmentation:

- **Attention Collapse:** Mitigated by using a lightweight, multi-branch Transformer in parallel with a CNN.
- **VIT and CNN feature fusion problem:** The feature of VIT is effectively close to and fused with the feature of CNN.

---

### 📋 **Features**

1. **Parallel CNN-Transformer Encoder:**  
   - Combines the strengths of CNNs and Transformers without attention collapse.
   - A custom feature interaction module ensures effective fusion of CNN and Transformer outputs.

2. **Boundary Attention Residual Module (BARM):**  
   - Captures hidden boundary details from foreground and background.
   - Utilizes residual structures to refine boundary features progressively.

3. **T2C:**  
   - Well fused dual path features.


---

### 🏥 **Clinical Significance**

- **Automatic AoP Measurement:** The proposed model enhances accuracy and reliability.
- **Clinical Application:** Demonstrates strong potential for real-world use in delivery outcome prediction and complication prevention.

---


### 📂 **Dataset**

- **Dataset A:** FHPS2023: https://ps-fh-aop-2023.grand-challenge.org/
- **Dataset B:** HC18: https://zenodo.org/record/1327317

---
**⭐️ Star this repository if you find it useful!**
