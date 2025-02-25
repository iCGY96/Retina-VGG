# Pretrained VGG16 Saliency Detection

This repository contains a complete training and evaluation pipeline for saliency detection using a pretrained VGG16 network. The model is trained on the SALICON dataset and evaluated on the MIT1003 and Toronto datasets.

---

## Overview

The project consists of the following key components:

- **Data Preparation:**  
  Functions to download, unify, and clean the SALICON, MIT1003, and Toronto datasets. It also includes preprocessing steps such as simulating retina processing using OpenCV's `cv2.bioinspired.Retina` module.

- **Model:**  
  A saliency detection model based on a pretrained VGG16 network, defined in the `PretrainedVGG16Saliency` class.

- **Loss Functions:**  
  A combination of a custom Focal Loss (for binary saliency maps) and Mean Squared Error (MSE) Loss. This combination helps the model focus on difficult samples during training.

- **Training & Evaluation:**  
  - **Training:** The model is trained on the SALICON dataset with per-epoch evaluation on the MIT1003 test set. The best model is saved based on the Pearson Correlation Coefficient (CC).
  - **Evaluation:** Final performance is measured on both the MIT1003 and Toronto test sets using various metrics, including CC, Similarity, EMD, KL divergence, NSS, AUC-Judd, AUC-Borji, and AUC-shuffled.

- **Utility Functions:**  
  Additional functions for setting random seeds (to ensure reproducibility) and for computing evaluation metrics.

