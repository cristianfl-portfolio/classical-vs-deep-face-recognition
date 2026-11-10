# Computer Vision - Face Recognition Project

An experimental computer vision project evaluating handcrafted feature extraction techniques (HOG, SIFT, PCA) against modern classifiers on a small subset of the VGG Face dataset, this is a group project as part of a Kaggle competition.

## Overview

The objective of this project is to implement feature extraction representations for face verification and multi-class recognition under limited sample conditions (using only 80 training images vs 1,816 evaluation images). Final evaluation is performed in-the-wild test images on Kaggle.
- Part 1: Understanding Handcrafted Features
- Part 2: Face recognition

**Input data:** 80 images used for training or evaluation (30 of Jesse Eisenberg, 30 of Mila Kunis, 10 of Michael Cera, 10 of Sara Hyland)

**Output data:** A fine-grained classification (1 - Jesse Eisenberg, 2 - Mila Kunis, 0 - Neither of them, for example: Michael Cera, Sara Hyland)

## Results and Comparison

All feature representations were paired with multi-layer perceptron (`MLPClassifier`) or custom PyTorch classification heads, 
tuned via 5-fold cross-validated randomized search (`RandomizedSearchCV`), and evaluated on the held-out Kaggle test leaderboard:

| Category | Preprocessing | Feature Extractor & Dimensionality | Classifier Architecture | Validation Accracy | Kaggle Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sparse Keypoints** | HAAR Cascade + Hist. Equalization | **SIFT + BoVW** (k=125 K-Means clusters, 96 Mutual Info features) | MLP (100, 200, 200, 100), α=0.0003 | 77.27% | **60.0% – 65.0%** |
| **Linear Subspace** | RetinaFace Alignment + Circular Mask + H-Flip | **PCA Eigenfaces** (p=10 components via randomized SVD) | MLP (50, 100, 50), α=0.001 | 93.75% | **~67.0%** |
| **Dense Gradients** | HAAR Cascade (100 x 100 grayscale) | **HOG** (6,912 raw -> top 30% via Mutual Info) | MLP (100, 100), α=0.0001 | 79.31% | **~75.0%** |
| **Deep Transfer Learning** | MTCNN (200 x 200 -> 160 x 160) + Affine/Color Aug | **Inception-ResNet-V1** (`vggface2` pretrained, 512-D) | Dual Binary PyTorch MLPs + Min-Logit Fusion | **100.00%** | **91.46% (Top 35%)** |

## Methodology

### 1. Data Preprocessing & Caching for PCA
Because classical subspace methods (like PCA) compute pixel-wise covariance across the image grid, unaligned faces and background clutter severely corrupt the principal components. To solve this, we have:
* Removed corrupted all-black frames and manually isolated target subjects in multi-person training frames prior to detection.
* Used RetinaFace (`align=True`) to normalize eye and mouth coordinates across rotational variations.
* Applied a vectorized elliptical mask to zero out corner background noise, and doubled the PCA training set via horizontal mirroring to enforce bilateral facial invariance.
* Implemented a disk caching layer during preprocessing to speed-up sub-second data loading because RetinaFace inference over 1,896 images introduces a major I/O and compute bottleneck.

### 2. Classical Feature Representations
* **Histogram of Oriented Gradients (HOG):** Configured `cv2.HOGDescriptor` with 10 × 10 px cells, 3 × 3 cell blocks (30 × 30 px), a 10 × 10 px block stride, and 12 orientation bins over 100 × 100 images, yielding **6,912 features per face**. Filtered down to the top 30% most informative features using Mutual Information (`mutual_info_classif`).
* **Scale-Invariant Feature Transform (SIFT) + Bag of Visual Words (BoVW):** Applied histogram equalization (`cv2.equalizeHist`) to boost local contrast, extracted 96-D keypoint descriptors, and quantized variable-length image descriptors into fixed-size L2-normalized frequency histograms via **K-Means clustering (k=125)**.
* **Principal Component Analysis (PCA):** Built a custom 2D-preserving `PCAFaceFeatureExtractor` using randomized SVD and Einstein summation (`np.einsum`) for forward projection and inverse image reconstruction. Retaining **p=10 components** compressed the 10,000-dimensional pixel space while preserving **~70% of cumulative variance** (and >80% at p=29).

### 3. Transfer Learning & Dual-Classifier Ensemble
To overcome the generalization ceiling of handcrafted features. We have:
* Evaluated `InceptionResnetV1` pretrained on both `casia-webface` and `vggface2`, selecting `vggface2` for superior validation separation.
* Applied `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2)`, and `GaussianBlur` prior to MTCNN extraction (3× augmentation multiplier) so synthetic transformations did not conflict with MTCNN's internal normalization (`post_process=True`).
* Processed test images in memory-safe batches of 256 with explicit garbage collection (`gc.collect()`) to prevent VRAM exhaustion.
* Used a Dual Classifier rather than training a single 3-class softmax head (which struggles when Class 0 is a heterogeneous mix of different people), we decoupled the problem into two specialized binary PyTorch networks:
  1. `classifier_jesse`: Jesse Eisenberg (1) vs. Non-Jesse (0)
  2. `classifier_mila`: Mila Kunis (1) vs. Non-Mila (0)
  
  Each network uses `BatchNorm1d(512)` → `Linear(512, 1024)` → `Dropout(0.5)` → `Linear(1024, 256)` → `Dropout(0.4)` → `Linear(256, 128)` → `Dropout(0.3)` → `Linear(128, 2)`, trained with `AdamW` (lr=10^-3, weight_decay=10^-2) and `ReduceLROnPlateau`. Predictions are merged via custom logit fusion (`combine_outputs`), assigning the negative class score as min(out_jesse, out_mila)$.

## Conclusions

1. **PCA Overfit:**
   While PCA achieved **93.75% validation accuracy**, it dropped to **67% on the Kaggle test set**. Visualizing the 2D feature space revealed that class separation occurred almost entirely along the **2nd principal component (y-axis)** rather than the 1st principal component (x-axis). Because PCA is unsupervised, its top eigenvector captured global illumination and skin-tone variance rather than identity-specific geometry.
2. **Spatial Grid Consistency vs. Sparse Keypoints (HOG vs. SIFT):**
   SIFT struggled (~61% test accuracy) because smooth, low-resolution (100 × 100) facial crops lack sharp corner extrema, causing keypoint starvation on several images. Conversely, HOG computes dense gradients across a fixed spatial grid regardless of texture sharpness, making it the strongest classical baseline (**75% test accuracy**).
3. **Pretrained Inductive Bias under Data Scarcity:**
   With fewer than 80 training samples, classical models must learn both feature relevance and decision boundaries simultaneously from scratch. Pretrained deep embeddings (`vggface2`) inject prior knowledge from millions of faces, mapping images into a linearly separable 512-D metric space within just a few epochs.

## 🛠 Tech Stack

* **Python 3.10+**
* **Deep Learning:** PyTorch, facenet-pytorch (MTCNN, InceptionResnetV1), retina-face
* **Computer Vision & Classical ML:** OpenCV, Scikit-Learn
* **Data and Visualization:** NumPy, Pandas, Matplotlib, Seaborn, ipywidgets

## 🔒 Code Access
To comply with academic integrity policies, this repository is private. 

Full Jupyter Notebook is readily available for recruiters and technical evaluations upon request.
**Contact me** via [LinkedIn](https://linkedin.com/in/cristian-ferlop).

## Academic Integrity & License
This repository is for portfolio and demonstration purposes only. 
Unauthorized copying, sharing, or academic submission of this code is strictly prohibited.

All rights reserved © 2025.

**Authors:** Cristian Fernández, Jan Cichomski, Dimitris Mystriotis, Hanlin Wu, Jakub

