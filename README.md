# Explainable and Uncertainty-Aware VGRNN for Brain Tumor Classification

## 📌 Overview

This project proposes an **Explainable and Uncertainty-Aware Variational Graph Recurrent Neural Network (VGRNN)** for brain tumor classification from MRI data.

Instead of treating images as regular grids, we convert MRI scans into **graph-structured representations**, enabling the model to capture spatial relationships between anatomical regions.

The model integrates:

* **Graph Neural Networks (GCN)** for spatial representation
* **Variational Inference (VAE)** for uncertainty modeling
* **Recurrent Units (GRU)** for latent temporal dynamics

This approach aims to improve:

* robustness under noisy conditions
* interpretability of predictions
* reliability in high-stakes medical applications

---

## 🧠 Key Contributions

* 🧩 Graph-based representation of MRI using **SLIC superpixels**
* 🔄 Novel use of **VGRNN for static medical images via simulated temporal dynamics**
* 🎲 Explicit modeling of **uncertainty through latent variables**
* 📊 Analysis of **confidence calibration and misclassification behavior**
* 🔍 Integration of **explainability techniques (e.g., Grad-CAM on graph features)**

---

## 🗂️ Dataset

* Brain MRI dataset with 4 classes:

  * Glioma
  * Meningioma
  * Pituitary tumor
  * No tumor

📁 Expected structure:

```bash
data/
  train/
    glioma/
    meningioma/
    pituitary/
    notumor/
  test/
    ...
```

---

## ⚙️ Methodology

### 1. Image → Graph Conversion

* Apply **SLIC segmentation**
* Extract region-level features:

  * mean intensity
  * texture (LBP)
  * geometric properties (area, eccentricity)
* Build adjacency graph using **Region Adjacency Graph (RAG)**

---

### 2. Model Architecture

The model combines:

* **GCN layers** → node embedding
* **Variational encoder** → latent distribution (μ, σ)
* **GRU** → temporal evolution in latent space

Pipeline:

```
Graph → GCN → Latent (VAE) → GRU → Classifier
```

---

### 3. Training

* Loss:

  * Cross-entropy (classification)
  * KL divergence (variational regularization)
* Optimization:

  * Adam optimizer
* Frameworks:

  * PyTorch
  * PyTorch Geometric
  * PyTorch Lightning

---

## 📊 Evaluation

### Metrics

* Accuracy
* Precision / Recall / F1-score
* Calibration analysis (confidence vs accuracy)

### Additional Analysis

* Misclassification case studies
* Confidence distribution
* Robustness to noise

---

## 🔍 Explainability

* Adaptation of **Grad-CAM** for graph-based representations
* Visualization of important regions in MRI
* Analysis of model attention vs tumor regions

---

## 🚀 Results (Example)

| Model            | Accuracy |
| ---------------- | -------- |
| Baseline CNN     | XX%      |
| GCN              | XX%      |
| **VGRNN (ours)** | XX%      |

👉 Replace with your real results

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/vgrnn-mri.git
cd vgrnn-mri

pip install -r requirements.txt
```

---

## ▶️ Usage

### Train model

```bash
python train.py
```

### Evaluate

```bash
python test.py
```

---

## 📁 Project Structure

```bash
.
├── data/
├── models/
├── utils/
├── train.py
├── test.py
├── config.yaml
└── README.md
```

---

## 🔬 Research Context

This project lies at the intersection of:

* Graph Representation Learning
* Probabilistic Deep Learning
* Explainable AI (XAI)

It is particularly motivated by **medical AI applications in resource-constrained environments**, where model reliability and interpretability are critical.

---

## 📜 Citation

```bibtex
@article{bonou2025vgrnn,
  title={Explainable and Uncertainty-Aware VGRNN for Brain Tumor Classification},
  author={Bonou, Bienvenu},
  year={2025}
}
```

---

## 📬 Contact

For questions or collaborations:
📧 [eudesbonou@gmail.com](mailto:eudesbonou@gmail.com)
