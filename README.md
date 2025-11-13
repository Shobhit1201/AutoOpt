# AutoOpt: Unified Framework for Automated Optimization Understanding (NeurIPS 2025)

<p align="center">
  <a href="https://arxiv.org/abs/2510.21436"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv%3A2510.21436-white"></a>
<a href="https://colab.research.google.com/github/Shobhit1201/AutoOpt/blob/main/Module%20M1/Module_M1_Inference.ipynb">
  <img alt="Colab" src="https://img.shields.io/badge/Colab-Demo-yellow">
</a>
  <a href="https://www.kaggle.com/datasets/ankurzing/autoopt-11k"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-Kaggle-blue"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

**AutoOpt** is an end-to-end **automated optimization framework** that takes an *image of a mathematical formulation* and solves it without human intervention.  

Currently available modules:

* **M1 · Image → LaTeX (MER)** — Hybrid CNN‑Transformer model for equation understanding.
* **M2 · LaTeX → Pyomo** — Fine‑tuned code generation model translating LaTeX to optimization code.

> **M3 · Solver** — *To be released soon*

---

## 🧩 Overview

The AutoOpt framework unifies perception, translation, and solving of optimization models through three sequential modules:

* **M1 (Image→LaTeX):** Recognizes handwritten/printed optimization models using a hybrid *ResNet + Swin Transformer* encoder and *mBART* decoder.
* **M2 (LaTeX→Pyomo):** Converts the LaTeX into clean Pyomo scripts using a fine‑tuned LLM.
* **M3 (Solver):** Integrates a Bilevel Optimization‑Based Decomposition (BOBD) solver *(coming soon)*.

**Dataset:** [AutoOpt‑11k on Kaggle](https://www.kaggle.com/datasets/ankurzing/autoopt-11k)
**Paper:** [AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](https://arxiv.org/abs/2510.21436)

---

## ⚙️ Installation

```bash
git clone https://github.com/Shobhit1201/AutoOpt.git
cd AutoOpt/Module_M1
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

> Make sure your PyTorch version matches your CUDA environment. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 📊 Evaluation

### BLEU Score (higher is better)

<p align="left">
  <img src="https://github.com/Shobhit1201/AutoOpt/blob/main/assets/BLEU.jpg" alt="BLEU Scores" width="500" height="300"/>
</p>


### Character Error Rate (lower is better)

| Model (Size)             |         HW |         PR |      HW+PR |
| ------------------------ | ---------: | ---------: | ---------: |
| GPT 4o (Large)           |     0.1465 |     0.0664 |     0.1017 |
| Gemini 2.0 Flash (Large) |     0.1607 |     0.1047 |     0.1338 |
| Nougat (348.7M)          |     0.0752 | **0.0168** |     0.0440 |
| **AutoOpt-M1 (393.3M)**  | **0.0412** |     0.0176 | **0.0286** |

<sub>HW: Handwritten; PR: Printed; HW+PR: Handwritten+Printed.</sub>

---

## 🚀 Usage

### 🧠 Predict (Image → LaTeX)

```bash
!gdown --fuzzy "https://drive.google.com/file/d/1WqfAv0b4FwR8-xqyDLXHTFxBMCas2Fej/view?usp=drive_link" -O checkpoints/m1_weights.pt
python tools/m1_infer.py --image path/to/image.png --checkpoint checkpoints/weights.pth
```


### 🧩 Fine‑tune (Custom Dataset)

Prepare your dataset similar to [AutoOpt‑11k](https://www.kaggle.com/datasets/ankurzing/autoopt-11k) and execute:

```bash
python tools/train_experiment.py --config_file config/base.yaml --phase train
```

### ☁️ Colab (Interactive Demo)

Use the hosted Colab notebook to perform inference or fine‑tuning directly:
<a href="https://colab.research.google.com/github/Shobhit1201/AutoOpt/blob/main/Module%20M1/Module_M1_Inference.ipynb">
  <img alt="Colab" src="https://img.shields.io/badge/Colab-Demo-yellow">
</a>

---

## 📁 Directory Structure

```
AutoOpt/
├── Module M1/
│   ├── nougat‑latex‑ocr/           # MER model code and training utilities
│   ├── Module_M1_Inference.ipynb   # Colab‑ready notebook for inference & finetuning
│
├── Other MER models/
│   ├── Original_Nougat/
│   ├── ChatgptComparison.ipynb
│   └── GeminiComparison.ipynb
│
├── Module_M2.ipynb                 # LaTeX → Pyomo translation notebook
└── README.md
```

> The Colab notebook `Module_M1_Inference.ipynb` allows users to perform **end‑to‑end inference or fine‑tuning without setup**.

---

## 📚 Dataset — AutoOpt‑11k

**Access:** [Kaggle Dataset](https://www.kaggle.com/datasets/ankurzing/autoopt-11k)

* 11,554 images (handwritten + printed) labeled with verified LaTeX.
* 1,018 paired Pyomo scripts for direct LaTeX→code training.
The dataset spans linear, nonlinear, convex, nonconvex, and stochastic optimization problems across domains. See [AutoOpt_Paper.pdf](./docs/AutoOpt_Paper.pdf) for detailed composition and statistics.

---

## 🧾 Citation

```bibtex
@article{AutoOpt2025,
  title   = {AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving},
  author  = {Ankur Sinha and Shobhit Arora and Dhaval Pujara},
  journal = {arXiv preprint arXiv:2510.21436},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.21436}
}
```

---

## 🪪 License

Released under the **MIT License**.

---

⭐ *If you find this repository useful, please consider leaving a star on GitHub!*
