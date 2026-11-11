# 🧪 PyLabFlow Workshop

> **A Local-First, Reproducible Framework for Classical ML & Deep Learning Experiments**  
> **Publication Citation:** *Hati & Dadwhal (2025). PyTorchLabFlow: A Local-First Framework for Reproducible Deep Learning Experiments. Journal of Applied Bioanalysis (JAB), 11(S6), 538–546. [DOI: 10.53555/jab.v11si6.1940](https://doi.org/10.53555/jab.v11si6.1940)*

---

## 📌 Overview

Welcome to the **PyLabFlow Workshop**! This repository contains all code, modular components, and interactive Jupyter notebooks for the hands-on session.

Everything runs **100% offline on your local machine** (Linux, macOS, and Windows) using an embedded SQLite database (`ppls.db`) without requiring external cloud tracking servers, accounts, or subscriptions.

---

## 🚀 Quick Start Guide

### 1. Pre-Workshop Setup
Before attending the workshop, complete the one-time offline environment setup:
👉 **[Read the Pre-Workshop Setup Guide (WORKSHOP_PREREQUISITES.md)](WORKSHOP_PREREQUISITES.md)**

Summary commands:
```bash
# 1. Clone this repository & navigate into it
git clone https://github.com/ExperQuick/PyLabFlow-Workshop.git
cd PyLabFlow-Workshop

# 2. Create and activate dedicated environment (Python 3.10)
conda create -n ubucnplf python=3.10 -y
conda activate ubucnplf

# 3. Install PyLabFlow, ML stack, and Jupyter Notebook
pip install PyLabFlow scikit-learn pandas joblib numpy matplotlib jupyter
pip install torch torchvision   # or with --index-url https://download.pytorch.org/whl/cu121 for NVIDIA GPUs

# 4. Launch Jupyter Notebook from the PyLabFlow-Workshop root directory
jupyter notebook
```

---

## 📂 Repository Structure

All materials are organized into two self-contained tracks right inside this repository:

```text
PyLabFlow-Workshop/
│
├── ML/                          <-- 📊 Classical ML Track (Diabetes Progression)
│   ├── MLComps_ubucon/          <-- Provided Starter Components (myComps.py)
│   ├── WorkSpace/               <-- Interactive Notebooks (setup, deBug, exptrail, Ana)
│   └── README.md                <-- Track Guide
│
├── DL/                          <-- 🧠 Deep Learning Track (MNIST Digit Classification)
│   ├── StarterCompDir_ubucon/   <-- Provided Starter Components (datasets, models, workflows)
│   ├── WorkSpace/               <-- Interactive Notebooks (setup, experiment, Monitor)
│   └── README.md                <-- Track Guide
│
├── WORKSHOP_PREREQUISITES.md    <-- 🛠️ Attendee pre-workshop checklist & setup instructions
├── WORKSHOP_COMPLETE_GUIDE.md   <-- 📖 Master workshop guide (Theory, Labs, CLI & FAQs)
└── README.md                    <-- 📍 You are here
```

---

## 📚 Hands-On Tracks

| Track | Directory | Objective | Key Files |
| :--- | :--- | :--- | :--- |
| **Classical ML** | [`ML/`](ML/) | Build modular regression pipelines, isolate unit tests, benchmark Linear, Ridge & Random Forest models, and query the SQLite leaderboard. | [`ML/MLComps_ubucon/myComps.py`](ML/MLComps_ubucon/myComps.py)<br>[`ML/WorkSpace/setup.ipynb`](ML/WorkSpace/setup.ipynb) |
| **Deep Learning** | [`DL/`](DL/) | Build and train configurable CNNs on MNIST, monitor live training curves, iterate hyperparameters, and track experiment lineage. | [`DL/StarterCompDir_ubucon/`](DL/StarterCompDir_ubucon/)<br>[`DL/WorkSpace/setup.ipynb`](DL/WorkSpace/setup.ipynb) |

---

## 📖 Complete Documentation

For the full conceptual background, step-by-step code annotations, CLI utility reference, and multi-OS troubleshooting:
👉 **[Read the Complete Participant Guide (WORKSHOP_COMPLETE_GUIDE.md)](WORKSHOP_COMPLETE_GUIDE.md)**

