# 🧪 PyLabFlow Workshop: Complete All-in-One Participant Guide

> **Official Step-by-Step Hands-On Workshop Guide for Attendees**  
> **Platforms Supported:** Ubuntu / Linux, macOS (Apple Silicon & Intel), Windows 10/11  
> **Execution Mode:** 100% Offline Local Machine Execution (No Cloud or Internet Required After Setup)  
> **Publication Citation:** *Hati & Dadwhal (2025). PyTorchLabFlow: A Local-First Framework for Reproducible Deep Learning Experiments. Journal of Applied Bioanalysis (JAB), 11(S6), 538–546. [DOI: 10.53555/jab.v11si6.1940](https://doi.org/10.53555/jab.v11si6.1940)*

---

## 📌 Master Table of Contents
1. [Workshop Overview & Philosophy](#1-workshop-overview--philosophy)
2. [Part 1: Pre-Requisites & Offline Environment Setup (All OS)](#part-1-pre-requisites--offline-environment-setup-all-os)
   - [Hardware Requirements](#hardware-requirements)
   - [Complete Dependency Manifest](#complete-dependency-manifest)
   - [Step 1.1: Install Miniconda or Anaconda](#step-11-install-miniconda-or-anaconda)
   - [Step 1.2: Create Dedicated Conda Environment](#step-12-create-dedicated-conda-environment)
   - [Step 1.3: Install All Required Python Libraries](#step-13-install-all-required-python-libraries)
   - [Step 1.4: Launch Jupyter Notebook Directly](#step-14-launch-jupyter-notebook-directly)
   - [Step 1.5: Offline Verification Script](#step-15-offline-verification-script)
   - [Step 1.6: Clone Repository & Launch Jupyter Notebook Offline](#step-16-clone-repository--launch-jupyter-notebook-offline)
3. [Part 2: PyLabFlow Core Architecture & Mental Model](#part-2-pylabflow-core-architecture--mental-model)
   - [The Problem: Folder Illusion & Notebook Sprawl](#the-problem-folder-illusion--notebook-sprawl)
   - [The Three-Tier Architecture: Component, WorkFlow, PipeLine](#the-three-tier-architecture-component-workflow-pipeline)
   - [The Lab Environment & Local SQLite Engine](#the-lab-environment--local-sqlite-engine)
4. [Part 3: Hands-On Lab 1 — Classical Machine Learning & Benchmarking](#part-3-hands-on-lab-1--classical-machine-learning--benchmarking)
   - [Lab 1 Workflow Summary](#lab-1-workflow-summary)
   - [Step 3.1: Initialize the ML Lab Workspace (`setup.ipynb`)](#step-31-initialize-the-ml-lab-workspace-setupipynb)
   - [Step 3.2: Inspect Components in `MLComps_ubucon/myComps.py`](#step-32-inspect-components-in-mlcomps_ubuconmycompspy)
   - [Step 3.3: Unit-Test Components in Isolation (`deBug.ipynb`)](#step-33-unit-test-components-in-isolation-debugipynb)
   - [Step 3.4: Configure Declarative Pipeline Dictionary (`ppl_config`)](#step-34-configure-declarative-pipeline-dictionary-ppl_config)
   - [Step 3.5: Register, Prepare, and Run Pipeline Trial 1](#step-35-register-prepare-and-run-pipeline-trial-1)
   - [Step 3.6: Run Multi-Model Benchmarks (Ridge & Random Forest)](#step-36-run-multi-model-benchmarks-ridge--random-forest)
   - [Step 3.7: Analyze and Compare Results Leaderboard (`Ana.ipynb`)](#step-37-analyze-and-compare-results-leaderboard-anaipynb)
5. [Part 4: Hands-On Lab 2 — Deep Learning MNIST Classification](#part-4-hands-on-lab-2--deep-learning-mnist-classification)
   - [Lab 2 Workflow Summary](#lab-2-workflow-summary)
   - [Step 4.1: Initialize the Deep Learning Lab Workspace (`setup.ipynb`)](#step-41-initialize-the-deep-learning-lab-workspace-setupipynb)
   - [Step 4.2: Inspect Components in `StarterCompDir_ubucon/`](#step-42-inspect-components-in-startercompdir_ubucon)
   - [Step 4.3: Unit-Test DL Components in Isolation (`experiment.ipynb`)](#step-43-unit-test-dl-components-in-isolation-experimentipynb)
   - [Step 4.4: Define Declarative Training Pipeline Configuration](#step-44-define-declarative-training-pipeline-configuration)
   - [Step 4.5: Execute Baseline Training Run `p1_baseline`](#step-45-execute-baseline-training-run-p1_baseline)
   - [Step 4.6: Live Progress Monitoring & Querying (`Monitor.ipynb`)](#step-46-live-progress-monitoring--querying-monitoripynb)
   - [Step 4.7: Hyperparameter Iteration (Pipeline `p2_deep_conv`)](#step-47-hyperparameter-iteration-pipeline-p2_deep_conv)
6. [Part 5: PyLabFlow Command-Line Interface (CLI)](#part-5-pylabflow-command-line-interface-cli)
7. [Part 6: Multi-OS Troubleshooting Matrix & FAQs](#part-6-multi-os-troubleshooting-matrix--faqs)

---

## 1. Workshop Overview & Philosophy

**PyLabFlow (PLF)** is a local-first, reproducible framework for computational science, machine learning, and deep learning experiments.

### The Research Problem It Solves
Traditional machine learning and scientific experimentation often suffer from **Jupyter Notebook Sprawl** and the **"Folder Illusion"**:
- Cells executed out of order lead to hidden state and non-reproducible models.
- Hardcoded hyperparameters, datasets, and file paths make model comparisons noisy.
- Model weight files (`best_weights_v2_final(1).pt`) lose their connection to the exact code and hyperparameters that created them.
- Cloud tracking services (W&B, Neptune, MLflow Cloud) require internet connections, create vendor lock-in, and present privacy risks for sensitive data.

**PyLabFlow eliminates these issues by providing:**
1. **Component Decoupling**: Isolate datasets, models, scalers, and evaluators into clean Python classes.
2. **Declarative Pipelines**: Define experiments with immutable configuration dictionaries that are verified with cryptographic hashes.
3. **Local-First Air-Gapped Tracking**: All experiment metadata, metrics, and models are tracked locally in lightweight SQLite databases (`ppls.db`). No servers or cloud accounts required.

---

# Part 1: Pre-Requisites & Offline Environment Setup (All OS)

Please complete this setup on your laptop before attending the workshop. Once packages are installed, **all workshop labs run 100% offline without internet**.

### Hardware Requirements
- **OS:** Ubuntu / Linux, macOS (Intel or Apple Silicon M1-M4), Windows 10/11.
- **RAM:** Minimum 8 GB (16 GB recommended).
- **Disk Space:** At least 5 GB free disk space.
- **GPU (Optional):** NVIDIA GPU with CUDA. *(Note: CPU-only mode is 100% supported; all exercises are optimized to run quickly on CPU).*

---

### Complete Dependency Manifest

All required libraries across both workshop tracks:

| Library | Minimum Version | Category | Purpose in Workshop |
| :--- | :--- | :--- | :--- |
| **`PyLabFlow`** | `>=0.1.0` | Core | Experiment orchestrator, Lab manifest, DAG workflow engine, lineage tracking |
| **`scikit-learn`** | `>=1.2.0` | Classical ML | Regression models (Linear, Ridge, Lasso, Random Forest), scalers, Diabetes dataset |
| **`pandas`** | `>=1.4.0` | Classical ML | Tabular data manipulation and multi-run experiment leaderboards |
| **`joblib`** | `>=1.2.0` | Classical ML | Model serialization and pipeline artifact persistence |
| **`numpy`** | `>=1.22.0` | Math / Arrays | Multi-dimensional array operations and tensor transformations |
| **`matplotlib`** | `>=3.5.0` | Visualization | Plotting evaluation curves, metric comparisons, and diagnostic plots |
| **`jupyter`** | `>=1.0.0` | Environment | Offline web-based notebook interface for running workshop `.ipynb` notebooks |
| **`ipykernel`** | `>=6.0.0` | Environment | Connects your isolated conda environment to Jupyter Notebook |
| **`torch`** | `>=2.0.0` | Deep Learning | PyTorch neural network modules (`nn.Module`), autograd, and optimizers |
| **`torchvision`** | `>=0.15.0` | Deep Learning | MNIST image dataset downloading and tensor transformation pipelines |

---

### Step 1.1: Install Miniconda or Anaconda

If Conda is not already installed, install **Miniconda**:

#### 🐧 Linux (Ubuntu / Debian / Fedora)
```bash
# 1. Download installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 2. Run batch install
bash miniconda.sh -b -p $HOME/miniconda3

# 3. Initialize conda for bash
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 4. Verify
conda --version
```

#### 🍏 macOS (Apple Silicon M1-M4 & Intel)
```bash
# For Apple Silicon (M1/M2/M3/M4):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# For Intel-based Macs:
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda for zsh (default shell on macOS):
$HOME/miniconda3/bin/conda init zsh
source ~/.zshrc

# Verify
conda --version
```

#### 🪟 Windows 10/11
1. Download the [Miniconda Windows 64-bit Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).
2. Run the `.exe` installer with default settings.
3. Open the **"Anaconda Prompt (miniconda3)"** or **"Miniconda Prompt"** from the Start Menu.
4. Verify:
   ```cmd
   conda --version
   ```
*(Note for Windows users: Run all subsequent workshop commands inside this Anaconda/Miniconda Prompt).*

---

### Step 1.2: Create Dedicated Conda Environment

Never install packages in your base environment. Create an isolated environment named **`ubucnplf`** with **Python 3.10**:

```bash
# 1. Create the dedicated environment
conda create -n ubucnplf python=3.10 -y

# 2. Activate the environment
conda activate ubucnplf
```
*(Your command prompt should now display `(ubucnplf)`).*

---

### Step 1.3: Install All Required Python Libraries

Inside your active `ubucnplf` environment, run:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install PyLabFlow, ML stack, and Jupyter
pip install PyLabFlow scikit-learn pandas joblib numpy matplotlib jupyter
```

#### Install Deep Learning Stack (PyTorch & TorchVision):

- **🟢 For Machines with NVIDIA GPU (CUDA Acceleration):**
  ```bash
  # Check GPU recognition
  nvidia-smi

  # Install PyTorch with CUDA 12.1
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  *(If your system uses CUDA 11.8, replace `cu121` with `cu118`)*.

- **🔵 For CPU-Only Machines (All Laptops without NVIDIA GPU / macOS):**
  ```bash
  # Linux & Windows (CPU optimized):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  # macOS (Leverages Apple Silicon Metal MPS acceleration automatically):
  pip install torch torchvision
  ```

---

### Step 1.4: Launch Jupyter Notebook Directly

Because `pip install jupyter` automatically configures the environment kernel, you simply activate the environment and launch Jupyter:

```bash
# 1. Activate the environment
conda activate ubucnplf

# 2. Launch Jupyter Notebook
jupyter notebook
```

---

### Step 1.5: Offline Verification Script

Test that every single required package imports cleanly:

```bash
python - << 'EOF'
import sys

def check(name, mod=None):
    if mod is None: mod = name
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'Installed')
        print(f"  [PASS] {name:<16} : {ver}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {name:<16} : MISSING ({e})")
        return False

print("=" * 65)
print(" 🧪 PyLabFlow Workshop — Offline Environment Verification")
print("=" * 65)
print(f"Python Version: {sys.version.split()[0]}")
print("\nChecking Required Packages:")
print("-" * 65)

packages = [
    ("PyLabFlow", "plf"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("jupyter", "notebook"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
]

all_ok = all([check(p, m) for p, m in packages])
print("-" * 65)
import torch
cuda_ok = torch.cuda.is_available()
print(f"PyTorch Hardware Mode    : {'CUDA GPU Acceleration' if cuda_ok else 'CPU Mode'}")
if cuda_ok:
    print(f"Detected GPU             : {torch.cuda.get_device_name(0)}")
else:
    print(f"Notice                   : Running on CPU (Fully supported for all workshop labs)")

print("=" * 65)
if all_ok:
    print("🚀 SUCCESS: ALL PREREQUISITES VERIFIED! ENVIRONMENT READY!")
else:
    print("⚠️  WARNING: Some packages failed to import. Review Step 1.3 above.")
print("=" * 65)
EOF
```

---

### Step 1.6: Clone Repository & Launch Jupyter Notebook Offline

For full multi-OS installation scripts and details, see [Prerequisites & Offline Setup Guide](WORKSHOP_PREREQUISITES.md).

```bash
# 1. Clone this workshop repository
git clone https://github.com/ExperQuick/PyLabFlow-Workshop.git
cd PyLabFlow-Workshop

# 2. Activate the dedicated environment
conda activate ubucnplf

# 3. Launch Jupyter Notebook
jupyter notebook
```

> ⚠️ **Kernel Selection Rule:** Whenever you open a notebook (`.ipynb`), verify that the kernel in the upper right reads **`Python (ubucnplf)`**. If it shows anything else, click it and select **`Python (ubucnplf)`**.

---

# Part 2: PyLabFlow Core Architecture & Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                           The Lab                               │
│      (Managed via plf.lab.create_project & plf.lab.lab_setup)   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     Components                          │   │
│   │        Inherit from plf.utils.Component                 │   │
│   │                                                         │   │
│   │   [DataLoader]    [Preprocessor]   [Model]   [Evaluator]│   │
│   └────────────────────────────┬────────────────────────────┘   │
│                                │ Loaded dynamically             │
│   ┌────────────────────────────▼────────────────────────────┐   │
│   │                      WorkFlow                           │   │
│   │        Inherits from plf.utils.WorkFlow                 │   │
│   │                                                         │   │
│   │   • Defines DAG: loader -> preprocessor -> model -> eval│   │
│   │   • Defines artifact paths: get_path(of, pplid, args)   │   │
│   └────────────────────────────┬────────────────────────────┘   │
│                                │ Executed via                   │
│   ┌────────────────────────────▼────────────────────────────┐   │
│   │                      PipeLine                           │   │
│   │        Managed via plf.experiment.PipeLine              │   │
│   │                                                         │   │
│   │   P.verify()  ──>  P.new(pplid)  ──>  P.prepare()  ──>   │   │
│   │   P.run()     ──>  get_ppls() / filter_ppls()           │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

PyLabFlow organizes machine learning workflows into three tiers:
1. **Component (`plf.utils.Component`)**: An atomic unit of computation (e.g. data loading, feature scaling, neural network, metric evaluation).
2. **WorkFlow (`plf.utils.WorkFlow`)**: The blueprint/DAG defining how components pass data to one another, when training runs, and where outputs are stored.
3. **PipeLine (`plf.experiment.PipeLine`)**: A concrete execution instance defined by an immutable configuration dictionary and identified by a unique `pplid` and cryptographic hash.

All metadata is stored locally in an embedded SQLite database (`ppls.db`).

---

# Part 3: Hands-On Lab 1 — Classical Machine Learning & Benchmarking

*Track: Classical Machine Learning with Scikit-Learn & PyLabFlow*  
*Objective: Build a modular regression pipeline, test components in isolation, and benchmark Linear, Ridge, and Random Forest models on Diabetes Progression Prediction.*

```text
ML/
├── MLComps_ubucon/          <-- Provided Starter Components
│   └── myComps.py           # All modular ML classes (Loader, Scalers, Models, Evaluator)
└── WorkSpace/               <-- Interactive Lab Workspace
    ├── setup.ipynb          # Step 1: Project initialization & manifest creation
    ├── deBug.ipynb          # Step 2: Component-level unit testing
    ├── exptrail.ipynb       # Steps 3 & 4: Running pipeline trials
    └── Ana.ipynb            # Step 5: Comparative metric analysis & leaderboards
```

---

### Step 3.1: Initialize the ML Lab Workspace (`setup.ipynb`)

Open `ML/WorkSpace/setup.ipynb` and initialize your lab:

```python
from plf.lab import create_project

# Define lab settings
settings = {
    "project_name": "PLF_mL",       # Name of the project
    "project_dir": "./",            # Base directory for outputs and logs
    "component_dir": "../MLComps"   # Directory where custom ML components live
}

# Create the project manifest
settings_path = create_project(settings=settings)

print("✅ Lab created successfully!")
print("Settings JSON manifest path:", settings_path)
```
*(This generates `PLF_mL.json`, which acts as the root configuration for all subsequent notebooks).*

> 💡 **Component Folder Flow**: The starter classes are provided in `MLComps_ubucon/`. When `create_project()` initializes the workspace, it establishes the target component directory (`MLComps`). You will move or copy the files (or rename the folder) into place as explained during the workshop.

---

### Step 3.2: Inspect Components in `MLComps_ubucon/myComps.py`

The starter file `MLComps_ubucon/myComps.py` contains the modular building blocks:

```python
from plf.utils import Component, WorkFlow
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib, json, os

# 1. DataLoader Component
class DataLoaderComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"shuffle", "sample_size", "feature_indices"}
    def _setup(self, args):
        self.shuffle = args.get("shuffle", False)
        self.sample_size = args.get("sample_size", None)
        self.feature_indices = args.get("feature_indices", None)
    def run(self):
        data = load_diabetes()
        X, y = data.data, data.target
        if self.shuffle:
            p = np.random.permutation(len(X))
            X, y = X[p], y[p]
        if self.sample_size:
            X, y = X[:self.sample_size], y[:self.sample_size]
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
        return X, y

# 2. Preprocessor Component
class PreprocessorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"scaler_type", "log_transform"}
    def _setup(self, args):
        self.scaler_type = args.get("scaler_type", "minmax")
        self.log_transform = args.get("log_transform", False)
    def run(self, X):
        if self.scaler_type == "minmax": scaler = MinMaxScaler()
        elif self.scaler_type == "standard": scaler = StandardScaler()
        elif self.scaler_type == "robust": scaler = RobustScaler()
        else: scaler = None
        X_scaled = scaler.fit_transform(X) if scaler else X
        if self.log_transform:
            X_scaled = np.log1p(np.abs(X_scaled))
        return X_scaled, scaler

# 3. Regressor Component
class RegressorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"model_type", "model_params"}
    def _setup(self, args):
        self.model_type = args.get("model_type", "linear")
        self.model_params = args.get("model_params", {})
    def run(self, X, y):
        if self.model_type == "linear": model = LinearRegression(**self.model_params)
        elif self.model_type == "ridge": model = Ridge(**self.model_params)
        elif self.model_type == "lasso": model = Lasso(**self.model_params)
        elif self.model_type == "rf": model = RandomForestRegressor(**self.model_params)
        model.fit(X, y)
        return model

# 4. Evaluator Component
class EvaluatorComponent(Component):
    def __init__(self):
        super().__init__()
        self.args = {"metrics"}
    def _setup(self, args):
        self.metrics = args.get("metrics", ["r2"])
    def run(self, model, X, y):
        y_pred = model.predict(X)
        res = {}
        if "r2" in self.metrics: res["r2"] = float(r2_score(y, y_pred))
        if "rmse" in self.metrics: res["rmse"] = float(np.sqrt(mean_squared_error(y, y_pred)))
        if "mae" in self.metrics: res["mae"] = float(mean_absolute_error(y, y_pred))
        return res

# 5. SimpleRegressionWorkflow
class SimpleRegressionWorkflow(WorkFlow):
    def __init__(self):
        super().__init__()
        self.template = {"loader", "preprocessor", "regressor", "evaluator"}
        self.paths = {"model", "metrics"}
    def get_path(self, of, pplid, args):
        if of == "model": return f"models/{pplid}.pkl"
        elif of == "metrics": return f"metrics/{pplid}.json"
        return None
    def run(self):
        X, y = self.comps["loader"].run()
        X_scaled, scaler = self.comps["preprocessor"].run(X)
        model = self.comps["regressor"].run(X_scaled, y)
        metrics = self.comps["evaluator"].run(model, X_scaled, y)
        
        # Save model checkpoint
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, self.get_path("model", self.pplid, None))
        
        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        with open(self.get_path("metrics", self.pplid, None), "w") as f:
            json.dump(metrics, f)
        return metrics
    def status(self):
        metric_file = self.get_path("metrics", self.pplid, None)
        if os.path.exists(metric_file):
            with open(metric_file, "r") as f: return json.load(f)
        return {}
```

---

### Step 3.3: Unit-Test Components in Isolation (`deBug.ipynb`)

Open `deBug.ipynb` to verify each component before running pipelines:

```python
from plf.lab import lab_setup
from plf.utils import load_component

# Connect to lab
lab_setup(settings_path="PLF_mL.json")

# 1. Test DataLoaderComponent
ds = load_component(
    loc="myComps.DataLoaderComponent", 
    args={"shuffle": True, "sample_size": 200, "feature_indices": [0, 2, 3, 4, 8]}
)
X, y = ds.run()
print("✅ DataLoaderComponent tested. X shape:", X.shape, "y shape:", y.shape)

# 2. Test PreprocessorComponent
pp = load_component(
    loc="myComps.PreprocessorComponent", 
    args={"scaler_type": "robust", "log_transform": False}
)
X_scaled, scaler = pp.run(X)
print("✅ PreprocessorComponent tested. Scaled shape:", X_scaled.shape)

# 3. Test RegressorComponent
rg = load_component(
    loc="myComps.RegressorComponent", 
    args={"model_type": "ridge", "model_params": {"alpha": 1.0}}
)
model = rg.run(X_scaled, y)
print("✅ RegressorComponent tested. Model:", type(model).__name__)

# 4. Test EvaluatorComponent
ev = load_component(
    loc="myComps.EvaluatorComponent", 
    args={"metrics": ["r2", "rmse", "mae"]}
)
metrics = ev.run(model, X_scaled, y)
print("✅ EvaluatorComponent tested. Metrics:", metrics)
```

---

### Step 3.4: Configure Declarative Pipeline Dictionary (`ppl_config`)

Open `exptrail.ipynb` and define the baseline configuration:

```python
ppl_config = {
    'workflow': {
        'loc': "myComps.SimpleRegressionWorkflow",
        'args': {}
    },
    'args': {
        'loader': {
            'loc': 'myComps.DataLoaderComponent',
            'args': {
                'shuffle': True,
                'sample_size': 400,
                'feature_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # All features
            }
        },
        'preprocessor': {
            'loc': "myComps.PreprocessorComponent",
            'args': {
                'scaler_type': "standard",
                'log_transform': False
            }
        },
        'regressor': {
            'loc': "myComps.RegressorComponent",
            'args': {
                'model_type': "linear",
                'model_params': {}
            }
        },
        'evaluator': {
            'loc': "myComps.EvaluatorComponent",
            'args': {
                'metrics': ["r2", "rmse", "mae"]
            }
        }
    }
}
```

---

### Step 3.5: Register, Prepare, and Run Pipeline Trial 1

```python
from plf.experiment import PipeLine

P = PipeLine()

# 1. Verify if this configuration already ran
existing_id = P.verify(cnfg=ppl_config)

if not existing_id:
    # 2. Create new pipeline
    P.new(pplid='exp1_linear_standard', args=ppl_config)

    # 3. Prepare components
    P.prepare()

    # 4. Run pipeline
    print("🚀 Running pipeline 'exp1_linear_standard'...")
    P.run()
    print("🎉 Pipeline 'exp1_linear_standard' finished!")
else:
    print(f"Config already executed under ID: {existing_id}")
```

---

### Step 3.6: Run Multi-Model Benchmarks (Ridge & Random Forest)

Benchmark Ridge and Random Forest models by cloning and modifying the configuration dictionary:

#### Trial 2: Ridge Regression with MinMax Scaler
```python
from copy import deepcopy

cfg_ridge = deepcopy(ppl_config)
cfg_ridge['args']['preprocessor']['args']['scaler_type'] = "minmax"
cfg_ridge['args']['regressor']['args']['model_type'] = "ridge"
cfg_ridge['args']['regressor']['args']['model_params'] = {"alpha": 0.5}

P = PipeLine()
if not P.verify(cnfg=cfg_ridge):
    P.new(pplid='exp2_ridge_minmax', args=cfg_ridge)
    P.prepare()
    P.run()
```

#### Trial 3: Random Forest Regressor with Robust Scaler
```python
cfg_rf = deepcopy(ppl_config)
cfg_rf['args']['preprocessor']['args']['scaler_type'] = "robust"
cfg_rf['args']['regressor']['args']['model_type'] = "rf"
cfg_rf['args']['regressor']['args']['model_params'] = {
    "n_estimators": 100, 
    "max_depth": 5, 
    "random_state": 42
}

P = PipeLine()
if not P.verify(cnfg=cfg_rf):
    P.new(pplid='exp3_rf_robust', args=cfg_rf)
    P.prepare()
    P.run()
```

---

### Step 3.7: Analyze and Compare Results Leaderboard (`Ana.ipynb`)

Open `Ana.ipynb` to aggregate metrics across all pipelines into a leaderboard:

```python
from plf.lab import lab_setup
from plf.experiment import get_ppl_status
import pandas as pd

# Connect to lab
lab_setup(settings_path="PLF_mL.json")

# Fetch all metrics across all runs
status_dict = get_ppl_status()

# Convert to DataFrame
df_results = pd.DataFrame.from_dict(status_dict, orient='index')

# Sort by R² score (highest accuracy first)
print("🏆 Model Benchmark Leaderboard:")
display(df_results.sort_values(by='r2', ascending=False))
```

Example Leaderboard Output:
| Pipeline ID | R² Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| `exp3_rf_robust` | **0.542** | **52.1** | **41.3** |
| `exp1_linear_standard` | 0.518 | 54.3 | 43.2 |
| `exp2_ridge_minmax` | 0.495 | 55.8 | 44.7 |

All models are deterministically saved in `models/<pplid>.pkl` and metrics in `metrics/<pplid>.json`.

---

# Part 4: Hands-On Lab 2 — Deep Learning MNIST Classification

*Track: Deep Learning with PyTorch & PyLabFlow*  
*Objective: Build, test, orchestrate, and audit a modular CNN pipeline on MNIST.*

```text
DL/
├── StarterCompDir_ubucon/           <-- Provided Starter Components
│   ├── my_datasets.py               # Data loader & transforms
│   ├── my_models.py                 # Neural network architectures
│   └── my_workflows.py              # End-to-end training & evaluation logic
└── WorkSpace/                       <-- Interactive Lab Workspace
    ├── setup.ipynb                  # Step 1: Lab creation & manifest setup
    ├── experiment.ipynb             # Step 2: Interactive component testing & training
    └── Monitor.ipynb                # Step 3: Multi-pipeline live progress monitor
```

---

### Step 4.1: Initialize the Deep Learning Lab Workspace (`setup.ipynb`)

Open `DL/WorkSpace/setup.ipynb` and initialize the lab:

```python
from plf.lab import create_project

# Define lab settings
settings = {
    "project_name": "PLF_Starter",      # Name of the project workspace
    "project_dir": "./",                # Base directory where logs & models will live
    "component_dir": "../StarterCompDir"# Directory where your custom component files live
}

# Create the project
settings_path = create_project(settings)

print("✅ Lab created successfully!")
print("Settings JSON manifest path:", settings_path)
```
*(This generates `PLF_Starter.json`).*

> 💡 **Component Folder Flow**: The starter classes are provided in `StarterCompDir_ubucon/`. When `create_project()` initializes the workspace, it sets up the target component directory (`StarterCompDir`). You will move or copy the files (or rename the folder) into place as explained during the workshop.

---

### Step 4.2: Inspect Components in `StarterCompDir_ubucon/`

The directory contains three modular files:

#### 1. `StarterCompDir_ubucon/my_datasets.py`
```python
from plf.utils import Component
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

class Augment(Component):
    def __init__(self):
        super().__init__()
        self.args = {}
    def run(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

class MNISTDataset(Component):
    def __init__(self):
        super().__init__()
        self.args = {"root_dir", "train", "transform"}
    def _setup(self, args):
        self.root = args.get("root_dir", "./data")
        self.train = args.get("train", True)
        self.tf_cnfg = args.get("transform", None)
    def run(self):
        from plf.utils import load_component
        tf = load_component(**self.tf_cnfg).run() if self.tf_cnfg else None
        return MNIST(root=self.root, train=self.train, download=True, transform=tf)
```

#### 2. `StarterCompDir/my_models.py`
```python
from plf.utils import Component
import torch
import torch.nn as nn

class Model1st(Component, nn.Module):
    def __init__(self):
        Component.__init__(self)
        nn.Module.__init__(self)
        self.args = {"conv_deep", "dense_deep"}
    def _setup(self, args):
        conv_factor = args.get("conv_deep", 1)
        dense_factor = args.get("dense_deep", 1)
        
        c1 = 16 * conv_factor
        c2 = 32 * conv_factor
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        fc_in = c2 * 7 * 7
        fc_hidden = 64 * dense_factor
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

#### 3. `StarterCompDir/my_workflows.py`
```python
from plf.utils import WorkFlow
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os, json

class BasicWorkFlow(WorkFlow):
    def __init__(self):
        super().__init__()
        self.template = {"dataset", "model", "batch_size"}
        self.paths = {"model", "metrics"}
        self.args = {"num_epochs"}
    def _setup(self, args):
        self.num_epochs = args.get("num_epochs", 3)
    def get_path(self, of, pplid, args):
        if of == "model": return f"checkpoints/{pplid}_model.pt"
        elif of == "metrics": return f"metrics/{pplid}.json"
        return None
    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = self.comps["dataset"].run()
        loader = DataLoader(dataset, batch_size=self.args["batch_size"], shuffle=True)
        model = self.comps["model"].to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            history.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save model
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), self.get_path("model", self.pplid, None))
        
        # Save metrics
        os.makedirs("metrics", exist_ok=True)
        final_metrics = {"history": history, "final_loss": history[-1]["loss"]}
        with open(self.get_path("metrics", self.pplid, None), "w") as f:
            json.dump(final_metrics, f)
        return final_metrics
    def status(self):
        p = self.get_path("metrics", self.pplid, None)
        if os.path.exists(p):
            with open(p, "r") as f: return json.load(f)
        return {}
```

---

### Step 4.3: Unit-Test DL Components in Isolation (`experiment.ipynb`)

Open `experiment.ipynb` to verify your dataset and CNN forward pass:

```python
from plf.lab import lab_setup
from plf.utils import load_component

# Connect to lab
lab_setup(settings_path="PLF_Starter.json")

# 1. Test Dataset
ds_cnfg = {
    'loc': 'my_datasets.MNISTDataset',
    'args': {
        'root_dir': './MNISTdata',
        'train': True,
        'transform': {'loc': 'my_datasets.Augment', 'args': {}}
    }
}
dataset_comp = load_component(**ds_cnfg)
mnist_train = dataset_comp.run()
sample_img, sample_lbl = mnist_train[0]
print("✅ Dataset loaded. Image tensor shape:", sample_img.shape, "Label:", sample_lbl)

# 2. Test Model forward pass
model_cnfg = {
    'loc': 'my_models.Model1st',
    'args': {'conv_deep': 1, 'dense_deep': 1}
}
model = load_component(**model_cnfg)
output_logits = model(sample_img.unsqueeze(0)) # Shape: (1, 1, 28, 28)
print("✅ Model forward pass successful. Logits shape:", output_logits.shape)
```

---

### Step 4.4: Define Declarative Training Pipeline Configuration

Define the pipeline configuration dictionary:

```python
ppl_cnfg = {
    'workflow': {
        'loc': 'my_workflows.BasicWorkFlow',
        'args': {'num_epochs': 5}  # Train for 5 epochs
    },
    'args': {
        'dataset': {
            'loc': 'my_datasets.MNISTDataset',
            'args': {
                'root_dir': './MNISTdata',
                'train': True,
                'transform': {'loc': 'my_datasets.Augment', 'args': {}}
            }
        },
        'model': {
            'loc': 'my_models.Model1st',
            'args': {
                'conv_deep': 2,    # Convolutional depth factor = 2
                'dense_deep': 1     # Dense layer factor = 1
            }
        },
        'batch_size': 64
    }
}
```

---

### Step 4.5: Execute Baseline Training Run `p1_baseline`

```python
from plf.experiment import PipeLine

P = PipeLine()

# 1. Verify if identical config ran before
existing_id = P.verify(cnfg=ppl_cnfg)

if not existing_id:
    # 2. Register new pipeline
    P.new(pplid='p1_baseline', args=ppl_cnfg)

    # 3. Prepare components
    P.prepare()

    # 4. Run training
    print("🚀 Running training pipeline 'p1_baseline'...")
    P.run()
    print("🎉 Pipeline 'p1_baseline' finished successfully!")
else:
    print(f"⚠️ Identical configuration already executed under: {existing_id}")
```

---

### Step 4.6: Live Progress Monitoring & Querying (`Monitor.ipynb`)

Open `Monitor.ipynb` to view all executed runs and metrics:

```python
from plf.lab import lab_setup
from plf.experiment import get_ppls, get_ppl_status, get_ppl_details

# Attach to lab
lab_setup(settings_path="PLF_Starter.json")

# 1. View all pipelines registered so far
df_ppls = get_ppls()
display(df_ppls)

# 2. View execution status and recorded metrics
status = get_ppl_status()
display(status)

# 3. View detailed component arguments for all pipelines
details = get_ppl_details()
display(details)
```

---

### Step 4.7: Hyperparameter Iteration (Pipeline `p2_deep_conv`)

Hypothesis: **"Does a deeper convolutional model (`conv_deep = 4`) reduce loss faster?"**

```python
from copy import deepcopy

# 1. Clone existing config
p2_cnfg = deepcopy(ppl_cnfg)

# 2. Modify hyperparameter
p2_cnfg['args']['model']['args']['conv_deep'] = 4

# 3. Execute pipeline p2
P = PipeLine()
existing_id = P.verify(cnfg=p2_cnfg)

if not existing_id:
    P.new(pplid='p2_deep_conv', args=p2_cnfg)
    P.prepare()
    print("🚀 Training pipeline 'p2_deep_conv'...")
    P.run()
    print("🎉 Pipeline 'p2_deep_conv' completed!")
```

#### Compare `p1_baseline` vs `p2_deep_conv`:
```python
from plf.experiment import get_ppl_status

status = get_ppl_status()
print("p1_baseline final loss:", status.get('p1_baseline', {}).get('final_loss'))
print("p2_deep_conv final loss:", status.get('p2_deep_conv', {}).get('final_loss'))
```

---

# Part 5: PyLabFlow Command-Line Interface (CLI)

PyLabFlow includes a native command-line utility (`plf`) that automatically discovers your workspace manifest by traversing parent directories (just like `git` finds `.git/`):

```bash
# Display help and available commands
plf --help

# View the status table of all pipelines
plf status

# List active pipeline identifiers
plf list

# Execute a registered pipeline from the command line
plf run <pplid>

# Gracefully request running pipeline(s) to stop after the current iteration
plf stop

# Export pipeline configuration as YAML or JSON
plf export <pplid> --format yaml

# Archive completed pipelines to cold storage
plf archive <pplid>

# Permanently delete an archived pipeline
plf delete <pplid>

# Initialize a new Lab from a settings file
plf init --config settings.json
```

---

# Part 6: Multi-OS Troubleshooting Matrix & FAQs

| Issue / Error | Operating System | Root Cause | Solution |
| :--- | :--- | :--- | :--- |
| `conda: command not found` | **Linux / macOS** | Conda binaries not in `$PATH`. | Run `source ~/miniconda3/bin/activate` or close and reopen the terminal. |
| `conda: command not found` | **Windows** | Running inside standard `cmd.exe` instead of Anaconda Prompt. | Open the **Anaconda Prompt** or **Miniconda Prompt** from the Start Menu. |
| `ModuleNotFoundError: No module named 'plf'` in notebook | **All OS** | The notebook was launched with the `base` conda kernel instead of `ubucnplf`. | In Jupyter Notebook, go to **Kernel** > **Change Kernel...** and select **`Python (ubucnplf)`**. |
| `torch.cuda.is_available()` returns `False` | **Linux / Windows** | NVIDIA GPU drivers are missing or CPU build of PyTorch was installed. | Verify with `nvidia-smi`. Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu121`. If your machine does not have an NVIDIA GPU, this is expected: **CPU execution is 100% fine**. |
| Conda activation fails in bash script | **Linux** | Shell environment not hooked to conda. | Run `eval "$(conda shell.bash hook)"` before calling `conda activate ubucnplf`. |
| PowerShell script execution policy error | **Windows** | PowerShell execution policy blocks conda scripts. | In an Administrator PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or simply use the **Miniconda Prompt**. |

---

*PyLabFlow is developed under the Apache License 2.0. Have a great workshop!*

