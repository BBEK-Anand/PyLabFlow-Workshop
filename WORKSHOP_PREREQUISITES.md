# 🛠️ PyLabFlow Workshop: Prerequisites & Offline Environment Setup Guide

> **All-in-One Local Setup Guide for Workshop Attendees**  
> **Platforms Supported:** Ubuntu / Linux, macOS (Apple Silicon & Intel), Windows 10/11  
> **Execution Mode:** 100% Offline Local Machine Execution (No Cloud or Internet Required After Initial Install)  
> **Dedicated Conda Environment:** `ubucnplf` (Python 3.10)

---

## 📋 Pre-Workshop Checklist

Please complete these steps on your laptop **before arriving at the session**:

- [ ] **1. Install Miniconda (or Anaconda)** for your operating system.
- [ ] **2. Create and activate** the dedicated `ubucnplf` Conda environment (Python 3.10).
- [ ] **3. Install the required Python packages** (`PyLabFlow`, ML stack, PyTorch, Jupyter Notebook).
- [ ] **4. Register the kernel** (`ipykernel`) so notebooks run in `ubucnplf`.
- [ ] **5. Run the offline verification test** to confirm all imports pass.
- [ ] **6. Clone this repository** (`PyLabFlow-Workshop`) to your machine.

---

## 1. Hardware & OS Recommendations

- **Operating System:** Ubuntu / Linux, macOS (Intel or Apple Silicon M1–M4), or Windows 10/11.
- **RAM:** Minimum 8 GB (16 GB recommended).
- **Disk Space:** At least 5 GB free disk space for Conda, libraries, and datasets.
- **GPU (Optional):** NVIDIA GPU with CUDA drivers. **CPU-only mode is 100% supported**; all workshop labs run quickly on standard laptops without a dedicated GPU.

---

## 2. Required Python Libraries & Versions

All dependencies below are required across the **Classical ML** and **Deep Learning** workshop tracks:

| Package | Minimum Version | Category | Role in Workshop |
| :--- | :--- | :--- | :--- |
| **`PyLabFlow`** | `>=0.1.0` | Core | Experiment orchestrator, Lab manifest creator, DAG workflow engine, lineage tracking |
| **`scikit-learn`** | `>=1.2.0` | Classical ML | Regression models (Linear, Ridge, Lasso, Random Forest), scalers, Diabetes dataset |
| **`pandas`** | `>=1.4.0` | Classical ML | Tabular dataframes and multi-run experiment leaderboards |
| **`joblib`** | `>=1.2.0` | Classical ML | Serializing and saving trained models and pipeline artifacts |
| **`numpy`** | `>=1.22.0` | Math / Arrays | Multi-dimensional array manipulations and tensor transformations |
| **`matplotlib`** | `>=3.5.0` | Visualization | Plotting evaluation graphs, ROC curves, loss curves, and diagnostic figures |
| **`jupyter`** | `>=1.0.0` | Environment | Offline web-based notebook interface for running workshop `.ipynb` notebooks |
| **`ipykernel`** | `>=6.0.0` | Environment | Connects your isolated conda environment to Jupyter Notebook |
| **`torch`** | `>=2.0.0` | Deep Learning | PyTorch neural network modules (`nn.Module`), autograd, and optimizers |
| **`torchvision`** | `>=0.15.0` | Deep Learning | MNIST image dataset downloading and image tensor transforms |

---

## Step 1: Install Miniconda or Anaconda

If your computer does not already have Conda installed, install **Miniconda** (lightweight, ~100 MB).

### 🐧 Linux (Ubuntu / Debian / Fedora)
```bash
# 1. Download the latest Linux 64-bit installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 2. Run the installer in batch mode
bash miniconda.sh -b -p $HOME/miniconda3

# 3. Initialize conda for your bash shell
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 4. Verify installation
conda --version
```

### 🍏 macOS (Apple Silicon M1–M4 & Intel)
```bash
# Apple Silicon (M1 / M2 / M3 / M4):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# Intel-based Mac:
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3

# Initialize shell (zsh is default on macOS):
$HOME/miniconda3/bin/conda init zsh
source ~/.zshrc

# Verify installation:
conda --version
```

### 🪟 Windows 10/11
1. Download the [Miniconda Windows 64-bit Installer](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).
2. Double-click the installer and complete setup with default options.
3. Open the **"Anaconda Prompt (miniconda3)"** or **"Miniconda Prompt"** from your Start Menu.
4. Verify:
   ```cmd
   conda --version
   ```
*(Note for Windows users: Run all remaining workshop commands inside this Anaconda/Miniconda Prompt).*

---

## Step 2: Create & Activate the Dedicated Conda Environment

Always isolate the workshop dependencies in the dedicated environment named **`ubucnplf`** with **Python 3.10**:

```bash
# 1. Create the environment
conda create -n ubucnplf python=3.10 -y

# 2. Activate the environment
conda activate ubucnplf
```

> 💡 **Check**: Your terminal prompt will now show `(ubucnplf)`.

---

## Step 3: Install Required Python Libraries

With your `(ubucnplf)` environment activated:

### 1. Upgrade `pip`
```bash
python -m pip install --upgrade pip
```

### 2. Install PyLabFlow & Classical ML Stack
```bash
pip install PyLabFlow scikit-learn pandas joblib numpy matplotlib jupyter ipykernel
```

### 3. Install Deep Learning Stack (PyTorch & TorchVision)

Choose the command matching your laptop's hardware:

- **🟢 For Machines with NVIDIA GPU (CUDA Acceleration):**
  ```bash
  # Check GPU recognition
  nvidia-smi

  # Install PyTorch with CUDA 12.1:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  *(If your machine runs CUDA 11.8, replace `cu121` with `cu118`)*.

- **🔵 For CPU-Only Machines (Laptops without discrete NVIDIA GPU / macOS):**
  ```bash
  # Linux & Windows (CPU optimized):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

  # macOS (automatically leverages Apple Silicon Metal MPS acceleration):
  pip install torch torchvision
  ```

---

## Step 4: Register the Jupyter Notebook Kernel

Ensure Jupyter Notebook runs notebooks against the dedicated `ubucnplf` environment:

```bash
# Ensure ubucnplf is active
conda activate ubucnplf

# Register the kernel
python -m ipykernel install --user --name=ubucnplf --display-name "Python (ubucnplf)"
```

---

## Step 5: Run Pre-Flight Verification

Verify that all packages import without error before attending the workshop.

### Quick One-Line Check
```bash
python -c "import plf, torch, sklearn, pandas, numpy, joblib, matplotlib; print('🎉 SUCCESS: All workshop libraries are ready!')"
```

### Comprehensive Verification Script
Run this script directly in your terminal:

```bash
python - << 'EOF'
import sys

def check_package(display_name, module_name=None):
    if module_name is None:
        module_name = display_name
    try:
        mod = __import__(module_name)
        ver = getattr(mod, "__version__", "Installed")
        print(f"  [PASS] {display_name:<16} : {ver}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {display_name:<16} : MISSING ({e})")
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
    ("ipykernel", "ipykernel"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
]

all_ok = all([check_package(disp, mod) for disp, mod in packages])

print("-" * 65)
import torch
cuda_ok = torch.cuda.is_available()
print(f"PyTorch Execution Mode   : {'NVIDIA CUDA Acceleration' if cuda_ok else 'CPU Mode'}")
if cuda_ok:
    print(f"Detected GPU             : {torch.cuda.get_device_name(0)}")
else:
    print(f"Notice                   : Running on CPU (Fully supported for all workshop labs)")

print("=" * 65)
if all_ok:
    print("🚀 SUCCESS: ENVIRONMENT IS READY FOR OFFLINE WORKSHOP LABS!")
else:
    print("⚠️  WARNING: Some packages failed to import. Review Step 3 above.")
print("=" * 65)
EOF
```

---

## Step 6: Clone Repository & Launch Workshop Labs

### 1. Clone the Workshop Repository
```bash
git clone https://github.com/ExperQuick/PyLabFlow-Workshop.git
cd PyLabFlow-Workshop
```

### 2. Workshop Directory Structure
The repository provides two independent hands-on tracks:

```text
PyLabFlow-Workshop/
│
├── ML/                          <-- 📊 Classical ML Track (Diabetes Progression)
│   ├── MLComps_ubucon/          <-- Provided Starter Components (myComps.py)
│   └── WorkSpace/               <-- Interactive Notebooks (setup, deBug, exptrail, Ana)
│
└── DL/                          <-- 🧠 Deep Learning Track (MNIST Digit Classification)
    ├── StarterCompDir_ubucon/   <-- Provided Starter Components (datasets, models, workflows)
    └── WorkSpace/               <-- Interactive Notebooks (setup, experiment, Monitor)
```

### 3. Launch Jupyter Notebook for Offline Labs
Always make sure you `cd` into the `PyLabFlow-Workshop` directory before launching Jupyter Notebook so all track folders and component files are visible:

```bash
# 1. Navigate to the workshop repository root
cd PyLabFlow-Workshop

# 2. Activate the dedicated environment
conda activate ubucnplf

# 3. Launch Jupyter Notebook
jupyter notebook
```

> 💡 **Selecting the Kernel:** When opening any workshop notebook, ensure the top-right kernel indicator displays **`Python (ubucnplf)`**. If not, click on **Kernel** > **Change Kernel...** and select `Python (ubucnplf)`.

---

## 🚨 Multi-OS Troubleshooting Matrix

| Issue | Operating System | Root Cause | Solution |
| :--- | :--- | :--- | :--- |
| `conda: command not found` | **Linux / macOS** | Conda binaries not in `$PATH`. | Run `source ~/miniconda3/bin/activate` or restart your terminal. |
| `conda: command not found` | **Windows** | Running in standard Command Prompt instead of Anaconda Prompt. | Open the **Anaconda Prompt** or **Miniconda Prompt** from the Start Menu. |
| `ModuleNotFoundError: No module named 'plf'` in notebook | **All OS** | The notebook was launched with the `base` conda kernel instead of `ubucnplf`. | In Jupyter Notebook, go to **Kernel** > **Change Kernel...** and select **`Python (ubucnplf)`**. |
| `torch.cuda.is_available()` returns `False` | **Linux / Windows** | NVIDIA GPU drivers missing or CPU build of PyTorch installed. | Run `nvidia-smi` to verify drivers. If you have an NVIDIA GPU, reinstall PyTorch with CUDA 12.1. If running without a GPU: **CPU mode is 100% fine**. |
| Conda activation fails in bash script | **Linux** | Shell environment not hooked to conda. | Run `eval "$(conda shell.bash hook)"` before calling `conda activate ubucnplf`. |
| PowerShell script execution policy error | **Windows** | PowerShell execution policy blocks conda activation scripts. | In an Administrator PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or use the **Miniconda Prompt**. |

---

## 📚 Next Steps: Workshop Guides

Once your offline environment is verified, proceed to:
- 📖 [Complete Participant Guide (WORKSHOP_COMPLETE_GUIDE.md)](WORKSHOP_COMPLETE_GUIDE.md) — End-to-end theory, architecture, Lab 1 & 2 walkthroughs, and CLI commands.
- 📊 [Classical ML Track README](ML/README.md)
- 🧠 [Deep Learning Track README](DL/README.md)
