# 🧠 Deep Learning: MNIST Digit Classification

Welcome to the **Deep Learning Track** of the PyLabFlow Workshop.

All components, neural network models, and notebooks for this track are self-contained within this `DL/` directory:
- **`StarterCompDir_ubucon/`**: Provided starter modular components:
  - [`my_datasets.py`](StarterCompDir_ubucon/my_datasets.py): MNIST data loader and tensor transforms.
  - [`my_models.py`](StarterCompDir_ubucon/my_models.py): PyTorch neural network architectures (configurable CNN / ConvNet).
  - [`my_workflows.py`](StarterCompDir_ubucon/my_workflows.py): End-to-end training loops, validation, optimizer configuration, and metrics.
- **`WorkSpace/`**: Interactive Jupyter notebooks to initialize, train, monitor, and iterate deep learning pipelines.

---

### 🚀 Hands-On Workflow Guide

1. **Prerequisites & Launching Jupyter**:
   Navigate to the repository root, activate the environment, and launch Jupyter Notebook:
   ```bash
   cd PyLabFlow-Workshop
   conda activate ubucnplf
   jupyter notebook
   ```
   In the Jupyter Notebook interface, browse to `DL/WorkSpace/` to open the notebooks.

2. **Track Directory Structure**:
   ```text
   DL/
   ├── StarterCompDir_ubucon/   <-- Provided Starter Components
   │   ├── my_datasets.py       # Data loader & transform components
   │   ├── my_models.py         # PyTorch ConvNet / CNN architectures
   │   └── my_workflows.py      # Training loop & evaluation workflow
   └── WorkSpace/               <-- Interactive Notebook Workspace
       ├── setup.ipynb          # Step 1: Initialize the Lab & generate PLF_Starter.json manifest
       ├── experiment.ipynb     # Step 2: Unit test components & execute training pipelines
       └── Monitor.ipynb        # Step 3: Live multi-pipeline progress monitoring & history inspection
   ```

3. **Step-by-Step Execution**:
   - **Step 1 — Setup Lab (`WorkSpace/setup.ipynb`)**:
     Initialize the project workspace to create the `PLF_Starter.json` manifest and prepare directory structures.
     > 💡 **Component Folder Flow:** When `create_project()` runs, it initializes the project component directory. You will then move/copy the starter code from `StarterCompDir_ubucon/` into the created component folder (or rename it), as walked through in the workshop.
   - **Step 2 — Experiment & Train (`WorkSpace/experiment.ipynb`)**:
     Test components in isolation, configure the declarative training pipeline dictionary (`ppl_cnfg`), and train:
     - Baseline pipeline `p1_baseline` (learning rate `1e-3`, batch size `64`).
     - Hyperparameter iteration `p2_deep_conv` (deeper conv layers `conv_deep=4`).
   - **Step 3 — Monitor Progress (`WorkSpace/Monitor.ipynb`)**:
     Track epoch-by-epoch loss and accuracy curves in real time, inspect training history, and compare pipelines side-by-side.

---

### 🧑‍🔬 Experimentation Ideas

- **Architecture Depth**: Compare `conv_deep=2` vs `conv_deep=4` vs `conv_deep=6` on convergence speed and accuracy.
- **Batch Size & Optimizer**: Compare `Adam` vs `SGD` or batch size `64` vs `128`.
- **Training Durations**: Observe training loss dynamics across 3 vs 5 epochs.

For a complete conceptual deep dive and line-by-line code explanation, see the [Complete Workshop Guide](../WORKSHOP_COMPLETE_GUIDE.md#part-4-hands-on-lab-2--deep-learning-mnist-classification).
