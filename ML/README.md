# 📊 Classical Machine Learning: Diabetes Progression Prediction

Welcome to the **Classical Machine Learning Track** of the PyLabFlow Workshop.

All components, datasets, and notebooks for this track are self-contained within this `ML/` directory:
- **`MLComps_ubucon/`**: Provided starter component implementations ([`myComps.py`](MLComps_ubucon/myComps.py)) containing the DataLoader, Preprocessors (StandardScaler, MinMaxScaler, RobustScaler), Regression Models (Linear, Ridge, Lasso, RandomForest), and Evaluator.
- **`WorkSpace/`**: Interactive Jupyter notebooks to initialize, test, execute, and analyze pipelines.

---

### 🚀 Hands-On Workflow Guide

1. **Prerequisites & Launching Jupyter**:
   Navigate to the repository root, activate the environment, and launch Jupyter Notebook:
   ```bash
   cd PyLabFlow-Workshop
   conda activate ubucnplf
   jupyter notebook
   ```
   In the Jupyter Notebook interface, browse to `ML/WorkSpace/` to open the notebooks.

2. **Track Directory Structure**:
   ```text
   ML/
   ├── MLComps_ubucon/          <-- Provided Starter Components
   │   └── myComps.py           # Custom Component classes (Loader, Scalers, Models, Evaluator)
   └── WorkSpace/               <-- Interactive Notebook Workspace
       ├── setup.ipynb          # Step 1: Initialize the Lab & generate PLF_mL.json manifest
       ├── deBug.ipynb          # Step 2: Unit-test components in isolation
       ├── exptrail.ipynb       # Step 3: Configure, register, and run pipeline trials
       └── Ana.ipynb            # Step 4: Query SQLite database & compare model leaderboards
   ```

3. **Step-by-Step Execution**:
   - **Step 1 — Setup Lab (`WorkSpace/setup.ipynb`)**:
     Run this notebook first to initialize the PyLabFlow project workspace and generate the `PLF_mL.json` manifest.
     > 💡 **Component Folder Flow:** When `create_project()` runs, it initializes the project component directory. You will then move/copy the starter code from `MLComps_ubucon/` into the created component folder (or rename it), as walked through in the workshop.
   - **Step 2 — Debug & Unit Test (`WorkSpace/deBug.ipynb`)**:
     Test each component (`DataLoaderComponent`, `ScalerComponent`, `ModelTrainComponent`, `ModelEvaluateComponent`) independently to verify input/output contracts.
   - **Step 3 — Run Pipeline Experiments (`WorkSpace/exptrail.ipynb`)**:
     Define declarative pipeline configs (`ppl_config`), verify immutable cryptographic hashes with `P.verify()`, and train different model configurations (Linear Regression, Ridge, Random Forest).
   - **Step 4 — Leaderboard & Analysis (`WorkSpace/Ana.ipynb`)**:
     Query the local SQLite database (`ppls.db`) and compare model performance (R², RMSE, MAE) across runs.

---

### 🧑‍🔬 Experimentation Ideas

- **Feature Scalers**: Compare `StandardScaler`, `MinMaxScaler`, and `RobustScaler` to evaluate sensitivity to outliers.
- **Regression Algorithms**: Benchmark `LinearRegression` vs. `Ridge` (L2 regularization) vs. `RandomForestRegressor`.
- **Feature Selection**: Test training with all 10 features versus a subset (e.g. top 4 features `['bmi', 's5', 'bp', 's3']`).

For a complete conceptual deep dive and line-by-line code explanation, see the [Complete Workshop Guide](../WORKSHOP_COMPLETE_GUIDE.md#part-3-hands-on-lab-1--classical-machine-learning--benchmarking).
