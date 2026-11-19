# LoanShield — Transparent Financial Credit Scoring & Risk Analytics

LoanShield is an academic Streamlit prototype for exploring credit-risk prediction, model explainability, customer segmentation, model evaluation, and an optional signature-image screening workflow using the **UCI German Credit dataset**.

> **Research-use warning:** LoanShield is not a real lending system. Its credit model uses a small, historical dataset and has not been calibrated or validated for present-day lending. Do not use it to approve or reject real applications. SHAP explains the behaviour of this model; it does not establish causation or prove creditworthiness.

## Included modules

1. **Customer Assessment** — 20 German Credit input fields, saved Random Forest inference, predicted Bad Credit probability, and an individual SHAP contribution chart and table.
2. **SHAP Audit** — global mean absolute SHAP importance and per-customer contribution distributions.
3. **Customer Segmentation** — K-Means with `k=3`, PCA projection, cluster profiles, and observed Bad Credit prevalence per cluster.
4. **Model Evaluation** — held-out test metrics, confusion matrix, ROC and precision-recall curves, plus five-fold cross-validation comparisons for Logistic Regression, Random Forest, and RBF-SVM.
5. **Document & Signature Verification** — image upload and preprocessing UI with an optional CNN inference path. No fabricated CNN weights are included; a labelled signature dataset is required before inference is enabled.

## Project structure

```text
LoanShield/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Customer_Assessment.py
│       ├── 2_SHAP_Audit.py
│       ├── 3_Customer_Segmentation.py
│       ├── 4_Model_Evaluation.py
│       └── 5_Document_Verification.py
├── data/
│   ├── raw/german.data
│   ├── processed/                 # generated/used by training and evaluation
│   └── signatures/                # optional user-provided labelled images
├── models/
│   ├── credit/                    # trained credit model and clustering artifacts
│   └── cnn/                       # generated after optional signature CNN training
├── reports/
│   ├── figures/                   # evaluation plots
│   └── results/                   # metrics, profiles, predictions and diagnostics
├── src/
│   ├── preprocessing/credit_preprocessor.py
│   ├── models/train_credit_model.py
│   ├── models/signature_cnn.py
│   ├── models/train_signature_cnn.py
│   ├── explainability/local_shap.py
│   ├── clustering/kmeans.py
│   └── evaluation/
├── tests/
├── requirements.txt
├── PROJECT_STATUS.md
└── README.md
```

The project uses `pathlib`-based paths, so the Python modules work with Windows and Linux path conventions. Run commands from the **LoanShield project root** unless otherwise stated.

---

# 1. Installation on Windows 10/11

These instructions use **Windows PowerShell** and Python 3.11 (64-bit is recommended). Install Python from [python.org](https://www.python.org/downloads/windows/) and enable the Python Launcher (`py`) during installation. Open a new PowerShell window after installing Python.

## 1.1 Open the project folder

Extract `LoanShield_complete_project.zip` to a folder, for example `C:\Users\<your-user>\Projects\LoanShield`, then open PowerShell in that folder:

```powershell
cd C:\Users\<your-user>\Projects\LoanShield
```

Check Python:

```powershell
py -3.11 --version
```

If the Python Launcher is unavailable but `python` points to Python 3.11, use `python` instead of `py -3.11` in the next commands.

## 1.2 Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The prompt should now show `(.venv)`. This execution-policy change applies only to the current PowerShell process.

**Windows Command Prompt alternative:** if you use `cmd.exe` instead of PowerShell:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

## 1.3 Install Python dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include Streamlit, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Pillow, PyTorch, torchvision, and pytest. The optional signature CNN can run on CPU; GPU acceleration is not required for the credit-risk dashboard.

## 1.4 Launch LoanShield on Windows

From the project root, with `.venv` activated:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

To stop the application, press `Ctrl+C` in the terminal.

---

# 2. Installation on Ubuntu Linux / WSL 2

The Ubuntu instructions also apply to Ubuntu running inside **Windows Subsystem for Linux (WSL 2)**. For WSL, keep the project in the Linux filesystem (for example `~/LoanShield`) rather than under `/mnt/c` for better file-operation performance.

## 2.1 Install Python and virtual-environment support

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
```

Check your version:

```bash
python3 --version
```

Python 3.11 is recommended for a consistent environment. If your Ubuntu release provides a different Python version, use that version only if the packages in `requirements.txt` have compatible wheels for it.

## 2.2 Open the project folder

For example, after extracting the archive:

```bash
cd ~/LoanShield
```

## 2.3 Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

The prompt should now show `(.venv)`.

## 2.4 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2.5 Launch LoanShield on Ubuntu / WSL

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. In WSL 2, this localhost address is normally accessible from the Windows browser. To stop Streamlit, press `Ctrl+C` in the terminal.

---

# 3. Use the bundled model artifacts or regenerate them

The project archive includes credit-model and clustering artifacts under `models/credit/`, along with the evaluation and segmentation outputs under `reports/` and `data/processed/`. **You can install the requirements and launch the dashboard without retraining**, provided those files remain in their original relative locations.

If the model artifacts are missing, or if you want to regenerate them in the Python environment on your current machine, use the following steps from the project root with the virtual environment activated.

## 3.1 Generate credit model and evaluation artifacts

The source dataset is included at `data/raw/german.data`. The training script makes a stratified 80/20 split using `random_state=42`. The preprocessor is fitted only on the training partition; the 200-row held-out test set is not used to fit the preprocessor or model.

**Windows PowerShell or Ubuntu / WSL:**

```bash
python -m src.models.train_credit_model
```

This creates or replaces:

- `models/credit/random_forest.joblib`
- `models/credit/preprocessor.joblib`
- `models/credit/feature_names.joblib`
- `models/credit/model_metadata.json`
- `reports/results/final_test_metrics.json`
- `reports/results/final_test_predictions.csv`
- `reports/results/model_comparison_cv.csv`
- `data/processed/X_test.csv` and `data/processed/y_test.csv`
- Confusion matrix, ROC, and precision-recall images under `reports/figures/`

**Important:** this command trains a new reproducible model from the included dataset. It does not recover a binary model previously trained on another machine. Back up any existing model artifacts before running it if you need to preserve them. Scikit-learn/joblib model files can be sensitive to Python and package-version differences; when moving artifacts between machines or environments, use compatible dependency versions or retrain on the destination machine if loading fails.

## 3.2 Generate customer segmentation artifacts

Run this after the credit dataset and model environment are available:

```bash
python -m src.clustering.kmeans
```

This generates:

- `models/credit/kmeans.joblib`
- `models/credit/cluster_scaler.joblib`
- `models/credit/cluster_pca.joblib`
- `models/credit/clustering_metadata.json`
- `reports/results/cluster_assignments.csv`
- `reports/results/cluster_profiles.csv`
- `reports/results/kmeans_diagnostics.csv`

The selected `k=3` follows the project's exploratory segmentation design. Clusters are descriptive groupings, not supervised risk labels. Review silhouette and Davies–Bouldin diagnostics before drawing conclusions; observed Bad Credit prevalence in a cluster does not establish causation.

After regenerating artifacts, restart Streamlit so the application reloads the updated files.

---

# 4. Optional signature CNN training

The signature CNN cannot be meaningfully trained without labelled signature images. Acquire a dataset with appropriate usage rights and arrange it as follows:

```text
data/signatures/
├── train/
│   ├── forged/
│   └── genuine/
└── val/
    ├── forged/
    └── genuine/
```

Create the directories using **Windows PowerShell**:

```powershell
New-Item -ItemType Directory -Force data\signatures\train\forged, data\signatures\train\genuine, data\signatures\val\forged, data\signatures\val\genuine
```

Or using **Ubuntu / WSL**:

```bash
mkdir -p data/signatures/train/forged data/signatures/train/genuine \
         data/signatures/val/forged data/signatures/val/genuine
```

Place the corresponding images in each class folder, then run this command on either platform:

```bash
python -m src.models.train_signature_cnn
```

On successful training, the script writes `models/cnn/signature_cnn.pt`. The inference page remains in an informational/untrained state until that file exists.

Do not treat CNN output as verified identity. For signer-independent evaluation, split the dataset by signer rather than randomly splitting individual images. Keep the validation set separate from training data and use a dataset whose licence permits this research use.

---

# 5. Tests and code checks

Run these commands from the project root with the virtual environment activated. They are the same on Windows PowerShell and Ubuntu / WSL:

```bash
python -m pytest -q
python -m compileall -q app src tests
```

The first command runs the automated tests. The second checks Python source files for syntax/compilation errors; it does not replace running the application and exercising each page.

---

# 6. Model interpretation

- Original target label `1` is mapped to `0 = Good Credit`.
- Original target label `2` is mapped to `1 = Bad Credit`.
- The probability displayed by the app is `predict_proba(...)[class 1]`.
- The default classifier threshold is `0.5`; the displayed class is the model's threshold-based output.
- Positive SHAP contributions push this model's output toward the Bad Credit class; negative contributions push it away.
- Global feature importance aggregates contribution magnitude over a sample of records. It is not a causal effect.
- K-Means clusters are unsupervised descriptive groups, not credit-risk classes.

---

# 7. Troubleshooting

## Windows

- **`py` or `python` is not recognized:** reinstall Python and enable the Python Launcher / PATH options, then open a new terminal. You can also use the full Python executable path.
- **PowerShell says scripts are disabled:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window, then activate `.venv` again.
- **`streamlit` is not recognized:** activate `.venv` and launch using `python -m streamlit run app/streamlit_app.py` rather than calling `streamlit` directly.
- **Virtual environment activation fails:** confirm that `.venv\Scripts\Activate.ps1` exists; if using Command Prompt, use `.venv\Scripts\activate.bat`.
- **A package cannot be installed:** confirm you are using 64-bit Python 3.11 and update pip inside the activated environment.

## Ubuntu / WSL

- **`ensurepip` or `venv` is unavailable:** install `python3-venv` with `sudo apt install python3-venv`, then recreate `.venv`.
- **Permission errors during pip install:** activate `.venv` and install packages inside it. Do not use `sudo pip`.
- **WSL cannot access project files:** keep the project under the Linux home directory (for example `~/LoanShield`) and check that the archive was fully extracted.
- **Port 8501 is already in use:** launch on another port using `python -m streamlit run app/streamlit_app.py --server.port 8502`.

## Both platforms

- **Missing `.joblib` files:** restore the bundled `models/credit/` files or run `python -m src.models.train_credit_model` and `python -m src.clustering.kmeans`.
- **`ModuleNotFoundError: src`:** run the command from the LoanShield project root, not from inside `app/` or `src/`.
- **Missing reports or segmentation CSVs:** run the corresponding training and clustering scripts described above.
- **SHAP memory/runtime issue:** reduce the sample-size control on the SHAP Audit page.
- **No signature CNN:** expected until a labelled dataset is supplied and the optional training script is run.
- **PyTorch/torchvision installation issue:** install a mutually compatible PyTorch and torchvision pair for your OS and Python version using the official PyTorch installation selector, then rerun the requirements installation if necessary.

---

# 8. Dataset and responsible-use note

The included `german.data` file is the whitespace-delimited German Credit dataset used by this project. Check the dataset's distribution and usage terms before redistributing the project publicly. The dataset is historical and small; results may not generalize to present-day applicants, other populations, or real lending conditions.

LoanShield is an academic demonstration and must not be used as a production credit-scoring, identity-verification, or lending-decision system.
