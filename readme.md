# Diabetic Retinopathy Grading — Foundation of AI Coursework

Hybrid CNN + Knowledge Base system for automated Diabetic Retinopathy screening.  
Three approaches are implemented and compared:

| Notebook | Method |
|---|---|
| `01_EDA_Dataset02.ipynb` | Exploratory Data Analysis |
| `02_Method1_ClassImbalance_ResNet50.ipynb` | Single 5-class ResNet-50 with class-imbalance handling |
| `03_Method2_Binary_Hierarchical_ResNet50.ipynb` | Two-stage hierarchical ResNet-50 (No-DR gate → severity) |
| `04_Hybrid_KnowledgeBase_Evaluation.ipynb` | Full evaluation on the held-out test set |
| `05_Inference_Hybrid_KnowledgeBase.ipynb` | **Inference demo on sample data** ← start here |


---

## Project Structure

```
src/
├── notebooks (*.ipynb)
├── models/                          # trained model weights (.pth)
│   ├── best_method1_multiclass_resnet50.pth
│   ├── best_method2_binary_resnet50.pth
│   └── best_method2_severity_resnet50.pth
├── images/                          # saved output plots (.png)
├── sample_data/                     # ← safe to commit to git
│   ├── images/                      # 50 synthetic fundus images
│   ├── sample_test.csv              # matching clinical metadata
│   └── inference_results.csv        # written after running notebook 05
└── runs/                            # TensorBoard logs
```

---

## Quick Start — Inference Demo

> **No dataset required.** The inference notebook uses the 50 synthetic images
> already included in `src/sample_data/`.  
> You only need the trained model weights in `src/models/`.

### 1. Install dependencies

```bash
pip install -r src/requirements.txt
```

### 2. Obtain trained model weights

The `.pth` weight files are **not stored in this repository** (each is ~94 MB — too large for GitHub).
You have two options:

**Option A — Train from scratch** *(recommended)*  
Run notebooks **02** and **03** in order.  
Weights are automatically saved to `src/models/` when training completes.

**Option B — Copy pre-trained weights manually**  
Place the following files in `src/models/` before running the inference notebook:
```
src/models/best_method1_multiclass_resnet50.pth
src/models/best_method2_binary_resnet50.pth
src/models/best_method2_severity_resnet50.pth
```

### 3. Open the inference notebook

```bash
cd src
jupyter notebook 05_Inference_Hybrid_KnowledgeBase_Evaluation.ipynb
```

Then **Run All Cells** (`Kernel → Restart & Run All`).

The notebook will:
1. Load all three trained ResNet-50 models.
2. Run inference on the 46 balanced real test images (Method 1 — multi-class, Method 2 — hierarchical).
3. Feed each CNN prediction + the patient's clinical metadata into the **Triage Engine**.
4. Display a sample image grid, confidence heatmap, triage action bar chart, and per-patient report.
5. Save the full results table to `sample_data/inference_results.csv`.

> **Regenerate sample data at any time:**
> ```bash
> cd src
> python generate_sample_data.py
> ```

---

### About the Sample Data

### Location
```
src/sample_data/
├── images/          # 46 real fundus JPEG images (256 × 256 px)
└── sample_test.csv  # clinical metadata for each image
```

### Class distribution (class-balanced)

| Grade | Label | Count |
|---|---|---|
| 0 | No DR | 10 |
| 1 | Mild NPDR | 10 |
| 2 | Moderate NPDR | 10 |
| 3 | Severe NPDR | 6 (all available in test set) |
| 4 | PDR | 10 |
| **Total** | | **46** |

Images are sampled from the held-out test split of the research dataset using
`src/generate_sample_data.py` (reproducible with `random_state=42`).

### CSV columns

| Column | Description |
|---|---|
| `image_id` | Image filename (with `.jpg` extension) |
| `DR_ICDR` | Ground-truth DR grade (0–4) |
| `split` | Always `test` |
| `patient_age` | Patient age (years) |
| `diabetes_time_y` | Diabetes duration (years) |
| `insuline` | Insulin use (`Yes` / `No`) |
| `comorbidities` | Free-text comorbidity list |
| `hypertensive_retinopathy` | Binary flag (0 / 1) |
| `quality` | Image quality (`Adequate` / `Inadequate`) |
| `macular_edema` | Binary flag (0 / 1) |


## Running the Streamlit Inference App

The quickest way to test the system is the interactive web UI:

```bash
cd src
streamlit run streamlit_app/app.py
```

The app opens in your browser at `http://localhost:8501`.

**What you can do:**
- **Upload** your own fundus image or **pick** from the 46 balanced sample images
- Fill in the patient's clinical data (age, diabetes duration, comorbidities, etc.)
- Click **Run Inference** to get:
  - Method 1 (multi-class) grade + confidence bar chart
  - Method 2 (hierarchical) grade
  - Colour-coded triage action with clinical rationale

See `src/streamlit_app/sample_inputs.txt` for 7 ready-to-use patient profiles that
demonstrate every possible triage outcome.

---

## Running the Full Pipeline

```
01_EDA              → explore the dataset
02_Method1          → train multi-class ResNet-50   → saves best_method1_*.pth
03_Method2          → train hierarchical ResNet-50  → saves best_method2_*.pth
04_Hybrid           → evaluate on real held-out test set
05_Inference        → quick demo on sample_data/
05_Results_Comparison → compare all methods side-by-side
```

---

## Requirements

Key packages (see `src/requirements.txt` for pinned versions):

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA recommended)
- torchvision
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- Pillow
- tqdm
- streamlit
