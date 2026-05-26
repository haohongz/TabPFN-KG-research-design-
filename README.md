# KG-TabICL: Knowledge Graph-Guided Feature Reweighting for Tabular In-Context Learning

---

## What We're Doing

TabPFN and TabICL are state-of-the-art tabular prediction models based on in-context learning. After a forward pass, they produce a representation vector `h` for every cell in the table — but **only the target column's `h` is used for prediction**. All feature-column representations are discarded.

We believe these discarded representations contain useful information, especially in high-dimensional, low-sample settings (d >> n). Our idea: **use Knowledge Graph (KG) embeddings to decide which features matter most, then aggregate their representations back into the prediction**.

This is a post-hoc adapter — we don't modify or retrain the base model.

---

## Project Roadmap

```
Phase 1: Validate on TabICL (nanotabicl)     ← WE ARE HERE
Phase 2: Full benchmark + ablation studies
Phase 3: Transfer to TabPFN
Phase 4: Paper writing (targeting ICML/NeurIPS workshop)
```

---

## Phase 1: TabICL Experiments

### 1.1 Method Overview (Method 3: Feature Weighted Aggregation)

```
Input:
  - TabICL's output representations: h_{i,j} for all cells, h_{i,y} for target column
  - KG embeddings: e_ext_j for each feature j

Step 1: Compute feature prototype
  h_bar_j = mean of h_{i,j} over training samples

Step 2: Compute KG-guided importance weights
  alpha_j = softmax( e_ext_j^T @ W_alpha @ h_bar_j )
  → global weights, shared across all test samples

Step 3: Weighted aggregation
  h_pooled = sum_j( alpha_j * h_{i,j} )

Step 4: Fusion with original target representation
  h_final = beta * h_{i,y} + (1 - beta) * h_pooled
  → beta initialized to 0.9 (biased toward original model)
  → degradation guarantee: beta → 1 recovers original TabICL

Step 5: Prediction
  y_hat = classifier(h_final)

Trainable parameters: W_alpha (d_ext × d) + beta (scalar) + classifier
All TabICL parameters are FROZEN.
```

### 1.2 KG Embedding Pipeline

```
For each feature column j:

  Text encoding:
    column_name_j → Sentence-BERT → e_text_j (384-dim)

  KG structure encoding (when available):
    KG_node_j → TransE (via PyKEEN) → e_kg_j (128-dim)

  Concatenation:
    e_ext_j = [e_text_j ; e_kg_j]  (512-dim)

  If feature not found in KG:
    e_ext_j = [e_text_j ; zeros]   (384 + 128 = 512-dim)

All embeddings are precomputed offline. Zero inference overhead.
```

### 1.3 Codebase

```
KG-TabICL/
├── model.py                  # nanotabicl source (from github.com/soda-inria/nanotabicl)
├── kg_adapter.py             # KGFeatureWeightedAdapter module (our code)
├── kg_embeddings.py          # Text + KG encoding pipeline
├── extract_representations.py # Hook TabICL to extract h vectors
├── run_experiment.py         # Full experiment: data → extract h → train adapter → evaluate
├── data/
│   ├── breast_cancer/        # sklearn toy dataset (for debugging)
│   ├── tcga_brca/            # TCGA breast cancer gene expression
│   └── lincs_l1000/          # LINCS drug response
├── kg/
│   ├── string_v12/           # STRING protein interaction network
│   └── hetionet/             # Hetionet medical KG
├── results/
│   ├── tables/               # CSV result tables
│   └── figures/              # Plots
└── README.md                 # This file
```

### 1.4 Experiment Design

**Datasets**

| Dataset | Domain | Features (m) | Samples (n) | Task | KG Source |
|---------|--------|-------------|-------------|------|-----------|
| Breast Cancer (sklearn) | Medical | 30 | 569 | Classification | Text only |
| TCGA-BRCA | Genomics | ~5000 | ~300 | Classification | STRING v12 |
| LINCS L1000 | Pharmacology | 978 | ~200 | Classification | STRING + DrugBank |
| GDSC | Pharmacogenomics | ~1000 | ~200 | Regression | STRING + KEGG |

**Methods to Compare**

| Method | Description | Purpose |
|--------|-------------|---------|
| TabICL (original) | No KG, standard prediction | **Core comparison** |
| TabICL + KG (ours) | Method 3 adapter | Our method |
| XGBoost | Tree-based baseline | Sanity check |
| Lasso / Ridge | Linear baseline | Sanity check |

**Ablation Studies**

| Config | What it tests |
|--------|---------------|
| Full (text + KG) | Complete method |
| Text only | Is KG structure needed? |
| KG only | Is text semantics needed? |
| Random embedding | Is it just extra parameters? |
| Permuted KG | Is the KG structure important? |
| Uniform weights (1/m) | Is KG weighting needed, or just using feature h? |
| No adapter (beta=1) | Equivalent to original TabICL |

**Evaluation Protocol**

- 5-fold stratified cross-validation
- Metrics: AUROC (primary), Accuracy, F1 for classification; Pearson r, R² for regression
- Report mean ± std across folds
- Paired t-test for significance (p < 0.05)

### 1.5 Implementation Steps

```
Week 1:
  [x] Write KGFeatureWeightedAdapter module
  [x] Write experiment runner with 5-fold CV
  [ ] Download nanotabicl model.py + pretrained checkpoint
  [ ] Run on breast cancer dataset (debugging)
  [ ] Verify: baseline accuracy matches original TabICL

Week 2:
  [ ] Set up Sentence-BERT encoding pipeline
  [ ] Download STRING v12 + set up TransE via PyKEEN
  [ ] Prepare TCGA-BRCA dataset + KG matching
  [ ] Run Method 3 on TCGA-BRCA
  [ ] Run ablation experiments

Week 3:
  [ ] Run on LINCS and GDSC datasets
  [ ] Generate result tables + figures
  [ ] Analyze alpha weights (interpretability)
  [ ] Write up Phase 1 results
```

---

## Phase 2: Full Benchmark

After Phase 1 shows positive results:

```
[ ] Add more datasets from PLATO benchmark
[ ] Compare against PLATO directly
[ ] Hyperparameter sensitivity analysis
[ ] KG coverage vs. performance gain analysis
[ ] Learned beta analysis per dataset
```

---

## Phase 3: Transfer to TabPFN

### Why It Transfers

The KG adapter module is **model-agnostic**. It only needs:
1. Feature-column representations `h_{i,j}` (any dimension)
2. Target-column representation `h_{i,y}` (same dimension)
3. KG embeddings `e_ext` (fixed)

Both TabICL and TabPFN produce these. The adapter doesn't care who generated them.

### What Changes

```
TabICL (Phase 1):
  - d_pfn = 128 (or 256)
  - h extraction: modify forward() directly (open source)
  - KG adapter: KGFeatureWeightedAdapter(d_pfn=128, d_ext=512)

TabPFN (Phase 3):
  - d_pfn = 512
  - h extraction: use PyTorch hooks (closed source model)
  - KG adapter: KGFeatureWeightedAdapter(d_pfn=512, d_ext=512)
                                          ^^^ only this number changes
```

### Transfer Steps

```
[ ] Use Aurora's hook code to extract h from TabPFN
[ ] Change d_pfn from 128 to 512 in KGFeatureWeightedAdapter
[ ] Retrain W_alpha and beta on same datasets
[ ] Compare TabICL+KG vs TabPFN+KG
[ ] Final result table with both base models
```

### Expected Result Table (for paper)

```
                    TCGA-BRCA    LINCS    GDSC
TabICL               0.72       0.81     0.75
TabICL + KG (ours)   0.78       0.85     0.80
TabPFN               0.80       0.86     0.82
TabPFN + KG (ours)   0.83       0.88     0.84
XGBoost              0.65       0.72     0.68
PLATO                0.76       0.83     0.78
```

Both base models improve → method is **model-agnostic** (strong selling point for reviewers).

---

## Key Design Decisions

### Why Method 3 first (not Method 1 or 2)?

- **Fewest parameters**: only W_alpha + beta + classifier
- **Least risk of overfitting** in d >> n regime
- **Fastest to implement and debug**
- If Method 3 works → validates the core idea (KG-guided feature reweighting)
- If Method 3 doesn't work → Method 1/2 probably won't either, saves time

### Why post-hoc (not modifying the model)?

- **Degradation guarantee**: beta → 1 recovers original model, worst case = no harm
- **No retraining needed**: base model weights are frozen
- **Model-agnostic**: same adapter works on TabICL and TabPFN

### Why global weights (not per-sample)?

- d >> n means we have very few samples to estimate per-sample weights
- Global weights have fewer parameters → less overfitting
- Method 1 uses per-sample weights → can compare later

---

## Dependencies

```
# Core
torch >= 2.0
numpy
scikit-learn
pandas

# KG embeddings
sentence-transformers   # for text encoding
pykeen                  # for TransE / KG structure encoding

# nanotabicl
# clone from: https://github.com/soda-inria/nanotabicl
```

---

## References

- TabPFN: Hollmann et al., "Accurate predictions on small data with a tabular foundation model", Nature 2025
- TabICL: Qu et al., "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data", ICML 2025
- PLATO: Ruiz et al., "High dimensional, tabular deep learning with an auxiliary knowledge graph", NeurIPS 2023
- ConTextTab: Spinaci et al., "ConTextTab: A Semantics-Aware Tabular In-Context Learner", NeurIPS 2025
- TransE: Bordes et al., "Translating embeddings for modeling multi-relational data", NeurIPS 2013
- Sentence-BERT: Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
