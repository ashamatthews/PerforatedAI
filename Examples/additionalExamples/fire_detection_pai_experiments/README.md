# Wildfire Prediction from Satellite Data (California 2020)

This example demonstrates wildfire prediction using satellite-derived vegetation indices and land surface temperature, comparing:

- Baseline Neural Network (PyTorch)
- Neural Network with PerforatedAI dendrite restructuring

The goal is to evaluate whether dynamic dendrite growth improves classification performance under extreme class imbalance.

---

## Dataset

The dataset integrates three NASA Earthdata sources:

- FIRMS (VIIRS 375m) – Active fire detections  
- MODIS MOD13Q1 – NDVI & EVI vegetation indices (16-day composite)  
- MODIS MOD11A2 – Land Surface Temperature (8-day composite, averaged to 16-day windows)  

Final processed dataset:

- ~18M valid 1 km² pixels  
- ~0.08% fire pixels (extremely imbalanced)  
- Train / Val / Test split: 70 / 15 / 15  

For neural network experiments, a balanced subset is sampled for stability and speed.

---

## Model Architecture

```
Input (4 features)
->  Linear(4 -> 64)
->  ReLU
->  Linear(64 -> 64)
->  ReLU
->  Linear(64 -> 1)
->  Sigmoid
```

The same architecture is used for:

- Baseline model
- Dendrite-enhanced model (PerforatedAI)

---

## Experiment

The script automatically runs two versions:

- `is_dendrite = False` ->  Standard MLP  
- `is_dendrite = True` ->  MLP with dynamic dendrite growth  

### Example Results

| Model | Test Accuracy | Precision | Recall | F1 |
|-------|--------------|----------|--------|-----|
| Baseline | 59.52% | 0.5811 | 0.6822 | 0.6276 |
| + Dendrites | 64.09% | 0.6187 | 0.7341 | 0.6715 |

Dendrites improved overall accuracy and F1 while increasing model capacity dynamically.

---

## Class Imbalance

Fire pixels represent <0.01% of total data.

Because of this:

- Accuracy alone is misleading  
- Recall is critical (missing fires is costly)  
- Balanced sampling is used for neural training  

---

## How to Run

### Install Dependencies

```bash
pip install torch numpy scikit-learn perforatedai
```

### Dataset Location

Expected path:

```
fire_detection_pai_experiments/data/processed/modis_firms_train_val_test_dataset.npz
```

The file must contain:

```
X_train, y_train
X_val, y_val
X_test, y_test
```

### Launch Jupyter

From the project root directory:

```
jupyter notebook
```

Then open:

```
notebooks/fire_dendrite_vs_baseline.ipynb
```

Run all cells.

---

## Key Observations

- Dendrite restructuring increases model capacity dynamically.
- Validation and test performance improved compared to baseline.
- Dynamic architecture growth can help under imbalanced conditions.

---

## Future Work

- Add meteorological features (wind, humidity, drought index)
- Explore spatio-temporal models (CNN / RNN / Transformers)
- Improve precision-recall tradeoff via threshold tuning
- Evaluate full imbalanced dataset without balancing

---

## Folder Structure

```
fire_detection_pai_experiments/
│
├── README.md
├── data/
│   └── processed/
│       └── modis_firms_train_val_test_dataset.npz
│
└── notebooks/
    └── fire_dendrite_vs_baseline.ipynb
```

---

