This repository contains  implementation of gradient-boosted classification tree ensemble, built entirely from first principles. It follows the deviance-based boosting framework (negative log-likelihood) as outlined in Sections 10.9–10.10 of Hastie, Tibshirani & Friedman’s Elements of Statistical Learning. You’ll find full support for binary and multiclass targets, subsampling, early stopping, custom losses, and comprehensive evaluation routines.

---

## Overview

Our `GradientBoostingClassifier` constructs an additive model of shallow regression trees by repeatedly fitting each tree to the negative gradient (residual) of the log-loss.  At each iteration:

1. **Computing scores** for each class (initialized to log-odds for binary, zero for multiclass)  
2. **Evaluating probabilities** via sigmoid (binary) or softmax (multiclass)  
3. **Forming residuals** = (one-hot or {0,1}) – predicted probabilities  
4. **Fitting** one small decision tree per class on those residuals (with optional subsampling and parallel split search)  
5. **Updating** raw class scores by adding a shrunken tree output  
6. **Logging** training and validation loss; stopping early if it stalls  

In addition, the model offers:

- **`predict_proba`** and `predict`  
- **`evaluate`**: accuracy, precision, recall, F1, confusion matrix  
- **Custom loss** plug-in  
- **Error-handling** for bad inputs and calling methods before fitting  

---

## Project Structure

```
gradientboosting/
│
├── model/
│   └── gradientboosting.py        # Core GBM implementation
│
├── tests/
│   ├── __init__.py
│   └── test_gradientboost.py      # Pytest suite exercising all features
│
├── test_data/                     # Small CSVs for unit tests
│   ├── binary_linear.csv
│   ├── binary_xor.csv
│   ├── multiclass_clusters.csv
│   ├── binary_imbalanced.csv
│   └── multiclass_highdim.csv
│
├── visualizations/
│   └── visualizations.ipynb       # Notebook plotting loss curves & comparisons
│
├── requirements.txt               # pip dependencies
└── README.md                      # This documentation
```

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/YourUser/gradientboosting.git
   cd gradientboosting
   ```

2. **Create & activate a virtual environment**  
   Using **venv**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
   Or **conda**:
   ```bash
   conda create -n gbm python=3.10
   conda activate gbm
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Basic Usage

```python
from model.gradientboosting import GradientBoostingClassifier

# Initializing the model
model = GradientBoostingClassifier(
    n_estimators=100,           # number of boosting rounds
    learning_rate=0.05,         # shrinkage factor
    max_depth=3,                # depth of each tree
    min_samples_split=2,        # min samples per split
    subsample=0.7,              # row subsampling fraction
    early_stopping_rounds=10,   # stop if val loss stalls
    validation_fraction=0.1     # fraction held out for early stopping
)

# Fitting on numpy arrays X_train, y_train
model.fit(X_train, y_train)

# Predicting probabilities and classes
probs = model.predict_proba(X_test)
preds = model.predict(X_test)

# Evaluating performance
metrics = model.evaluate(X_test, y_test)
print(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'])
```

---

## Running the Test Suite

We use **pytest** to verify correctness:

```bash
# From project root
pytest -v -s tests/test_gradientboost.py
```

<details>
<summary>📸 Example test run output</summary>

```
test_gradientboost.py::test_learning_simple_boundary PASSED
test_gradientboost.py::test_probability_bounds PASSED
…  
test_gradientboost.py::test_edge_cases PASSED
```

*(Insert your own screenshot here)*  
</details>

---

## Frequently Asked Questions

**Q: What does this model do and when should I use it?**  
It implements a gradient-boosted decision-tree classifier from scratch—ideal when you need a transparent, customizable GBM (no external C/C++ dependencies), or if you’re learning the internals of boosting.  Use it for both binary and multiclass tasks where log-loss deviance is appropriate.

**Q: How did you test it?**  
We generated five small synthetic datasets (linear separability, XOR, multiclass clusters, imbalanced, and high-dimensional with rare classes) under `test_data/`.  Our pytest suite covers:

- Basic and non-linear decision boundaries  
- Probability validity (0 ≤ p ≤ 1)  
- Multiclass recall and confusion matrices  
- Early stopping behavior  
- Custom loss‐function plumbing  
- Hyperparameter validation and dimension checks  
- Edge cases: single sample, constant target, NaNs, calling predict before fit  

Passing these tests ensures the core algorithms and error-handling are sound.

**Q: What hyperparameters can I tune?**  
- `n_estimators` (number of trees)  
- `learning_rate` (shrinkage)  
- `max_depth`, `min_samples_split` (tree complexity)  
- `subsample` (stochastic boosting)  
- `early_stopping_rounds`, `validation_fraction`  
- Optional `custom_loss` for alternative objectives  

**Q: Any inputs the model struggles with?**  
Small datasets with extremely low sample counts or highly imbalanced rare-class scenarios can be unstable.  In our visualization notebook you’ll see that after ≈10 iterations the validation and training curves diverge—indicating overfitting.  We plan to add:

- **Lower early-stop thresholds** (e.g. 5 rounds)  
- **Adaptive learning-rate schedules** (decay η as training progresses)  

to mitigate this and smooth out the loss plateau.  

---

## Additional Information

Open **`visualizations/visualizations.ipynb`** to:

- Plot training vs. validation loss curves  
- Compare performance on both binary (breast cancer) and multiclass (digits) datasets against `sklearn`’s GBM and `RandomForestClassifier`  
- Observe that on the multiclass digits task:  
  ```
  Random Forest > Custom GBM > Single Decision Tree
  ```
  confirming that our implementation trains reasonably well on real-world data.

Feel free to explore, extend, and tune!
