<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=28&width=680&color=00E5FF&center=true&vCenter=true&lines=Logistic+Regression+in+Agriculture;Interpretable+Classification+for+Agri+Problems;PhD+Research+%40+ICAR-IARI+(IASRI)" />
</p>

---

## 🌾 Repository: Logistic Regression for Agricultural Applications

This repository demonstrates practical uses of **Logistic Regression** in **agricultural statistics** — solving binary/multiclass classification problems common in agronomy, extension, plant protection, and policy.

Perfect for researchers, students, and practitioners who want interpretable models over black-box ML.

---

## ✨ Key Features

- Agricultural datasets & use cases (adoption, disease risk, yield classes)  
- Statistical focus: odds ratios, confidence intervals, model diagnostics  
- Handling real-world issues: class imbalance, multicollinearity, categorical predictors  
- Code in Python using **statsmodels** (detailed inference) & **scikit-learn** (pipelines & prediction)  
- Jupyter notebooks with step-by-step explanations  

---

## 📂 Notebooks & Projects Included

| Notebook | Description | Key Techniques |
|--------|-------------|----------------|
| `01_binary_logistic_adoption.ipynb` | Predict farmer adoption of improved seeds/practices (Yes/No) | Odds ratios, Wald tests, marginal effects |
| `02_pest_outbreak_risk.ipynb` | Early warning: Will pest outbreak occur? (imbalanced data) | Class weights, SMOTE, PR-AUC |
| `03_multinomial_yield_class.ipynb` | Classify yield as low/medium/high based on climate & soil | Multinomial logit, softmax interpretation |
| `04_model_diagnostics_agri.ipynb` | Validate logistic models (Hosmer-Lemeshow, ROC, calibration plots) | Goodness-of-fit, multicollinearity (VIF) |
| `05_from_scratch_gradient_descent.ipynb` | Logistic regression implemented from scratch | Sigmoid, log-loss, SGD optimizer |

*(Add your own datasets — anonymized field trial data works great!)*

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,statsmodels,pandas,numpy,matplotlib,seaborn,jupyter,git,github,vscode,latex" />
</p>

Core libraries:  
- `statsmodels` → statistical summaries & inference  
- `scikit-learn` → modeling pipelines & metrics  
- `imbalanced-learn` → handling rare events (e.g., outbreaks)

---

## 🚀 Quick Start Example (Farmer Adoption)

```python
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# Load example agri data
df = pd.read_csv("data/farmer_adoption.csv")  # columns: age, farm_size_ha, extension_visits, risk_score, adopted (0/1)

X = df[['age', 'farm_size_ha', 'extension_visits', 'risk_score']]
X = sm.add_constant(X)
y = df['adopted']

# Fit logistic model
model = sm.Logit(y, X).fit(disp=0)
print(model.summary())  # odds ratios, p-values, pseudo-R²

# Predict probabilities
probs = model.predict(X)
print("Sample ROC-AUC:", roc_auc_score(y, probs))
