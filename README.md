# 📊 Synthetic Data Generation (DDPM)

## 🚀 Overview
Generates synthetic customer churn data using a DDPM model (SynthCity) and evaluates its quality against real data.

---

## 📁 Dataset
- `botswana_bank_customer_churn.csv`
- Target: `Churn Flag`
- Drops irrelevant columns and missing values

---

## ⚙️ Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch synthcity
```

---

## ▶️ Usage
```bash
python your_script_name.py
```

---

## 🧠 Features
- Train DDPM model
- Generate synthetic dataset
- Save output: `synthetic_data_generated.csv`

---

## 📊 Evaluation
- Correlation similarity (Frobenius norm)
- AUC score (Random Forest)
- Distribution plots (real vs synthetic)

---

## ⚠️ Notes
- CPU training may be slow
- Increase iterations for better results

---

## 📜 License
For academic and research use only.
