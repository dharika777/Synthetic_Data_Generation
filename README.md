#  Synthetic Data Generation using DDPM

##  Overview
This project generates synthetic customer churn data using a DDPM model from SynthCity and evaluates its quality against real data.

---

## 📁 Dataset
- File: `botswana_bank_customer_churn.csv`
- Target column: `Churn Flag`
- Removes unnecessary and sensitive columns
- Drops missing values

---

## ⚙️ Installation
Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch synthcity
