import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Load Dataset
FILE_PATH = "botswana_bank_customer_churn.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print("✅ Dataset loaded successfully!")
    print("Shape:", df.shape)
except Exception as e:
    print(f"❌ ERROR: Could not load dataset at {FILE_PATH}. Details:\n{e}")
    exit()

# 2️⃣ Clean and Prepare Data
drop_cols = [
    "RowNumber", "CustomerId", "Surname", "First Name", "Date of Birth",
    "Address", "Contact Information", "Churn Reason", "Churn Date"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

TARGET_COLUMN = "Churn Flag"
if TARGET_COLUMN not in df.columns:
    print(f"❌ ERROR: '{TARGET_COLUMN}' column not found in dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

df = df.dropna(axis=0).reset_index(drop=True)

numerical_features = df.select_dtypes(include=np.number).columns.drop(TARGET_COLUMN)
categorical_features = df.select_dtypes(include=["object"]).columns

print(f"Numerical features: {list(numerical_features)}")
print(f"Categorical features: {list(categorical_features)}")

print("\n🚀 Training SynthCity DDPM Model...")

# Prepare data for SynthCity
loader = GenericDataLoader(df, target_column=TARGET_COLUMN)

# Initialize DDPM model (simplified args for new SynthCity)
import torch

# use PyTorch device object
device = torch.device("cpu")

model = Plugins().get(
    "ddpm",
    device=device,
    n_iter=200
)

# Train the DDPM
model.fit(loader)
print("✅ DDPM training complete!")

# -----------------------------
# 5️⃣ Generate Synthetic Data
# -----------------------------
synthetic_df = model.generate(count=len(df)).dataframe()
print("🎉 Synthetic data generated!")
print("Synthetic shape:", synthetic_df.shape)
# Save the generated synthetic data to a CSV file
synthetic_df.to_csv("synthetic_data_generated.csv", index=False)
print("💾 Synthetic data saved as 'synthetic_data_generated.csv' in the current directory.")


# Ensure valid integer columns
for col in ['NumOfProducts', 'NumComplaints', TARGET_COLUMN]:
    if col in synthetic_df.columns:
        synthetic_df[col] = synthetic_df[col].clip(lower=0).round().astype(int)

# -----------------------------
# 6️⃣ Compare Correlation Structures
# -----------------------------
print("\n📊 Comparing correlations...")
common_numeric = [col for col in numerical_features if col in synthetic_df.columns]
original_corr = df[common_numeric + [TARGET_COLUMN]].corr().fillna(0)
synthetic_corr = synthetic_df[common_numeric + [TARGET_COLUMN]].corr().fillna(0)
corr_diff = np.linalg.norm(original_corr.values - synthetic_corr.values, "fro")
print(f"Frobenius Norm (Correlation Diff): {corr_diff:.4f}")

# -----------------------------
# 7️⃣ Predictive Utility Test
# -----------------------------
print("\n📈 Training classifier on synthetic vs real data...")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

X_real = df.drop(columns=[TARGET_COLUMN])
y_real = df[TARGET_COLUMN]
X_synth = synthetic_df.drop(columns=[TARGET_COLUMN])
y_synth = synthetic_df[TARGET_COLUMN]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
)

X_train_r_proc = preprocessor.fit_transform(X_train_r)
X_test_r_proc = preprocessor.transform(X_test_r)

clf_real = RandomForestClassifier(random_state=42, n_jobs=-1)
clf_real.fit(X_train_r_proc, y_train_r)
auc_real = roc_auc_score(y_test_r, clf_real.predict_proba(X_test_r_proc)[:, 1])
print(f"Real Data AUC: {auc_real:.4f}")

X_synth_proc = preprocessor.transform(X_synth)
clf_synth = RandomForestClassifier(random_state=42, n_jobs=-1)
clf_synth.fit(X_synth_proc, y_synth)
auc_synth = roc_auc_score(y_test_r, clf_synth.predict_proba(X_test_r_proc)[:, 1])
print(f"Synthetic Data AUC: {auc_synth:.4f}")

# -----------------------------
# 8️⃣ Visualization
# -----------------------------
def plot_distribution_comparison(original, synthetic, cols, is_numeric=True):
    for col in cols:
        if col not in synthetic.columns:
            continue
        plt.figure(figsize=(6, 3))
        if is_numeric:
            sns.kdeplot(original[col], color="blue", fill=True, label="Original")
            sns.kdeplot(synthetic[col], color="red", fill=True, label="Synthetic")
            plt.title(f"{col} Distribution")
        else:
            orig_counts = original[col].value_counts(normalize=True)
            synth_counts = synthetic[col].value_counts(normalize=True)
            pd.concat([orig_counts, synth_counts], axis=1, keys=["Original", "Synthetic"]).plot(kind="bar")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("\n🖼️ Plotting numerical distributions...")
plot_distribution_comparison(df, synthetic_df, numerical_features.tolist(), is_numeric=True)

if len(categorical_features) > 0:
    print("🖼️ Plotting categorical distributions...")
    plot_distribution_comparison(df, synthetic_df, categorical_features.tolist(), is_numeric=False)

print("✅ Done.")
