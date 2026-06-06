import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================================
# CONFIGURATION
# ======================================
INPUT_FILE = "heart.csv"
OUTPUT_FILE = "heart_cleaned.csv"
FIGURES_DIR = "eda_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ======================================
# LOAD DATASET
# ======================================
df = pd.read_csv(INPUT_FILE)

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

# FIX: df.info() returns None — capture output properly
print("\nDATA TYPES")
print(df.dtypes)

print("\nSUMMARY STATISTICS")
print(df.describe())

# ======================================
# MISSING VALUES
# ======================================
print("\nMISSING VALUES")
missing = df.isnull().sum()
print(missing)
missing_pct = (missing / len(df) * 100).round(2)
print("\nMissing %")
print(missing_pct[missing_pct > 0] if missing_pct.any() else "No missing values found.")

# ======================================
# DUPLICATE ROWS
# ======================================
print("\nDUPLICATE ROWS")
dup_count = df.duplicated().sum()
print(f"Duplicates found: {dup_count}")

# ======================================
# TARGET DISTRIBUTION
# ======================================
print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True).round(3))

fig, ax = plt.subplots(figsize=(6, 4))
df["target"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
ax.set_title("Target Distribution")
ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
ax.set_ylabel("Count")
ax.set_xticklabels(["0 - No Disease", "1 - Disease"], rotation=0)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/target_distribution.png")
plt.close()
print(f"Saved: {FIGURES_DIR}/target_distribution.png")

# ======================================
# FEATURE DISTRIBUTIONS
# ======================================
numerical_cols = df.select_dtypes(include="number").columns.tolist()
numerical_cols = [c for c in numerical_cols if c != "target"]

n_cols = 4
n_rows = int(np.ceil(len(numerical_cols) / n_cols))
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col], bins=20, color="steelblue", edgecolor="white")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Count")

# Hide any unused subplots
for j in range(len(numerical_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/feature_distributions.png", bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR}/feature_distributions.png")

# ======================================
# CORRELATION HEATMAP
# ======================================
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = corr.isnull()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
    mask=mask
)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png")
plt.close()
print(f"Saved: {FIGURES_DIR}/correlation_heatmap.png")

# ======================================
# BOXPLOTS: KEY FEATURES BY TARGET
# ======================================
key_features = ["age", "thalach", "chol", "trestbps", "oldpeak"]
fig, axes = plt.subplots(1, len(key_features), figsize=(18, 5))

for ax, col in zip(axes, key_features):
    df.boxplot(column=col, by="target", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("Target")

plt.suptitle("Key Features by Target", fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/boxplots_by_target.png")
plt.close()
print(f"Saved: {FIGURES_DIR}/boxplots_by_target.png")

# ======================================
# OUTLIER DETECTION (IQR Method)
# ======================================
print("\nOUTLIER SUMMARY (IQR Method)")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        print(f"  {col}: {len(outliers)} outliers (range: {lower:.2f} – {upper:.2f})")

# ======================================
# REMOVE DUPLICATES & SAVE
# ======================================
print(f"\nBefore removing duplicates: {df.shape}")
df = df.drop_duplicates()
print(f"After removing duplicates:  {df.shape}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned dataset saved as '{OUTPUT_FILE}'")
print("EDA complete. Figures saved to:", FIGURES_DIR)