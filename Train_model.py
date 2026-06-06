import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("heart_cleaned.csv")

X = df.drop("target", axis=1)
y = df["target"]

print("=" * 50)
print("DATASET INFO")
print("=" * 50)
print(f"Total samples : {len(df)}")
print(f"Features      : {X.shape[1]}")
print(f"Target split  : {y.value_counts().to_dict()}")

# ======================================
# TRAIN / TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,       # preserve class balance in both splits
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ======================================
# BUILD PIPELINE
# FIX: wrapping scaler + model in a Pipeline prevents data leakage
# during cross-validation. Without this, the old code called
# scaler.fit_transform(X) on the full dataset before CV, letting
# test-fold statistics leak into training.
# ======================================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ))
])

# ======================================
# TRAIN
# ======================================
pipeline.fit(X_train, y_train)

# ======================================
# EVALUATE ON TEST SET
# ======================================
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report")
print(classification_report(y_test, predictions, target_names=["No Disease", "Disease"]))

# ======================================
# CONFUSION MATRIX
# ======================================
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved: confusion_matrix.png")

# ======================================
# CROSS-VALIDATION (uses Pipeline — no leakage)
# ======================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("\nCross-Validation Scores (5-fold)")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean CV Accuracy : {cv_scores.mean():.4f}")
print(f"Std CV Accuracy  : {cv_scores.std():.4f}")

# ======================================
# FEATURE IMPORTANCE
# ======================================
rf_model = pipeline.named_steps["model"]
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nFeature Importances")
print(feat_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("Saved: feature_importances.png")

# ======================================
# SAVE MODEL & SCALER
# FIX: save the full pipeline so the scaler is always applied
# consistently at inference time — no need to save separately.
# ======================================
joblib.dump(pipeline, "model.pkl")
print("\nPipeline (scaler + model) saved as 'model.pkl'")
print("Training complete.")