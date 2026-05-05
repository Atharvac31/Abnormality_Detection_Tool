import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv("synthetic_dataset/labels.csv")

print("\n==============================")
print("DATASET OVERVIEW")
print("==============================")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

# -------------------------------------------------------
# Target column
# -------------------------------------------------------

target = "Abnormality"

X = df.select_dtypes(include=[np.number]).drop(columns=[target])
y = df[target]

# -------------------------------------------------------
# Missing Values Check
# -------------------------------------------------------

print("\n==============================")
print("MISSING VALUES")
print("==============================")

print(df.isnull().sum())

# -------------------------------------------------------
# Duplicate Rows Check
# -------------------------------------------------------

print("\n==============================")
print("DUPLICATE ROWS")
print("==============================")

duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)
print("Duplicate percentage:", duplicates / len(df))

# -------------------------------------------------------
# Class Balance
# -------------------------------------------------------

print("\n==============================")
print("CLASS DISTRIBUTION")
print("==============================")

print(y.value_counts())
print("\nClass Ratio:")
print(y.value_counts(normalize=True))

sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

# -------------------------------------------------------
# Feature Statistics
# -------------------------------------------------------

print("\n==============================")
print("FEATURE STATISTICS")
print("==============================")

print(X.describe())

# -------------------------------------------------------
# Variety Check (Unique Values)
# -------------------------------------------------------

print("\n==============================")
print("FEATURE VARIETY (UNIQUE VALUES)")
print("==============================")

unique_vals = X.nunique().sort_values()

print(unique_vals)

unique_vals.plot(kind="barh", figsize=(10,10))
plt.title("Unique Values per Feature")
plt.show()

# -------------------------------------------------------
# Feature Variance
# -------------------------------------------------------

print("\n==============================")
print("FEATURE VARIANCE")
print("==============================")

variance = X.var().sort_values()

print(variance)

variance.plot(kind="barh", figsize=(10,10))
plt.title("Feature Variance")
plt.show()

# -------------------------------------------------------
# Correlation Heatmap
# -------------------------------------------------------

print("\n==============================")
print("FEATURE CORRELATION")
print("==============================")

plt.figure(figsize=(12,10))
sns.heatmap(X.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

# -------------------------------------------------------
# PCA Visualization (Dataset Variety)
# -------------------------------------------------------

print("\n==============================")
print("PCA VARIETY TEST")
print("==============================")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm")
plt.title("PCA Projection of Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------------------------------------
# Feature Importance Test
# -------------------------------------------------------

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=200)

rf.fit(X_train, y_train)

preds = rf.predict(X_val)

acc = accuracy_score(y_val, preds)

print("Validation Accuracy:", acc)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importances)

importances.plot(kind="bar", figsize=(12,5))
plt.title("Feature Importance")
plt.show()

# -------------------------------------------------------
# Learning Curve Test
# -------------------------------------------------------

print("\n==============================")
print("LEARNING CURVE TEST")
print("==============================")

sizes = [50,100,200,300,400,len(X_train)]

for s in sizes:

    X_subset = X_train.iloc[:s]
    y_subset = y_train.iloc[:s]

    rf.fit(X_subset, y_subset)

    preds = rf.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print("Training Samples:", s, "Validation Accuracy:", acc)

# -------------------------------------------------------
# Dataset Quality Summary
# -------------------------------------------------------

print("\n==============================")
print("DATASET QUALITY SUMMARY")
print("==============================")

print("Average Unique Values:", X.nunique().mean())
print("Average Variance:", X.var().mean())
print("Duplicate %:", df.duplicated().mean())

print("\nDataset quality test completed.")