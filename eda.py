# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\rarju\OneDrive\Desktop\bank\bank-full.csv", sep=";")

# Dataset Information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types and Non-Null Counts:")
df.info()

# Missing Values and Duplicate Rows
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Unknown Values
print("\nUnknown Values in Each Column:")
for col in df.select_dtypes(include="object").columns:
    count = (df[col] == "unknown").sum()
    if count > 0:
        print(col, ":", count)

# Unique Values
print("\nUnique Values in Each Column:")
for col in df.columns:
    print(col, ":", df[col].nunique())

# Statistical Summary
print("\nNumerical Columns Summary:")
print(df.describe())
print("\nCategorical Columns Summary:")
print(df.describe(include="object"))

# Target Variable Analysis
print("\nTarget Distribution:")
print(df["y"].value_counts())
print("\nTarget Distribution in Percentage:")
print((df["y"].value_counts(normalize=True) * 100).round(2))

# Numerical Values Check
print("\nNumerical Values Check:")
print("Negative balance records:", (df["balance"] < 0).sum())
print("Pdays equal to -1:", (df["pdays"] == -1).sum())
print("Duration equal to 0:", (df["duration"] == 0).sum())

# Outlier Detection
print("\nPotential Outlier Summary:")
for col in ["age", "balance", "duration", "campaign", "previous"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {count} potential outliers")

# Target Distribution Graph
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="y")
plt.title("Target Distribution")
plt.xlabel("Term Deposit Subscription")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Age Distribution Graph
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# Balance Boxplot
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="balance")
plt.title("Balance Distribution and Potential Outliers")
plt.tight_layout()
plt.show()

# Job-wise Subscription Graph
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="job", hue="y")
plt.title("Subscription by Job")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Education-wise Subscription Graph
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="education", hue="y")
plt.title("Subscription by Education")
plt.tight_layout()
plt.show()

# Housing Loan Subscription Graph
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="housing", hue="y")
plt.title("Subscription by Housing Loan")
plt.tight_layout()
plt.show()