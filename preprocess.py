# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Dataset
df = pd.read_csv("bank-full.csv", sep=";")
print("Original Shape:", df.shape)

# Check Missing Values and Duplicate Rows
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicate Rows
df = df.drop_duplicates()

# Handle Unknown Values in Job and Education (replace with most frequent value)
df["job"] = df["job"].replace("unknown", df["job"].mode()[0])
df["education"] = df["education"].replace("unknown", df["education"].mode()[0])

# Unknown Values in Contact and Poutcome (kept as a category)
print("\nUnknown in contact :", (df["contact"] == "unknown").sum())
print("Unknown in poutcome:", (df["poutcome"] == "unknown").sum())

# Encode Binary Columns and Target
df["default"] = df["default"].map({"no": 0, "yes": 1})
df["housing"] = df["housing"].map({"no": 0, "yes": 1})
df["loan"] = df["loan"].map({"no": 0, "yes": 1})
df["y"] = df["y"].map({"no": 0, "yes": 1})

# Encode Education
df["education"] = df["education"].map({"primary": 0, "secondary": 1, "tertiary": 2})

# One-Hot Encode Categorical Columns
df = pd.get_dummies(df, columns=["job", "marital", "contact", "month", "poutcome"],
                    drop_first=True, dtype=int)

# Normalization of Numeric Columns
num_cols = ["age", "education", "balance", "day", "duration", "campaign", "pdays", "previous"]
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Final Check
print("\nProcessed Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("\nFirst 5 Rows:")
print(df.head())

# Save Processed Dataset
df.to_csv("bank-full-processed.csv", sep=";", index=False)
print("\nProcessed file saved: bank-full-processed.csv")