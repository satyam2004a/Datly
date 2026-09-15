import pandas as pd

df = pd.read_csv("data/dataset.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistics:")
print(df.describe())


print("\n--- DATA QUALITY CHECKS ---")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nInvalid Ages:")
print(df[(df["age"] < 0) | (df["age"] > 100)])

print("\nNegative Purchase Amounts:")
print(df[df["purchase_amount"] < 0])

print("\nInvalid Quantities:")
print(df[df["quantity"] <= 0])

print("\nDuplicate Customer IDs:")
print(df[df.duplicated("customer_id", keep=False)])