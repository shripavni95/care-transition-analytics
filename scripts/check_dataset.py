import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

print("\n FIRST 5 ROWS")
print(df.head())

print("\n COLUMN NAMES")
print(df.columns.tolist())

print("\n DATASET INFO")
print(df.info())