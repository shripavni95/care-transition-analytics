import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")

# ----------------------------
# BASIC CLEANING
# ----------------------------

# Remove completely empty rows
df.dropna(how='all', inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df.sort_values('Date', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# ----------------------------
# FIX DATA TYPES
# ----------------------------

# Convert HHS Care column to numeric
df['Children in HHS Care'] = (
    df['Children in HHS Care']
    .astype(str)
    .str.replace(',', '')
    .str.strip()
)

df['Children in HHS Care'] = pd.to_numeric(
    df['Children in HHS Care'],
    errors='coerce'
)

# Fill missing values using forward fill
df = df.ffill()

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------

# Extract year/month/day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Create operational efficiency metric
df['Transfer Efficiency'] = (
    df['Children transferred out of CBP custody'] /
    df['Children apprehended and placed in CBP custody*']
)

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill remaining nulls
df.fillna(0, inplace=True)

# ----------------------------
# SAVE CLEANED DATASET
# ----------------------------

df.to_csv("cleaned_dataset.csv", index=False)

print("\n CLEANING COMPLETED")
print(df.head())

print("\n FINAL DATA INFO")
print(df.info())