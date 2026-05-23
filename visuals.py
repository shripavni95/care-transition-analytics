import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Style
sns.set_style("darkgrid")

# -----------------------------
# 1. Apprehended Children Trend
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(
    df['Date'],
    df['Children apprehended and placed in CBP custody*'],
)
plt.title("Children Apprehended Over Time")
plt.xlabel("Date")
plt.ylabel("Children Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 2. HHS Care Trend
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(
    df['Date'],
    df['Children in HHS Care'],
)
plt.title("Children in HHS Care Over Time")
plt.xlabel("Date")
plt.ylabel("Children Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Transfer Efficiency
# -----------------------------
plt.figure(figsize=(12,6))
sns.histplot(df['Transfer Efficiency'], bins=30)
plt.title("Transfer Efficiency Distribution")
plt.xlabel("Efficiency")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Monthly Trend
# -----------------------------
monthly = df.groupby('Month')[
    'Children in HHS Care'
].mean()

plt.figure(figsize=(10,5))
monthly.plot(kind='bar')

plt.title("Average Monthly HHS Care")
plt.xlabel("Month")
plt.ylabel("Average Children Count")
plt.tight_layout()
plt.show()