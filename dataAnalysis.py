import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_parquet("UNSW_NB15/UNSW_NB15_training-set.parquet")


# Print some features of the dataset
print(df.shape)
print(df.head().to_string())

print(df.info())
print(df["label"].value_counts())  # 0 for normal, 1 for attack
print(df["attack_cat"].value_counts())  # Count of each attack category. Includes Normal as a category.
print(df["service"].value_counts())

print(df.describe())

constantColumns = [df.columns[df.nunique() <= 1]]
print(constantColumns)  # There are no constant columns

# Correlation
sampledf = df.sample(n=1000, random_state=42)

plt.figure(figsize=(10, 6))
sns.heatmap(sampledf.corr(numeric_only=True), cmap='viridis')
plt.title("UNSW-NB15 correlation heatmap")
plt.show()



