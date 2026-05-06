import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np


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



# 1. Codificar las categorías para asegurar el orden
le = LabelEncoder()
df['attack_cat_encoded'] = le.fit_transform(df['attack_cat'])

# Obtener el mapeo para el paper (ej: 0: Analysis, 1: Backdoor...)
class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("Mapeo de clases:", class_mapping)

# 2. Calcular frecuencias
counts = df['attack_cat_encoded'].value_counts().sort_index().values
total_samples = len(df)
n_classes = len(le.classes_)

# 3. Calcular Alpha (Frecuencia Inversa Balanceada)
# Fórmula: total / (n_clases * counts)
alpha = total_samples / (n_classes * counts)

# 4. Normalización (Opcional pero recomendada)
# Hace que el peso promedio sea 1, lo que estabiliza el entrenamiento
alpha_norm = alpha / alpha.mean()

# 5. Convertir a tensor para PyTorch
alpha_tensor = torch.tensor(alpha, dtype=torch.float32)

print("\nPesos Alpha calculados:")
for cls, weight in zip(le.classes_, alpha_norm):
    print(f"{cls:15s}: {weight:.4f}")

print("\nTensor listo para Focal Loss:")
print(alpha_tensor)



# Correlation
# sampledf = df.sample(n=1000, random_state=42)

# plt.figure(figsize=(10, 6))
# sns.heatmap(sampledf.corr(numeric_only=True), cmap='viridis')
# plt.title("UNSW-NB15 correlation heatmap")
# plt.show()



