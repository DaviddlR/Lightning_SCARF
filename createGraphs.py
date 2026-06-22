import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


{'Analysis': 0, 'Backdoor': 1, 'DoS': 2, 'Exploits': 3, 'Fuzzers': 4, 'Generic': 5, 'Normal': 6, 'Reconnaissance': 7, 'Shellcode': 8, 'Worms': 9}

classes = ['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaissance', 'Shellcode', 'Worms']

classes = [f"Class {i+1}" for i in range(10)]

# Pon aquí los 10 valores medios de F1 para cada setting
s_100 = [0.00, 0.0007, 0.042, 0.6374, 0.3507, 0.9788, 0.765, 0.5289, 0.1659, 0.0111]
s_1   = [0.0033, 0.0038, 0.2287, 0.5784, 0.2848, 0.9778, 0.7501, 0.3889, 0.00, 0.00]
contr = [0.00, 0.00, 0.199, 0.6208, 0.3361, 0.9792, 0.8135, 0.6564, 0.0324, 0.00]
recons = [0.00, 0.00, 0.1839, 0.6043, 0.3171, 0.9770, 0.7845, 0.5639, 0.00, 0.00]

# Creamos el DataFrame
data = {
    'Class': classes * 4,
    'Setting': (['Supervised 100%'] * 10 + 
                ['Supervised 1%'] * 10 + 
                ['Contrastive'] * 10 + 
                ['Reconstruction'] * 10),
    'F1-Score': s_100 + s_1 + contr + recons
}
df = pd.DataFrame(data)




plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Crear el gráfico
ax = sns.barplot(data=df, x='Class', y='F1-Score', hue='Setting', palette='viridis')

# Ajustes para ver mejor los datos cercanos
# Cambia el 0.5 por un valor ligeramente inferior a tu mínimo (ej: 0.3)
plt.ylim(0.00, 1.0) 


# Tamaño de las etiquetas del eje X e Y
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)



plt.title('F1-Score Comparison by Class and Setting', fontsize=20)
plt.ylabel('Average F1-Score', fontsize=17)
plt.xlabel('Classes', fontsize=17)
plt.legend(
    title='Configuration', 
    loc='upper left',       # Posición: 'upper right', 'upper left', 'lower center', etc.
    frameon=True,           # Dibujar caja de fondo
    framealpha=0.8,         # Transparencia del fondo (0 es invisible, 1 es sólido)
    edgecolor='gray',       # Color del borde de la caja
    fontsize='large'
)
plt.tight_layout()

plt.savefig('f1_score_comparison.pdf')  # Guardar la figura con alta resolución

plt.show()





