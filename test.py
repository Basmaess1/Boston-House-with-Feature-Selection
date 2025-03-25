import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import VarianceThreshold

# 📌 1. Charger le dataset California Housing (similaire au Boston Housing)
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 📌 2. Vérifier les variances de chaque feature
variances = df.var()

# 📌 3. Appliquer Variance Threshold (seuil = 0.01)
threshold = 0.01  # On fixe un seuil arbitraire
selector = VarianceThreshold(threshold=threshold)
df_selected = selector.fit_transform(df)

# 📌 4. Afficher les features supprimées
selected_features = df.columns[selector.get_support()]
removed_features = df.columns[~selector.get_support()]

print("✅ Features conservées :", list(selected_features))
print("❌ Features supprimées :", list(removed_features))

# 📌 5. Visualisation des variances avant sélection
plt.figure(figsize=(10, 5))
plt.bar(df.columns, variances, color='skyblue')
plt.axhline(y=threshold, color='red', linestyle='--', label='Seuil de variance')
plt.xticks(rotation=45)
plt.ylabel("Variance")
plt.title("Variance des Features avant sélection")
plt.legend()
plt.show()

