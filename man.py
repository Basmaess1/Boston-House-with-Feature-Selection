import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 📌 1. Charger le dataset California Housing
data = fetch_california_housing()
X = np.array(data.data)
feature_names = np.array(data.feature_names)

# 📌 2. Calculer les variances de chaque feature
variances = np.var(X, axis=0)

# 📌 3. Appliquer le seuil de variance (0.01)
threshold = 0.01
mask = variances > threshold
X_selected = X[:, mask]

# 📌 4. Afficher les features supprimées
selected_features = feature_names[mask]
removed_features = feature_names[~mask]

print("✅ Features conservées :", list(selected_features))
print("❌ Features supprimées :", list(removed_features))

# 📌 5. Visualisation des variances avant sélection
plt.figure(figsize=(10, 5))
plt.bar(feature_names, variances, color='skyblue')
plt.axhline(y=threshold, color='red', linestyle='--', label='Seuil de variance')
plt.xticks(rotation=45)
plt.ylabel("Variance")
plt.title("Variance des Features avant sélection")
plt.legend()
plt.show()
