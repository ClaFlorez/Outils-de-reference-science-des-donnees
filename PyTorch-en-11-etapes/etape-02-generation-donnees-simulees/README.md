# Etape 02 Generation Donnees Simulees

```python
# Génération de données simulées
np.random.seed(42)
torch.manual_seed(42)

# Variables : quantité, temps, coût main d'œuvre
X_data = np.random.rand(1000, 3) * 10  # 1000 échantillons avec 3 variables
y_data = 3 * X_data[:, 0] + 2 * X_data[:, 1] + 4 * X_data[:, 2] + np.random.randn(1000) * 2
```
