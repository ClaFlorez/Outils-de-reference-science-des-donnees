# 🔢 Étape 02 : Génération de données simulées

## 🎲 Configuration des seeds aléatoires

### Pourquoi fixer les seeds aléatoires ?
- Garantit la reproductibilité des résultats
- Permet de générer les mêmes nombres aléatoires à chaque exécution
- Essentiel pour la répétabilité scientifique

```python
# Génération de données simulées
np.random.seed(42)
torch.manual_seed(42)

# Variables : quantité, temps, coût main d'œuvre
```
## 🧮 Création des données d'entrée

### Détails de la génération
- Génère 1000 échantillons
- 3 variables par échantillon
- Valeurs entre 0 et 10
- Représente potentiellement :
  1. Quantité de production
  2. Temps de production
  3. Coût de main-d'œuvre
```python
X_data = np.random.rand(1000, 3) * 10  # 1000 échantillons avec 3 variables
```
## 📊 Génération des données de sortie

### Explication de la formule
- Relation linéaire simulée
- Coefficients : 3, 2, 4 pour chaque variable
- Ajout d'un bruit gaussien pour réalisme
- Simule un modèle de coût de production

## 🔍 Caractéristiques des données
- Nombre d'échantillons : 1000
- Nombre de features : 3
- Type : Données de régression
- Objectif : Prédire un coût en fonction de variables d'entrée
```python
y_data = 3 * X_data[:, 0] + 2 * X_data[:, 1] + 4 * X_data[:, 2] + np.random.randn(1000) * 2
```
