# ✂️ Étape 3 : Séparation des données
```python
# Séparation des données
train_size = int(0.8 * len(X_data))
X_train, X_test = X_data[:train_size], X_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

```
## 🔍 Objectif de la séparation

La séparation des données en ensembles d'entraînement et de test est une étape cruciale dans le processus d'apprentissage automatique. Elle nous permet d'évaluer la performance du modèle sur des données qu'il n'a jamais vues pendant l'entraînement.

## 📊 Code de séparation

### Explication détaillée

1. **Calcul de la taille de l'ensemble d'entraînement** :
   - `train_size = int(0.8 * len(X_data))` : Nous utilisons 80% des données pour l'entraînement.
   - La fonction `int()` assure que nous avons un nombre entier d'échantillons.

2. **Séparation des features (X)** :
   - `X_train = X_data[:train_size]` : Les premiers 80% des données vont dans l'ensemble d'entraînement.
   - `X_test = X_data[train_size:]` : Les 20% restants vont dans l'ensemble de test.

3. **Séparation des labels (y)** :
   - `y_train = y_data[:train_size]` : Les labels correspondant aux 80% des données d'entraînement.
   - `y_test = y_data[train_size:]` : Les labels correspondant aux 20% des données de test.

## 🧠 Pourquoi cette approche ?

- **Simplicité** : Cette méthode de séparation est simple et rapide à mettre en œuvre.
- **Préservation de l'ordre** : Si l'ordre des données est important (par exemple, pour des séries temporelles), cette méthode le préserve.
- **Reproductibilité** : En utilisant un indice fixe, nous obtenons toujours la même séparation.

## ⚠️ Points d'attention

- **Représentativité** : Cette méthode suppose que les données sont déjà mélangées aléatoirement. Si ce n'est pas le cas, il faudrait d'abord les mélanger.
- **Stratification** : Pour les problèmes de classification, il peut être préférable d'utiliser une séparation stratifiée pour maintenir la distribution des classes.

## 🔄 Alternatives possibles

- **train_test_split de scikit-learn** : Pour une séparation plus avancée, notamment avec stratification.
- **random_split de PyTorch** : Pour une séparation aléatoire directement sur les datasets PyTorch.

## 📈 Prochaines étapes

Après cette séparation, nous sommes prêts à créer nos datasets PyTorch et à commencer l'entraînement de notre modèle.


