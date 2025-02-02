# 📈 Étape 10 : Visualisation des performances

## 🎯 Objectif de la visualisation

La visualisation des performances permet de comprendre graphiquement l'évolution de l'apprentissage du modèle au fil des époques et d'identifier d'éventuels problèmes comme le surapprentissage.

## 💻 Code de visualisation

```python
# Visualisation des pertes
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training and Testing")
plt.legend()
plt.show()
```

## 🔍 Explication détaillée

### 1. Configuration de la figure
```python
# Visualisation des pertes
plt.figure(figsize=(10, 5))
```
- Crée une nouvelle figure matplotlib
- Définit la taille de la figure à 10x5 pouces

### 2. Traçage des courbes de perte
```python
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")

```
- Trace la courbe de perte d'entraînement
- Trace la courbe de perte de test
- Utilise le nombre d'époques comme axe des x

### 3. Étiquetage des axes et titre
```python
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training and Testing")
```
- Ajoute des étiquettes aux axes x et y
- Définit un titre pour le graphique

### 4. Ajout de la légende
```python
plt.legend()
```
- Ajoute une légende pour distinguer les courbes

### 5. Affichage du graphique

```python
plt.show()
```

- Affiche le graphique

## ⚠️ Points d'attention

- **Échelle des axes** : Vérifier si une échelle logarithmique serait plus appropriée
- **Lissage** : Envisager de lisser les courbes si elles sont trop bruitées
- **Comparaison** : S'assurer que les échelles sont comparables entre train et test

## 🔄 Variantes et améliorations

- Ajouter des marqueurs pour les meilleurs points de performance
- Inclure d'autres métriques comme l'accuracy ou le MAE
- Créer des sous-graphiques pour différentes métriques

## 📊 Exemple d'interprétation

- Si les courbes de train et test divergent, cela peut indiquer un surapprentissage
- Une stagnation des deux courbes peut suggérer un sous-apprentissage ou un besoin d'ajuster le taux d'apprentissage
- Des oscillations importantes peuvent indiquer un taux d'apprentissage trop élevé

## 🚀 Prochaines étapes

Après avoir visualisé les performances, nous pouvons passer à l'évaluation finale du modèle et à l'interprétation des résultats.















