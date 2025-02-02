# 🚀 Étape 9 : Entraînement du modèle

## 🎯 Objectif de l'entraînement

L'entraînement du modèle est le processus itératif où le modèle apprend à partir des données, en ajustant ses paramètres pour minimiser l'erreur de prédiction.

## 💻 Code de l'entraînement

```python
# Entraînement du modèle
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = train_model(train_loader, model, loss_fn, optimizer)
    test_loss = evaluate_model(test_loader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
```

## 🔍 Explication détaillée

### 1. Initialisation
```python
num_epochs = 50
train_losses = []
test_losses = []
```
- Définit le nombre total d'époques d'entraînement
- Initialise des listes pour stocker les pertes d'entraînement et de test

### 2. Boucle d'entraînement

`for epoch in range(num_epochs):`
- Itère sur le nombre spécifié d'époques

### 3. Entraînement et évaluation
```python
 train_loss = train_model(train_loader, model, loss_fn, optimizer)
 test_loss = evaluate_model(test_loader, model, loss_fn)
```
- Appelle les fonctions d'entraînement et d'évaluation définies précédemment
- Calcule les pertes pour l'ensemble d'entraînement et de test

### 4. Stockage des pertes

```python
train_losses.append(train_loss)
test_losses.append(test_loss)
```
- Enregistre les pertes pour chaque époque

### 5. Affichage des progrès
```python
  print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
```
- Affiche les pertes d'entraînement et de test pour chaque époque

## ⚠️ Points d'attention

- **Nombre d'époques** : Ajuster en fonction de la convergence
- **Surapprentissage** : Surveiller l'écart entre les pertes d'entraînement et de test
- **Sauvegarde du modèle** : Envisager de sauvegarder le meilleur modèle

## 🔄 Variantes et améliorations

- Implémentation d'un early stopping
- Utilisation d'un learning rate scheduler
- Sauvegarde du meilleur modèle basée sur la performance de validation

## 🚀 Prochaines étapes

Après l'entraînement, nous pouvons passer à la visualisation des performances et à l'évaluation finale du modèle.










