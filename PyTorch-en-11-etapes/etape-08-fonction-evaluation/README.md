# 📊 Étape 8 : Fonction d'évaluation

## 🎯 Objectif de la fonction d'évaluation

La fonction d'évaluation permet de mesurer les performances du modèle sur des données non vues pendant l'entraînement, sans modifier les paramètres du modèle.

## 💻 Code de la fonction d'évaluation

```python
# Fonction d'évaluation
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
# Désactivation du calcul des gradients
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)
```

## 🔍 Explication détaillée

### 1. Mode Évaluation 

`model.eval()`

- Désactive les couches spécifiques d'entraînement (dropout, batch normalization)
- Prépare le modèle pour l'inférence
- Assure une évaluation cohérente

### 2. Désactivation du calcul des gradients

`with torch.no_grad():`

- Économise de la mémoire
- Accélère le processus d'évaluation
- Empêche toute modification des paramètres

### 3. Calcul de la perte
```python
   predictions = model(X_batch)
   loss = loss_fn(predictions, y_batch)
```
- Propage les données à travers le modèle
- Calcule l'erreur sans rétropropagation

### 4. Accumulation de la perte

`total_loss += loss.item()`

- Accumule la perte pour chaque lot
- Permet de calculer la perte moyenne

### 5. Retour de la perte moyenne

`return total_loss / len(dataloader)`

- Calcule la perte moyenne sur tous les lots

## ⚠️ Points d'attention

- **Cohérence** : Toujours utiliser `model.eval()`
- **No Gradient** : Essentiel pour l'efficacité
- **Métriques variées** : Compléter la perte par d'autres indicateurs

## 🚀 Prochaines étapes

Avec notre fonction d'évaluation définie, nous sommes prêts à passer à l'entraînement complet du modèle et à l'analyse de ses performances.






















