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

1. **Mode Évaluation** : `model.eval()` désactive les couches spécifiques d'entraînement et prépare le modèle pour l'inférence.

2. **Désactivation du calcul des gradients** : `with torch.no_grad():` économise de la mémoire et accélère l'évaluation.

3. **Calcul de la perte** : Les prédictions sont faites et comparées aux vraies valeurs.

4. **Métriques supplémentaires** : MAE et MAPE sont calculées pour une évaluation plus complète.

5. **Accumulation et moyenne** : Les pertes et métriques sont accumulées et moyennées sur tous les lots.

## ⚠️ Points d'attention

- Toujours utiliser `model.eval()` pour une évaluation cohérente.
- La désactivation des gradients est essentielle pour l'efficacité.
- Utiliser diverses métriques pour une évaluation complète.

## 🔄 Variantes et améliorations possibles

- Ajout de métriques spécifiques au domaine.
- Calcul d'intervalles de confiance.
- Génération de rapports détaillés ou visualisations.

## 🚀 Prochaines étapes

Avec notre fonction d'évaluation définie, nous sommes prêts à passer à l'entraînement complet du modèle et à l'analyse approfondie de ses performances.
