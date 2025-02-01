# ⚙️ Étape 6 : Configuration de l'optimiseur et de la fonction de coût

```python
# Configuration de l'optimiseur et de la fonction de coût
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
```
## 🎯 Objectif de la configuration

La configuration de l'optimiseur et de la fonction de coût est essentielle pour guider l'apprentissage du modèle. Ces composants déterminent comment le modèle va ajuster ses paramètres pour minimiser l'erreur.

## 💻 Code de configuration
```python
# Configuration de l'optimiseur et de la fonction de coût
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
```

## 🔍 Explication détaillée

### 1. Choix de l'optimiseur Adam

#### Caractéristiques d'Adam
- **Algorithme adaptatif** : Ajuste le taux d'apprentissage pour chaque paramètre
- **Combinaison de RMSprop et momentum**
- Performant sur un large éventail de problèmes
- Gère bien les gradients de différentes échelles

#### Paramètres
- `model.parameters()` : Tous les paramètres entraînables du modèle
- `lr=0.01` : Taux d'apprentissage
  - Trop grand : Risque de divergence
  - Trop petit : Apprentissage très lent

### 2. Fonction de coût MSE (Erreur Quadratique Moyenne)

#### Pourquoi Mean Squared Error ?
- Adapté aux problèmes de régression
- Pénalise fortement les erreurs importantes
- Calcul : moyenne des différences au carré entre prédictions et valeurs réelles

#### Formule mathématique
MSE = (1/n) * Σ(y_pred - y_réel)²


## 🧠 Comparaison avec d'autres optimiseurs

| Optimiseur | Avantages | Inconvénients |
|-----------|-----------|---------------|
| SGD | Simple, contrôle précis | Convergence lente |
| RMSprop | Adaptatif | Moins stable |
| Adam | Adaptatif, stable | Parfois moins précis |

## ⚠️ Points d'attention

- **Taux d'apprentissage** : Peut nécessiter des ajustements
- **Initialisation** : Impacte la convergence
- **Échelle des données** : Influence la performance de l'optimiseur

## 🔄 Techniques avancées

- **Learning Rate Scheduler** : Ajuster dynamiquement le taux d'apprentissage
- **Weight Decay** : Régularisation pour éviter le surapprentissage
- **Gradient Clipping** : Limiter l'explosion des gradients

## 📈 Exemple de configuration avancée

```python
optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-5 # Régularisation L2)
```


## 🚀 Prochaines étapes

Avec l'optimiseur et la fonction de coût configurés, nous sommes prêts à définir nos fonctions d'entraînement et d'évaluation.












