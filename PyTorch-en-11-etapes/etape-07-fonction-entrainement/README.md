# 🏋️ Étape 7 : Fonction d'entraînement

## 🎯 Objectif de la fonction d'entraînement

La fonction d'entraînement est le cœur du processus d'apprentissage. Elle définit comment le modèle va apprendre à partir des données, en ajustant ses paramètres pour minimiser l'erreur.

## 💻 Code de la fonction d'entraînement
```python
# Fonction d'entraînement
def train_model(dataloader, model, loss_fn, optimizer):
```
## 🔍 Explication détaillée

### 1. Mise en mode entraînement
```python
    model.train()
```
- Active le mode entraînement du modèle
- Active des couches spécifiques comme Dropout
- Prépare le modèle à l'apprentissage

### 2. Initialisation de la perte totale

```python
    total_loss = 0
```
- Variable pour suivre la perte moyenne sur l'ensemble du dataset

### 3. Boucle sur les lots de données
```python
    for X_batch, y_batch in dataloader:
```
- Itère sur les lots de données
- Permet un apprentissage par lots (batch learning)

### 4. Réinitialisation des gradients
 - Itère sur les lots de données
- Permet un apprentissage par lots (batch learning)

### 4. Réinitialisation des gradients
```python
       optimizer.zero_grad()
```
- Réinitialise les gradients à zéro
- Évite l'accumulation des gradients entre les lots

### 5. Passe avant (Forward Pass)

```python
        predictions = model(X_batch)
```
- Calcule les prédictions du modèle
- Propage les données à travers le réseau

### 6. Calcul de la perte

```python
        loss = loss_fn(predictions, y_batch)
```
- Compare les prédictions aux vraies valeurs
- Calcule l'erreur du modèle

### 7. Rétropropagation

```python
        loss.backward()
```
- Calcule les gradients
- Propage l'erreur à travers le réseau
- Détermine comment ajuster les poids

### 8. Mise à jour des paramètres

```python
        optimizer.step()
```
- Ajuste les poids du modèle
- Utilise l'algorithme de l'optimiseur (Adam)

### 9. Accumulation de la perte

```python
        total_loss += loss.item()
```
- Accumule la perte pour chaque lot
- Permet de calculer la perte moyenne

### 10. Retour de la perte moyenne

```python
    return total_loss / len(dataloader)
```
- Calcule la perte moyenne sur tous les lots

## 🧠 Concepts clés

- **Batch Learning** : Apprentissage par lots
- **Gradient Descent** : Descente de gradient
- **Backpropagation** : Rétropropagation de l'erreur

## ⚠️ Points d'attention

- **Taille des lots** : Impact sur la convergence
- **Gradients** : Risque de disparition/explosion
- **Normalisation** : Peut améliorer la stabilité

## 🔄 Variantes possibles

- Ajout de regularization
- Gradient clipping
- Early stopping

## 📈 Exemple de configuration avancée

```python
    def train_model(dataloader, model, loss_fn, optimizer, clip_grad=1.0):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
```
# Gradient clipping
```python

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    total_loss += loss.item()
    return total_loss / len(dataloader)
```


## 🚀 Prochaines étapes

Avec notre fonction d'entraînement définie, nous sommes prêts à créer la fonction d'évaluation et à commencer l'entraînement du modèle.












