## - Code Complet
---

```python
import torch
from torch import nn, optim, utils, autograd
import numpy as np
import matplotlib.pyplot as plt

# Génération de données simulées
np.random.seed(42)
torch.manual_seed(42)

# Variables : quantité, temps, coût main d'œuvre
X_data = np.random.rand(1000, 3) * 10  # 1000 échantillons avec 3 variables
y_data = 3 * X_data[:, 0] + 2 * X_data[:, 1] + 4 * X_data[:, 2] + np.random.randn(1000) * 2

# Séparation des données
train_size = int(0.8 * len(X_data))
X_train, X_test = X_data[:train_size], X_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Création des datasets
class ProductionDataset(utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ajouter une dimension

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ProductionDataset(X_train, y_train)
test_dataset = ProductionDataset(X_test, y_test)

train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=32)

# Définition du modèle
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = NeuralNetwork()

# Configuration de l'optimiseur et de la fonction de coût
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Fonction d'entraînement
def train_model(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Fonction d'évaluation
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)

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

# Visualisation des pertes
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training and Testing")
plt.legend()
plt.show()

# Évaluation finale
model.eval()
X_sample = torch.tensor(X_test[:5], dtype=torch.float32)
y_sample = y_test[:5]

predictions = model(X_sample).detach().numpy()
for i, (real, pred) in enumerate(zip(y_sample, predictions)):
    print(f"Sample {i + 1}: Real Value = {real:.2f}, Predicted Value = {pred[0]:.2f}")
```


---
## - Points Clés
---

1. **Manipulation des Datasets et DataLoaders** : Charger, normaliser et préparer des données pour l'entraînement.
2. **Définition d'un Modèle** : Créer un modèle avec `nn.Sequential`, ajouter des activations non linéaires.
3. **Fonction de Coût et Optimiseur** : Utiliser des fonctions comme `MSELoss` et optimiser avec `Adam`.
4. **Entraînement et Évaluation** : Implémenter des fonctions d'entraînement et d'évaluation robustes.
5. **Visualisation des Performances** : Utiliser des graphiques pour analyser les pertes et performances du modèle.

---
##  - Résumé 
---

Dans ce chapitre, nous avons exploré en détail l'implémentation d'un réseau de neurones avec PyTorch pour prédire les coûts de production. Les points essentiels abordés sont :

1. **Préparation des Données**
   - Génération de données simulées pour l'entraînement
   - Séparation en ensembles d'entraînement et de test
   - Utilisation des DataLoaders pour une gestion efficace des lots

2. **Architecture du Modèle**
   - Création d'un réseau de neurones multicouches
   - Configuration des couches et des fonctions d'activation
   - Choix des hyperparamètres appropriés

3. **Processus d'Entraînement**
   - Implémentation des fonctions d'entraînement et d'évaluation
   - Utilisation de l'optimiseur Adam et de la fonction de perte MSE
   - Suivi de la progression avec les métriques de perte

4. **Évaluation et Visualisation**
   - Analyse des courbes d'apprentissage
   - Comparaison des performances sur les ensembles d'entraînement et de test
   - Validation des prédictions sur des échantillons spécifiques

Cette approche pratique permet de comprendre les concepts fondamentaux de PyTorch tout en résolvant un problème concret de régression.


