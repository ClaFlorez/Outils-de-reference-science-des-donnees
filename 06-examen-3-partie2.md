### **Examen : 20 Questions pour Comprendre l’Implémentation d’un Réseau de Neurones avec PyTorch**

📌 **Instructions :**  
- **Analysez le code fourni** avant chaque section.  
- **Répondez aux questions** basées sur l’implémentation et le fonctionnement du modèle PyTorch.  
- **Objectif :** Acquérir une compréhension **pratique et approfondie** de la construction d’un réseau de neurones en PyTorch.  



## **📝 Partie 1 : Importation des Bibliothèques**

### **Code :**
```python
import torch
from torch import nn, optim, utils
import numpy as np
import matplotlib.pyplot as plt
```

### **Contexte :**  
Ce segment de code importe **PyTorch**, **NumPy** et **Matplotlib**.  
- **PyTorch** est utilisé pour construire et entraîner le réseau de neurones.  
- **NumPy** sert à manipuler les données numériques.  
- **Matplotlib** permet de visualiser les résultats.  

### **Question 1**  
Pourquoi utilisons-nous PyTorch (`torch`) dans ce projet ?  

 \_A. Pour manipuler facilement des **matrices et tenseurs** et entraîner des modèles de deep learning. \_  
B. Pour améliorer la vitesse d’entraînement d’un modèle scikit-learn.  
C. Pour générer automatiquement des jeux de données simulées.  
D. Pour normaliser les variables d’entrée du modèle.  



## **📝 Partie 2 : Génération des Données Simulées**

### **Code :**
```python
# Génération de données simulées
np.random.seed(42)
torch.manual_seed(42)

# Données : quantité de matière première, temps de production, coût main d'œuvre
X_data = np.random.rand(1000, 3) * 10  # 1000 échantillons avec 3 variables
y_data = 3 * X_data[:, 0] + 2 * X_data[:, 1] + 4 * X_data[:, 2] + np.random.randn(1000) * 2
```

### **Contexte :**  
- On **génère artificiellement des données** pour entraîner le modèle.  
- Chaque échantillon a **3 variables d’entrée** (quantité de matière, temps de production, coût de main-d’œuvre).  
- La **relation entre les variables et la sortie** est linéaire, avec un bruit ajouté pour simuler des variations réelles.  

### **Question 2**  
Pourquoi utilisons-nous `np.random.seed(42)` et `torch.manual_seed(42)` ?  

A. Pour garantir que **les mêmes données sont générées** à chaque exécution du script.  
B. Pour réduire la complexité du modèle.  
C. Pour permettre au modèle de mieux généraliser sur des données inconnues.  
D. Pour améliorer la précision du modèle.  



## **📝 Partie 3 : Séparation des Données**

### **Code :**
```python
# Séparation des données (80% train, 20% test)
train_size = int(0.8 * len(X_data))
X_train, X_test = X_data[:train_size], X_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]
```

### **Contexte :**  
- On **divise les données** en un ensemble d’entraînement (**80%**) et un ensemble de test (**20%**).  
- L’objectif est de **vérifier si le modèle est capable de généraliser** sur des données qu’il n’a jamais vues.  

### **Question 3**  
Pourquoi devons-nous séparer les données en **train (80%) et test (20%)** ?  

A. Pour évaluer si le modèle **généralise bien** sur de nouvelles données.  
B. Pour augmenter la vitesse d’entraînement du modèle.  
C. Pour éviter d’avoir trop de données dans l’ensemble d’entraînement.  
D. Pour ajuster les hyperparamètres du modèle en temps réel.  



## **📝 Partie 4 : Création du Dataset et DataLoader**

### **Code :**
```python
class ProductionDataset(utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ajouter une dimension

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Création des datasets
train_dataset = ProductionDataset(X_train, y_train)
test_dataset = ProductionDataset(X_test, y_test)

# Chargement avec DataLoader
train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=32)
```

### **Contexte :**  
- **`ProductionDataset`** structure les données sous forme de `torch.Tensor`.  
- `DataLoader` charge les données **par batchs** pour faciliter l’entraînement.  
- **`shuffle=True`** mélange les données à chaque epoch pour éviter un biais d’apprentissage.  

### **Question 4**  
Quel est **l’avantage principal** d’utiliser `DataLoader` dans PyTorch ?  

A. Il **charge les données en mémoire par lots** et optimise la gestion des calculs.  
B. Il entraîne automatiquement le modèle à chaque batch.  
C. Il transforme les données en valeurs normalisées.  
D. Il ajuste les poids du modèle en fonction de la perte.  



## **📝 Partie 5 : Définition du Modèle**

### **Code :**
```python
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

# Initialisation du modèle
model = NeuralNetwork()
```

### **Contexte :**  
- Le réseau a **3 couches cachées** avec `ReLU` comme activation.  
- Il prend **3 entrées** et génère **une sortie** (coût estimé).  

### **Question 5**  
Pourquoi utilisons-nous la fonction d’activation `ReLU()` dans ce modèle ?  

A. Pour rendre le modèle **non linéaire** et capturer des relations complexes.  
B. Pour améliorer la vitesse de convergence du modèle.  
C. Pour transformer toutes les valeurs négatives en positives.  
D. Pour réduire le temps d’entraînement.  



## **📝 Partie 6 : Entraînement du Modèle**

### **Code :**
```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
```

### **Contexte :**  
- **Optimiseur `Adam`** met à jour les poids du modèle à chaque itération.  
- **Fonction de perte `MSELoss()`** mesure l’écart entre la prédiction et la réalité.  

### **Question 6**  
Pourquoi utilisons-nous `MSELoss()` comme fonction de coût ?  

A. Parce qu’elle est adaptée aux **problèmes de régression**.  
B. Parce qu’elle mesure l’erreur en classification.  
C. Parce qu’elle corrige automatiquement les erreurs du modèle.  
D. Parce qu’elle est plus rapide que `CrossEntropyLoss()`.  



## **📝 Partie 7 : Fonction d'Entraînement et d'Évaluation**

### **Code :**
```python
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
```

### **Contexte :**  
- `optimizer.zero_grad()` réinitialise les gradients.  
- `loss.backward()` **calcule le gradient**.  
- `optimizer.step()` **met à jour les poids**.  

### **Question 7**  
Que fait `optimizer.step()` dans cette fonction d'entraînement ?  

A. **Met à jour les poids** du modèle.  
B. Calcule la perte actuelle.  
C. Réinitialise les gradients.  
D. Modifie la structure du réseau.  













## **📝 Partie 8 : Évaluation du Modèle**

### **Code :**
```python
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(dataloader)
```

### **Contexte :**  
- `model.eval()` **désactive certaines couches** comme Dropout.  
- `torch.no_grad()` **réduit la consommation mémoire** en **désactivant le calcul des gradients**.  
- On calcule la **perte moyenne sur l’ensemble de test**.

### **Question 8**  
Pourquoi utilisons-nous `with torch.no_grad()` lors de l’évaluation ?  

A. Pour **désactiver le calcul des gradients** et économiser de la mémoire.  
B. Pour empêcher les données de test d’être modifiées.  
C. Pour rendre les prédictions plus rapides.  
D. Pour éviter la mise à jour des poids du modèle.  


## **📝 Partie 9 : Entraînement du Modèle**

### **Code :**
```python
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

### **Contexte :**  
- **On entraîne le modèle pendant 50 epochs**.  
- On stocke **les pertes d’entraînement et de test** pour analyser la convergence.  

### **Question 9**  
Comment pouvons-nous savoir si le modèle **surapprend (overfitting) ?**  

A. Si la **perte d'entraînement est basse** mais que la **perte de test reste élevée**.  
B. Si la perte de test est plus basse que la perte d’entraînement.  
C. Si le modèle atteint une précision de 100% après 10 epochs.  
D. Si la fonction d’activation utilisée est `ReLU()`.  



## **📝 Partie 10 : Visualisation des Performances**

### **Code :**
```python
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training and Testing")
plt.legend()
plt.show()
```

### **Contexte :**  
- On **visualise la courbe d’apprentissage** du modèle.  
- Une **courbe de perte qui diminue progressivement** signifie que l’entraînement se passe bien.  

### **Question 10**  
Que signifie une **courbe de perte d’entraînement qui diminue mais une courbe de test qui stagne ou augmente** ?  

A. Le modèle est **en overfitting**.  
B. Le modèle est **sous-entraîné (underfitting)**.  
C. Le modèle a une bonne généralisation.  
D. Il faut augmenter le taux d’apprentissage (`learning rate`).  



## **📝 Partie 11 : Évaluation Finale du Modèle**

### **Code :**
```python
model.eval()
X_sample = torch.tensor(X_test[:5], dtype=torch.float32)
y_sample = y_test[:5]

predictions = model(X_sample).detach().numpy()
for i, (real, pred) in enumerate(zip(y_sample, predictions)):
    print(f"Sample {i + 1}: Real Value = {real:.2f}, Predicted Value = {pred[0]:.2f}")
```

### **Contexte :**  
- On **compare les prédictions du modèle** avec les valeurs réelles sur 5 exemples.  
- `detach().numpy()` **convertit les tenseurs PyTorch en NumPy** pour affichage.  

### **Question 11**  
Pourquoi utilisons-nous `.detach().numpy()` avant d’afficher les prédictions ?  

A. Pour **convertir les tenseurs PyTorch en tableaux NumPy** et faciliter leur affichage.  
B. Pour accélérer les calculs.  
C. Pour sauvegarder les résultats en mémoire.  
D. Parce que NumPy est plus précis que PyTorch.  



## **📝 Partie 12 : Optimisation du Modèle**

### **Code :**
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### **Contexte :**  
- `lr=0.001` définit le **taux d’apprentissage** (learning rate).  
- Si `lr` est trop **élevé**, le modèle peut **diverger**.  
- Si `lr` est trop **faible**, l’apprentissage sera **trop lent**.  

### **Question 12**  
Que se passe-t-il si le taux d’apprentissage (`lr`) est **trop élevé** ?  

A. Le modèle risque **de ne jamais converger** et d’osciller entre différentes valeurs.  
B. Le modèle apprendra plus vite et sera plus précis.  
C. Le modèle prendra plus de temps pour apprendre.  
D. Le modèle ne pourra jamais faire d’erreur.  



## **📝 Partie 13 : Expérimentation avec un Réseau Profond (5 Couches)**

### **Code :**
```python
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
```

### **Contexte :**  
- **Ce réseau contient 5 couches cachées**, au lieu de 2 précédemment.  
- Il peut **apprendre des relations plus complexes**, mais peut aussi **surajuster (overfitting)**.  

### **Question 13**  
Pourquoi ajouter **plus de couches** dans un réseau de neurones ?  

A. Pour capturer des **relations plus complexes** dans les données.  
B. Pour rendre l’entraînement plus rapide.  
C. Pour éviter complètement le surajustement.  
D. Pour utiliser plus de mémoire GPU.  



## **📝 Partie 14 : Stratégies de Prévention du Surapprentissage**

### **Question 14**  
Quelle **technique** peut aider à **éviter le surajustement** ?  

A. **Ajouter du Dropout** entre les couches cachées.  
B. Augmenter la taille du batch (`batch_size`).  
C. Désactiver `ReLU()`.  
D. Supprimer l’ensemble de test.  



## **📝 Partie 15 : Comparaison avec une Régression Linéaire**

### **Question 15**  
Quelle est la **différence** entre une **régression linéaire classique** et un **réseau de neurones** ?  

A. Un **réseau de neurones** peut **apprendre des relations non linéaires**.  
B. Une **régression linéaire** est plus rapide mais moins flexible.  
C. Un **réseau de neurones** fonctionne mieux avec **des données complexes**.  
D. **Toutes les réponses sont correctes**.  






## **📝 Partie 16 : Choix du Batch Size et Son Impact**  

### **Code :**  
```python
train_loader = utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=64)
```

### **Contexte :**  
- **`batch_size=64`** signifie que les données seront traitées **par lots de 64 échantillons** au lieu de tout charger d’un coup.  
- Un **batch size plus grand** accélère l’apprentissage, mais **demande plus de mémoire GPU**.  
- Un **batch size trop petit** peut produire des **gradients instables**.  

### **Question 16**  
Quel est **l'impact du choix de la taille des batchs (`batch_size`)** pendant l’entraînement du modèle ?  

A. Un **batch size plus grand** permet d’entraîner le modèle **plus rapidement**, mais nécessite **plus de mémoire**.  
B. Un **batch size plus petit** permet d’avoir **des gradients plus précis**, mais ralentit l’entraînement.  
C. Un **batch size trop grand** peut faire **diverger** l’optimisation du modèle.  
D. **Toutes les réponses sont correctes.**  

---

## **📝 Partie 17 : Fonction de Perte et Alternatives**  

### **Code :**  
```python
loss_fn = nn.MSELoss()
```

### **Contexte :**  
- **MSELoss (Mean Squared Error)** est couramment utilisée pour **les problèmes de régression**.  
- Elle **pénalise davantage** les erreurs importantes (grandes différences entre la prédiction et la valeur réelle).  
- D’autres fonctions de perte comme `L1Loss` ou `HuberLoss` peuvent être utilisées selon le contexte.  

### **Question 17**  
Pourquoi utilisons-nous **MSELoss (Mean Squared Error)** comme fonction de perte dans ce modèle de régression ?  

A. Parce que c’est une **fonction adaptée aux problèmes de régression** et qu’elle punit les grandes erreurs.  
B. Parce qu’elle **fonctionne mieux avec des classes discrètes**.  
C. Parce qu’elle force le modèle à **ne prédire que des valeurs positives**.  
D. Parce qu’elle est plus rapide que `CrossEntropyLoss()`.  

---

## **?? Partie 18 : Comparaison des Optimiseurs**  

### **Code :**  
```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### **Contexte :**  
- **Adam** est un optimiseur qui ajuste **automatiquement** le taux d’apprentissage (`lr`) pour chaque paramètre du modèle.  
- Il est souvent préféré à **SGD (Stochastic Gradient Descent)**, car **il converge plus rapidement** et évite les minimas locaux.  
- SGD peut être utilisé avec **momentum** pour améliorer sa stabilité.  

### **Question 18**  
Pourquoi `Adam` est-il souvent préféré à `SGD` (Stochastic Gradient Descent) ?  

A. Parce qu’il ajuste **automatiquement le taux d’apprentissage** pour chaque paramètre du modèle.  
B. Parce qu’il est **moins sensible aux minimas locaux** que `SGD`.  
C. Parce qu’il converge **plus rapidement** dans la plupart des cas.  
D. **Toutes les réponses sont correctes.**  


## **📝 Partie 19 : Gestion des Problèmes de Convergence**  

### **Code :**  
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### **Contexte :**  
- **Un modèle qui ne converge pas** signifie que la perte ne diminue pas correctement.  
- **Causes possibles** :
  1. **Taux d’apprentissage trop élevé** → Oscillations sans convergence.  
  2. **Données insuffisantes** → Pas assez d'exemples pour généraliser.  
  3. **Modèle trop simple** → Il ne peut pas capturer les relations complexes.  

### **Question 19**  
Si un modèle **ne converge pas** (c'est-à-dire que la perte ne diminue pas), quelle peut être la cause principale ?  

A. **Un taux d’apprentissage (`lr`) trop élevé** qui empêche la convergence.  
B. **Un manque de données** pour apprendre correctement.  
C. **Un modèle trop simple** qui ne capture pas bien la relation entre les variables.  
D. **Toutes les réponses sont correctes.**  

---

## **📝 Partie 20 : Sauvegarde et Chargement du Modèle**  

### **Code :**  
```python
torch.save(model.state_dict(), "modele.pth")
```

### **Contexte :**  
- **Après l’entraînement**, on veut **sauvegarder le modèle** pour éviter de devoir le réentraîner.  
- **`state_dict()`** contient **tous les poids du modèle**, qu’on peut recharger plus tard avec `torch.load()`.  
- Cela permet de **restituer le modèle sans devoir recommencer l’entraînement**.  

### **Question 20**  
Comment peut-on **sauvegarder un modèle entraîné** en PyTorch ?  

A. En utilisant `torch.save(model.state_dict(), "modele.pth")`.  
B. En exportant les poids du modèle avec NumPy.  
C. En copiant le code source du modèle.  
D. Il n’est pas possible de sauvegarder un modèle en PyTorch.  

