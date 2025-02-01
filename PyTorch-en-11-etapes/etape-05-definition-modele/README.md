# 🧠 Étape 5 : Définition du modèle

## 🎯 Objectif de la définition du modèle

La définition du modèle est une étape cruciale dans le processus d'apprentissage profond. Elle détermine l'architecture du réseau neuronal qui sera utilisée pour apprendre à partir de nos données et faire des prédictions.

## 💻 Code de définition du modèle

```python
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
```


## 🔍 Explication détaillée

### 1. Définition de la classe NeuralNetwork
```python
# Définition du modèle
class NeuralNetwork(nn.Module):
```
- Hérite de `nn.Module`, la classe de base pour tous les modules PyTorch.
- Permet de définir une architecture de réseau neuronal personnalisée.

### 2. Méthode d'initialisation
```python
  def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
```
- `super(NeuralNetwork, self).__init__()` : Initialise la classe parent `nn.Module`.
- `nn.Sequential` : Crée une séquence de couches qui seront appliquées dans l'ordre.
- Architecture du réseau :
  1. Première couche linéaire : 3 entrées -> 64 neurones
  2. Fonction d'activation ReLU
  3. Deuxième couche linéaire : 64 -> 32 neurones
  4. Fonction d'activation ReLU
  5. Couche de sortie : 32 -> 1 neurone (pour la régression)

### 3. Méthode forward
```python
 def forward(self, x):
 return self.model(x)
```

- Définit comment les données traversent le réseau.
- Simplement passe l'entrée `x` à travers le modèle séquentiel défini.

### 4. Instanciation du modèle
```python
model = NeuralNetwork()
```
- Crée une instance de notre classe `NeuralNetwork`.

## 🧠 Pourquoi cette architecture ?

- **Couches linéaires** : Permettent au modèle d'apprendre des relations complexes entre les entrées et la sortie.
- **ReLU** : Fonction d'activation non linéaire qui aide à capturer des relations non linéaires dans les données.
- **Taille décroissante** : 64 -> 32 -> 1 permet une réduction progressive de la dimensionnalité.
- **Couche de sortie unique** : Adaptée à notre problème de régression (prédiction d'une seule valeur).

## ⚠️ Points d'attention

- **Nombre de paramètres** : Cette architecture a suffisamment de paramètres pour apprendre, sans être trop complexe.
- **Risque de surapprentissage** : Surveillez les performances sur l'ensemble de test.
- **Initialisation des poids** : PyTorch initialise automatiquement les poids, mais cela peut être personnalisé si nécessaire.

## 🔄 Alternatives possibles

- **Couches supplémentaires** : Pour des problèmes plus complexes.
- **Dropout** : Peut être ajouté pour réduire le surapprentissage.
- **Batch Normalization** : Pour stabiliser l'apprentissage dans les réseaux plus profonds.

## 📈 Prochaines étapes

Avec notre modèle défini, nous sommes prêts à configurer l'optimiseur et la fonction de coût pour commencer l'entraînement.








