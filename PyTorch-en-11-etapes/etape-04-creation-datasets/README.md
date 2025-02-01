# 📊 Étape 4 : Création des datasets

## 🎯 Objectif de la création des datasets

La création de datasets personnalisés dans PyTorch est une étape essentielle pour préparer nos données à l'entraînement du modèle. Elle nous permet de définir comment les données seront chargées, transformées et présentées au modèle.

## 💻 Code de création des datasets

```python
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
```

## 🔍 Explication détaillée

### 1. Définition de la classe ProductionDataset

`class ProductionDataset(utils.data.Dataset):` - Hérite de `utils.data.Dataset` pour créer un dataset personnalisé.
- Permet de définir comment les données sont stockées et accessibles.

### 2. Méthode d'initialisation

```python
        def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ajouter une dimension
```

- Convertit les données numpy en tenseurs PyTorch.
- Utilise `dtype=torch.float32` pour assurer la compatibilité avec le modèle.
- `unsqueeze(1)` ajoute une dimension à `y` pour le format (N, 1) requis par PyTorch.

### 3. Méthode __len__
 `  def __len__(self):
    return len(self.X)` 

- Retourne le nombre total d'échantillons dans le dataset.
- Utilisé par PyTorch pour déterminer la taille du dataset.

### 4. Méthode __getitem__
```python
def getitem(self, idx):
return self.X[idx], self.y[idx]
text
```
- Définit comment accéder à un échantillon spécifique.
- Retourne une paire (feature, label) pour un index donné.

### 5. Création des instances de dataset
```python
train_dataset = ProductionDataset(X_train, y_train)
test_dataset = ProductionDataset(X_test, y_test)
```
- Crée des datasets séparés pour l'entraînement et le test.

### 6. Création des DataLoaders
```python
train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = utils.data.DataLoader(test_dataset, batch_size=32)
text
```

---
- `DataLoader` gère le chargement des données par lots.
- `batch_size=32` : Définit la taille des lots.
- `shuffle=True` pour l'entraînement : Mélange les données à chaque époque.

## 🧠 Pourquoi cette approche ?

- **Flexibilité** : Permet de personnaliser le chargement et la préparation des données.
- **Efficacité** : Utilise les fonctionnalités de PyTorch pour un chargement optimisé.
- **Compatibilité** : Assure que les données sont dans le bon format pour le modèle PyTorch.

## ⚠️ Points d'attention

- **Taille des lots** : 32 est une valeur courante, mais peut être ajustée selon les besoins.
- **Mélange des données** : Important pour l'entraînement, pas nécessaire pour le test.
- **Types de données** : Assurez-vous que les types (float32) correspondent aux attentes du modèle.

## 🔄 Alternatives possibles

- **TensorDataset** : Pour des datasets plus simples sans transformation personnalisée.
- **Transformations** : Ajoutez des transformations dans `__getitem__` si nécessaire.

## 📈 Prochaines étapes

Avec nos datasets et DataLoaders prêts, nous pouvons maintenant passer à la définition de notre modèle de réseau neuronal.











