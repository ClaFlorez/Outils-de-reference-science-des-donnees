# Etape 01 Importation Bibliotheques

Dans cette étape, nous importons les bibliothèques nécessaires pour notre projet :

- `torch` : La bibliothèque principale de PyTorch pour le deep learning.
- `nn`, `optim`, `utils`, `autograd` de torch : Modules pour la création de réseaux de neurones, l'optimisation, les utilitaires de données et le calcul automatique des gradients.
- `numpy` : Pour les opérations numériques efficaces.
- `matplotlib.pyplot` : Pour la visualisation des données et des résultats.

Ces importations nous fournissent les outils essentiels pour construire, entraîner et évaluer notre modèle de deep learning.

```python
import torch
from torch import nn, optim, utils, autograd
import numpy as np
import matplotlib.pyplot as plt
```
