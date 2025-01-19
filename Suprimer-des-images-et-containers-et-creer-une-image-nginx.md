# Outils de référence en science de données.
# Commandes de base Docker - Partie 2 🐳

## Supprimer des images et des conteneurs 🗑️

### 1. **Créer et tester des conteneurs Nginx** :

1. **Créer un conteneur nommé `c1` accessible via `localhost:8080`** :
   ```bash
   docker run -it -d -p 8080:80 --name c1 nginx
   ```
   📝 Cela démarre un conteneur en arrière-plan et redirige le port local `8080` vers le port `80` du conteneur.

2. **Créer un conteneur nommé `c2` accessible via `localhost:8081`** :
   ```bash
   docker run -it -d -p 8081:80 --name c2 nginx
   ```
   📝 Même principe que ci-dessus mais pour un port différent.

3. **Créer un conteneur nommé `c3` accessible via `localhost:8082`** :
   ```bash
   docker run -it -d -p 8082:80 --name c3 nginx
   ```
   📝 Testez les trois conteneurs en accédant aux ports locaux.

---

### 2. **Gestion des conteneurs : démarrage et arrêt**

- **Lister les conteneurs actifs** :
  ```bash
  docker ps
  ```
  📝 Cela montre uniquement les conteneurs actuellement en cours d'exécution.

- **Obtenir uniquement les IDs des conteneurs actifs** :
  ```bash
  docker ps -q
  ```
  📝 Utile pour exécuter des commandes en boucle sur plusieurs conteneurs.

- **Arrêter un conteneur par son ID complet ou partiel** :
  ```bash
  docker stop a69983c1a0d3
  docker stop a69
  ```
  📝 Vous pouvez utiliser le début de l'ID pour simplifier la commande.

- **Arrêter tous les conteneurs actifs par leurs IDs** :
  ```bash
  docker stop $(docker ps -q)
  ```

- **Redémarrer un conteneur spécifique** :
  ```bash
  docker start c1
  docker start c2
  docker start c3
  ```

- **Supprimer un conteneur arrêté** :
  ```bash
  docker rm c1
  ```

- **Supprimer tous les conteneurs** :
  ```bash
  docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)
  ```
  📝 Cette commande arrête et supprime tous les conteneurs, qu'ils soient actifs ou non.

---

### 3. **Supprimer les images Docker restantes**

- **Lister toutes les images disponibles** :
  ```bash
  docker images
  ```

- **Obtenir uniquement les IDs des images** :
  ```bash
  docker images -q
  ```

- **Supprimer toutes les images Docker** :
  ```bash
  docker rmi $(docker images -q)
  ```
  📝 Supprime toutes les images locales. Attention : les images utilisées par des conteneurs actifs ne seront pas supprimées.

---

### 4. **Vérifications finales et nouvelle création**

- **Vérifiez si tout a été supprimé** :
  ```bash
  docker images
  ```

- **Recréez un conteneur pour tester à nouveau** :
  ```bash
  docker run -it -d -p 8080:80 --name c1 nginx
  ```

💡 **Astuce** : Avant de supprimer les conteneurs ou images, utilisez toujours les commandes de liste (`docker ps`, `docker images`) pour éviter de perdre des données importantes. 🎯

