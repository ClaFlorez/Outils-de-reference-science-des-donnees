# Outils de référence en science de données.
# Comment supprimer des conteneurs et des images Docker 🚀

## Supprimer les conteneurs 🛑

1. **Lister les conteneurs en cours d'exécution** :
   ```bash
   docker ps
   ```
   📝 Cette commande affiche uniquement les conteneurs actuellement actifs.

2. **Lister tous les conteneurs (actifs et arrêtés)** :
   ```bash
   docker ps -a
   ```
   📝 Cela montre une liste complète des conteneurs, y compris ceux qui ne sont pas en cours d'exécution.

3. **Obtenir uniquement les IDs des conteneurs** :
   ```bash
   docker ps -a -q
   ```
   📝 Cela renvoie une liste d'IDs des conteneurs sans aucun détail supplémentaire.

4. **Arrêter tous les conteneurs** :
   ```bash
   docker stop $(docker ps -a -q)
   ```
   📝 Cette commande utilise les IDs des conteneurs pour arrêter tous les conteneurs actifs.

5. **Supprimer tous les conteneurs** :
   ```bash
   docker rm $(docker ps -a -q)
   ```
   📝 Après avoir arrêté les conteneurs, cette commande supprime tous les conteneurs listés.

---

## Supprimer les images 🖼️

1. **Lister les images Docker disponibles** :
   ```bash
   docker images
   ```
   📝 Cette commande affiche toutes les images Docker stockées localement avec leurs détails (nom, tag, taille, etc.).

2. **Obtenir uniquement les IDs des images** :
   ```bash
   docker images -q
   ```
   📝 Cela renvoie une liste des IDs des images sans aucun détail supplémentaire.

3. **Supprimer toutes les images Docker** :
   ```bash
   docker rmi $(docker images -q)
   ```
   📝 Cette commande supprime toutes les images en utilisant leurs IDs. Attention : Docker ne supprimera pas les images utilisées par des conteneurs actifs.

---

⚠️ **Attention** : Avant de supprimer, vérifiez que vous n'avez pas besoin des conteneurs ou des images pour éviter toute perte de données ou d'environnement important !

# Commandes de base Docker 🐳

## Liste des commandes et explications ✨

1. **Télécharger une image Docker** :
   ```bash
   docker pull centos
   ```
   📝 Cette commande télécharge l'image officielle `centos` depuis Docker Hub et la stocke localement.

2. **Lister les images Docker disponibles** :
   ```bash
   docker images
   ```
   📝 Cela affiche toutes les images Docker présentes localement avec leurs détails (nom, tag, taille, etc.).

3. **Créer et exécuter un conteneur en arrière-plan** :
   ```bash
   docker run -d --name c1 nginx
   ```
   📝 Cette commande démarre un conteneur nommé `c1` en arrière-plan (`-d`) à partir de l'image `nginx`.

4. **Vérifier les images disponibles** :
   ```bash
   docker images
   ```
   📝 Revoir les images Docker locales après avoir utilisé ou téléchargé une image.

5. **Lister les conteneurs actifs** :
   ```bash
   docker ps
   ```
   📝 Affiche uniquement les conteneurs en cours d'exécution.

6. **Lister tous les conteneurs** :
   ```bash
   docker ps -a
   ```
   📝 Montre tous les conteneurs, y compris ceux qui sont arrêtés.

7. **Arrêter un conteneur actif** :
   ```bash
   docker stop c1
   ```
   📝 Arrête le conteneur nommé `c1` qui était en cours d'exécution.

8. **Supprimer un conteneur** :
   ```bash
   docker rm c1
   ```
   📝 Supprime complètement le conteneur nommé `c1`.

9. **Lister les conteneurs actifs après suppression** :
   ```bash
   docker ps
   ```
   📝 Vérifiez les conteneurs encore actifs après avoir arrêté ou supprimé certains d'entre eux.

10. **Lister tous les conteneurs après suppression** :
    ```bash
    docker ps -a
    ```
    📝 Revoir tous les conteneurs, y compris ceux qui sont arrêtés ou supprimés récemment.

11. **Vérifier les images disponibles après manipulation** :
    ```bash
    docker images
    ```
    📝 Consultez les images locales pour vous assurer qu'aucune image importante n'a été supprimée.

---

💡 **Note** : Ces commandes sont essentielles pour la gestion de base des conteneurs et des images Docker. Elles permettent de manipuler votre environnement Docker simplement et efficacement !

